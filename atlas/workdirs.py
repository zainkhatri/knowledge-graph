"""Attach every chat to a folder, even on boxes whose folders are not indexed.

ARES/EROS chats hang under real folder nodes (walker-indexed). ZEUS and the Macs have
almost none, so their sessions only sat under the per-box chats hub. This:
  1. canonicalizes a session cwd — ZEUS mergerfs branch disks → /srv/mergerfs/PROMETHEUS,
     Mac mounts of ARES/ZEUS (~/PROMETHEUS/ARES, .ARES.a, …) → that box's real path;
  2. links the chat under the existing folder node for that path when there is one;
  3. otherwise creates lightweight `folder` nodes (meta.workdir) for the path and its
     parents up to the nearest existing node, described from the sessions inside
     (no LLM). Idempotent; run after indexing.
"""
import os, re

_ZEUS_BRANCH = re.compile(r"^/srv/dev-disk-by-uuid-[^/]+(?=/|$)")
_MAC_MOUNT = re.compile(r"^/Users/[^/]+/PROMETHEUS/\.?(ARES|ZEUS)(?:\.a)?(?=/|$)")
_ROOTS = {"ARES": "/mnt/nvme/PROMETHEUS", "ZEUS": "/srv/mergerfs/PROMETHEUS"}
MAX_DEPTH = 40


def canonical(box, cwd):
    """(box, path) where a session's cwd really lives."""
    cwd = (cwd or "").rstrip("/")
    if box == "ZEUS" and _ZEUS_BRANCH.match(cwd):
        return "ZEUS", _ZEUS_BRANCH.sub(_ROOTS["ZEUS"], cwd, count=1)
    if box == "MAC":
        m = _MAC_MOUNT.match(cwd)
        if m:
            return m.group(1), _ROOTS[m.group(1)] + cwd[m.end():]
    return box, cwd


def _hub(store, box):
    from .chats import _ensure_hub
    hub = f"{box}:chats"
    return hub if store.get_node(hub) else _ensure_hub(store, box)


def _chain(store, box, path):
    """Create workdir nodes for path and parents up to the nearest existing node.
    Returns (node id for path, created count, whether path already existed)."""
    target = f"{box}:{path}"
    if store.get_node(target):
        return target, 0, True
    chain, anchor, p = [], None, path
    for _ in range(MAX_DEPTH):
        nid = f"{box}:{p}"
        if store.get_node(nid):
            anchor = nid
            break
        chain.append(p)
        parent = os.path.dirname(p)
        if parent in ("", "/", p):
            break
        p = parent
    for q in chain:
        store.upsert_node({"id": f"{box}:{q}", "box": box, "kind": "folder", "path": q,
                           "name": os.path.basename(q) or q, "understanding": "",
                           "meta": {"workdir": True}})
    for child, parent in zip(chain, chain[1:]):
        store.add_edge(f"{box}:{parent}", f"{box}:{child}", "contains")
    if chain:
        store.add_edge(anchor or _hub(store, box), f"{box}:{chain[-1]}", "contains")
    return target, len(chain), False


def _describe(store):
    """Refresh every workdir node's understanding from the sessions/folders inside it."""
    import json as _json
    rows = store.db.execute(
        "SELECT id, box, path, name, understanding FROM nodes "
        "WHERE kind='folder' AND json_extract(meta,'$.workdir')=1").fetchall()
    for r in rows:
        kids = store.db.execute(
            "SELECT n.kind, n.name, n.mtime FROM edges e JOIN nodes n ON n.id=e.dst "
            "WHERE e.src=? AND e.type='contains'", (r["id"],)).fetchall()
        chats = sorted((k for k in kids if k["kind"] == "chat"), key=lambda k: -(k["mtime"] or 0))
        dirs = [k["name"] for k in kids if k["kind"] != "chat"]
        parts = []
        if chats:
            dates = sorted(c["name"][:10] for c in chats if c["name"][:4].isdigit())
            span = f" ({dates[0]} → {dates[-1]})" if dates else ""
            recent = "; ".join(c["name"].split(" · ", 1)[-1][:80] for c in chats[:3])
            parts.append(f"Working directory on {r['box']}: {len(chats)} Claude Code sessions"
                         f"{span}. Recent: {recent}.")
        else:
            parts.append(f"Directory on {r['box']}.")
        if dirs:
            parts.append("Subfolders: " + ", ".join(sorted(dirs)[:12]) + ".")
        und = " ".join(parts)
        if und != (r["understanding"] or ""):
            with store.db:
                store.db.execute("UPDATE nodes SET understanding=? WHERE id=?", (und, r["id"]))
                store.db.execute("DELETE FROM nodes_fts WHERE id=?", (r["id"],))
                store.db.execute("INSERT INTO nodes_fts(id,name,understanding) VALUES(?,?,?)",
                                 (r["id"], r["name"], und))


def link_workdirs(store):
    import json as _json
    chats = store.db.execute(
        "SELECT id, box, meta FROM nodes WHERE kind='chat'").fetchall()
    linked = {r[0] for r in store.db.execute(
        "SELECT DISTINCT dst FROM edges WHERE type='contains' AND dst LIKE '%:chat/%' "
        "AND src NOT LIKE '%:chats'")}
    stats = {"already_linked": 0, "linked_existing": 0, "linked_workdir": 0,
             "workdirs_created": 0, "no_cwd": 0}
    for r in chats:
        meta = _json.loads(r["meta"] or "{}")
        cwd = (meta.get("cwd") or "").rstrip("/")
        if not cwd:
            stats["no_cwd"] += 1
            continue
        box2, path2 = canonical(r["box"], cwd)
        if (box2, path2) != (r["box"], cwd) and meta.get("cwd_canon") != path2:
            with store.db:
                store.db.execute("UPDATE nodes SET meta=json_set(meta,'$.cwd_canon',?) WHERE id=?",
                                 (path2, r["id"]))
        if r["id"] in linked:
            stats["already_linked"] += 1
            continue
        target, created, existed = _chain(store, box2, path2)
        store.add_edge(target, r["id"], "contains")
        stats["workdirs_created"] += created
        stats["linked_existing" if existed else "linked_workdir"] += 1
    store.db.commit()
    _describe(store)
    return stats
