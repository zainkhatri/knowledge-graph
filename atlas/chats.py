"""Index Claude Code conversation transcripts into the knowledge graph.

Each session (`<projects_root>/<dir>/<uuid>.jsonl`) becomes a `chat` node. The
source is normally the permanent archive (PERSONAL/CLAUDE-CODE-SESSIONS/<SRC>),
filled by kg-sync-sessions.sh, so a session keeps its node even after Claude Code
or a box deletes the original. The WHOLE transcript is read (atlas.transcript);
its understanding is an OpenRouter summary of a redacted digest (raw title+asks
until then). Linked under a per-box "Claude Code Chats" hub and under the
nearest folder node for the session's cwd.
"""
import os, glob, time
from concurrent.futures import ThreadPoolExecutor

HUB_PATH = "/mnt/nvme/PROMETHEUS/CLAUDE-CHATS"
# candidate box-root node paths to hang the chats hub under (best-effort)
ROOT_CANDS = ["/mnt/nvme/PROMETHEUS", "/srv/mergerfs/PROMETHEUS", "/srv", "/root"]
MAX_FILES = 200_000
SUMMARY_VERSION = 2          # bump to force every session to be re-summarized once


def _date(path, ts):
    if ts:
        return str(ts)[:10]
    try:
        return time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(path)))
    except Exception:
        return ""


def _anchor(store, box, cwd):
    """Nearest existing folder node id walking up the session cwd."""
    if not cwd:
        return None
    p = cwd.rstrip("/")
    for _ in range(64):
        if not p or p == "/":
            return None
        nid = f"{box}:{p}"
        if store.get_node(nid):
            return nid
        p = os.path.dirname(p)
    return None


def _ensure_hub(store, box):
    hub_id = f"{box}:chats"
    store.upsert_node({
        "id": hub_id, "box": box, "kind": "folder", "path": f"/{box}/CLAUDE-CHATS",
        "name": f"Claude Code Chats ({box})",
        "understanding": "Indexed Claude Code conversation transcripts — searchable session context "
                         f"across every project on {box}.",
    })
    for cand in ROOT_CANDS:
        if store.get_node(f"{box}:{cand}"):
            store.add_edge(f"{box}:{cand}", hub_id, "contains"); break
    return hub_id


def _sample(asks, k=12):
    if len(asks) <= k:
        return list(asks)
    step = max(1, len(asks) // (k - 1))
    return asks[::step][:k - 1] + asks[-1:]


def _node_for(path, box, sid, tr, fp, mt, prev):
    """Build the node dict from a parsed transcript. Keeps a previous summary as the
    understanding (status raw → re-summarized when idle) rather than regressing to asks."""
    asks = tr["asks"]
    sampled = _sample(asks)
    date = _date(path, tr["first_ts"])
    from .names import pick_title
    first = tr["title"] or (pick_title(asks, tr["summary"]) if (asks or tr["summary"]) else "") \
        or f"session {sid[:8]}"
    name = (f"{date} · " if date else "") + first[:90]
    raw = " · ".join(x for x in [tr["title"], tr["summary"]] if x)
    raw = ((raw + " · ") if raw else "") + " · ".join(a[:200] for a in sampled)
    und = prev["understanding"] if prev and prev.get("status") == "live" else raw[:1500]
    sv = (prev.get("meta") or {}).get("sv") if prev else None
    return {
        "id": f"{box}:chat/{sid}", "box": box, "kind": "chat", "path": path, "name": name,
        "understanding": und, "mtime": mt, "fingerprint": fp, "status": "raw",
        "meta": {"session": sid, "cwd": tr["cwd"] or "", "turns": len(asks), "asks": sampled,
                 "title": tr["title"] or "", "last_ts": tr["last_ts"] or "", "sv": sv},
        "embedding": prev.get("embedding") if prev else None,
    }


def index_chats(store, projects_root="/root/.claude/projects", box="ARES",
                summarize=True, summary_budget=None, min_idle=900, workers=8,
                now=None, embed_fn=None):
    """Index every session under projects_root as box. Unchanged files are skipped
    without a re-read. Sessions idle >= min_idle seconds and not yet summarized get an
    OpenRouter summary (max summary_budget this run, `workers` in parallel).
    HTTP 402 stops summarizing; indexing still completes."""
    from .transcript import read as read_transcript, digest as make_digest
    from . import understanding as U
    from . import embeddings as E
    embed_fn = embed_fn or E.embed
    now = now if now is not None else time.time()
    hub_id = _ensure_hub(store, box)

    files = sorted(glob.glob(os.path.join(projects_root, "*", "*.jsonl")))[:MAX_FILES]
    chats = linked = unchanged = 0
    todo = []                                    # (node, digest) awaiting a summary
    for path in files:
        sid = os.path.splitext(os.path.basename(path))[0]
        cid = f"{box}:chat/{sid}"
        try:
            st = os.stat(path); mt = int(st.st_mtime); fp = f"{mt}:{st.st_size}"
        except OSError:
            continue
        chats += 1
        prev = store.get_node(cid)
        same = bool(prev) and prev.get("fingerprint") == fp and prev.get("path") == path
        fresh = bool(prev) and prev.get("status") == "live" and \
            (prev.get("meta") or {}).get("sv") == SUMMARY_VERSION
        idle = (now - mt) >= min_idle
        wants_summary = summarize and idle
        if same and (fresh or not wants_summary):
            unchanged += 1
            continue
        tr = read_transcript(path)
        node = _node_for(path, box, sid, tr, fp, mt, prev)
        if not same:
            store.upsert_node(node)
            store.add_edge(hub_id, cid, "contains")
            a = _anchor(store, box, tr["cwd"])
            if a and a != hub_id:
                store.add_edge(a, cid, "contains"); linked += 1
        if wants_summary and tr["turns"] and \
                (summary_budget is None or len(todo) < summary_budget):
            todo.append((node, make_digest(tr["turns"]), tr))
    store.db.commit()

    summarized, out_of_credit = _summarize_batch(store, todo, box, U, embed_fn, workers)
    return {"chats": chats, "unchanged": unchanged, "project_linked": linked,
            "summarized": summarized, "queued": len(todo), "out_of_credit": out_of_credit,
            "hub": hub_id}


def _summarize_batch(store, todo, box, U, embed_fn, workers):
    """Run summaries in threads (HTTP only); write results on this thread."""
    if not todo:
        return 0, False
    state = {"broke": False}

    def work(item):
        node, dig, tr = item
        if state["broke"]:
            return node, None, None
        try:
            s = U.summarize_session(dig, title=tr["title"], cwd=tr["cwd"], box=box)
        except U.OutOfCredit:
            state["broke"] = True
            return node, None, None
        emb = embed_fn(s) if s else None
        return node, s, emb

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for node, s, emb in ex.map(work, todo):
            if not s:
                continue
            node = dict(node, understanding=s, status="live",
                        meta=dict(node["meta"], sv=SUMMARY_VERSION),
                        embedding=store.vec_to_blob(emb) if emb else node.get("embedding"))
            store.upsert_node(node)
            done += 1
    return done, state["broke"]


def summarize_pending(store, budget=500, kinds=("chat", "gpt-chat", "claude-chat"), embed_fn=None):
    """Summarize chat/gpt nodes still marked status='raw' from the asks in meta, so it
    works for ANY box without the source file. Uses OpenRouter when a key exists (the GPU
    loan does not apply), else local Ollama (skipped under .gpu-on-loan). Bounded by
    budget. Also stores a semantic-search embedding for each new summary."""
    import json as _json
    from . import understanding as U
    from . import embeddings as E
    embed_fn = embed_fn or E.embed
    if not U.openrouter_key() and U.gpu_on_loan():
        return {"summarized": 0, "note": "gpu-on-loan"}
    ph = ",".join("?" * len(kinds))
    rows = store.db.execute(
        f"SELECT id, name, path, meta, kind FROM nodes WHERE status='raw' AND kind IN ({ph})"
        " ORDER BY kind='chat' LIMIT ?", (*kinds, int(budget) * 4)).fetchall()
    done = 0
    for r in rows:
        if done >= budget:
            break
        # a Claude Code chat whose transcript still exists is index_chats' job (whole-file
        # summary once idle); only orphaned ones (source deleted) fall back to their asks
        if r["kind"] == "chat" and r["path"] and os.path.exists(r["path"]):
            continue
        asks = (_json.loads(r["meta"] or "{}")).get("asks") or []
        if not asks:
            continue
        try:
            s = U.summarize_chat(asks)
        except U.OutOfCredit:
            break
        if not s:
            continue
        emb = store.vec_to_blob(embed_fn(s))
        with store.db:
            store.db.execute("UPDATE nodes SET understanding=?, status='live', embedding=? WHERE id=?",
                             (s, emb, r["id"]))
            store.db.execute("DELETE FROM nodes_fts WHERE id=?", (r["id"],))
            store.db.execute("INSERT INTO nodes_fts(id,name,understanding) VALUES(?,?,?)",
                             (r["id"], r["name"], s))
        done += 1
    remaining = store.db.execute(
        f"SELECT count(*) c FROM nodes WHERE status='raw' AND kind IN ({ph})", kinds).fetchone()["c"]
    return {"summarized": done, "still_raw": remaining}
