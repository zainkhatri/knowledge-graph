"""Index Claude Code conversation transcripts into the knowledge graph.

Each session (`~/.claude/projects/<dir>/<uuid>.jsonl`) becomes a `chat` node:
its name/understanding are the user's asks (FTS-searchable), its path is the
transcript file, and it's linked (a) under a "Claude Code Chats" hub and
(b) under the nearest existing folder node for the session's cwd — so past
context shows up when you traverse the project it happened in.

Bounded + stdlib-only. Reads only the head of each transcript (even 70MB ones).
"""
import os, glob, json, time

HUB_PATH = "/mnt/nvme/PROMETHEUS/CLAUDE-CHATS"
# candidate box-root node paths to hang the chats hub under (best-effort)
ROOT_CANDS = ["/mnt/nvme/PROMETHEUS", "/srv/mergerfs/PROMETHEUS", "/srv", "/root"]


def _iter_head(path, max_lines=400):
    n = 0
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                n += 1
                if n > max_lines:
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except Exception:
        return


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") == "text")
    return ""


def _summarize(path):
    """Return (cwd, timestamp, [user asks], summary)."""
    cwd = ts = summary = None
    asks = []
    for d in _iter_head(path):
        t = d.get("type")
        if t == "summary" and d.get("summary"):
            summary = summary or d["summary"]
        if cwd is None and d.get("cwd"):
            cwd = d["cwd"]
        if ts is None and d.get("timestamp"):
            ts = d["timestamp"]
        if t == "user" and len(asks) < 6:
            txt = _text(d.get("message", {}).get("content")).strip().replace("\n", " ")
            # skip tool-results / injected command noise
            if txt and not txt.startswith("<") and "tool_result" not in txt[:40] \
               and "[Request interrupted" not in txt:
                asks.append(txt)
        if cwd and ts and len(asks) >= 6:
            break
    return cwd, ts, asks, summary


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
    while p and p != "/":
        nid = f"{box}:{p}"
        if store.get_node(nid):
            return nid
        p = os.path.dirname(p)
    return None


def index_chats(store, projects_root="/root/.claude/projects", box="ARES",
                summarize=True, summary_budget=None):
    """summary_budget: max NEW Ollama summaries to generate this run (None = unlimited).
    Chats over budget are still indexed with the raw-asks fallback and get summarized on a
    later run (fingerprint change-detection). Lets a big backfill spread across nights."""
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

    files = glob.glob(os.path.join(projects_root, "*", "*.jsonl"))
    chats = linked = summarized = 0
    for path in files:
        sid = os.path.splitext(os.path.basename(path))[0]
        cwd, ts, asks, summary = _summarize(path)
        if not asks and not summary:
            continue
        date = _date(path, ts)
        first = (asks[0] if asks else summary) or "chat"
        name = (f"{date} · " if date else "") + first[:70]
        try:
            st = os.stat(path); mt = int(st.st_mtime); fp = f"{mt}:{st.st_size}"
        except Exception:
            mt = None; fp = None
        cid = f"{box}:chat/{sid}"
        und = (summary or " · ".join(asks) or "")[:700]    # fallback: raw asks (searchable now)
        status = "raw"                                       # raw = not yet Ollama-summarized
        prev = store.get_node(cid)
        if prev and fp and prev.get("fingerprint") == fp and prev.get("status") == "live":
            und = prev["understanding"]; status = "live"     # unchanged + already summarized → keep
        elif summarize and (summary_budget is None or summarized < summary_budget):
            from .understanding import summarize_chat
            s = summarize_chat(asks)
            if s:
                und = s; status = "live"; summarized += 1
        store.upsert_node({
            "id": cid, "box": box, "kind": "chat", "path": path, "name": name,
            "understanding": und, "mtime": mt, "fingerprint": fp, "status": status,
            "meta": {"session": sid, "cwd": cwd or "", "turns": len(asks), "asks": asks[:6]},
        })
        store.add_edge(hub_id, cid, "contains")
        a = _anchor(store, box, cwd)
        if a and a != hub_id:
            store.add_edge(a, cid, "contains")
            linked += 1
        chats += 1
    store.db.commit()
    return {"chats": chats, "project_linked": linked, "summarized": summarized, "hub": hub_id}


def summarize_pending(store, budget=500, kinds=("chat", "gpt-chat", "claude-chat")):
    """Generate Ollama summaries for chat/gpt nodes still marked status='raw', from the
    asks stored in meta — so it works for ANY box (ZEUS chats included) without the source
    file. Bounded by budget; run nightly to spread a big backfill across days."""
    import json as _json
    from .understanding import summarize_chat, gpu_on_loan
    if gpu_on_loan():
        return {"summarized": 0, "note": "gpu-on-loan"}
    ph = ",".join("?" * len(kinds))
    rows = store.db.execute(
        f"SELECT id, name, meta FROM nodes WHERE status='raw' AND kind IN ({ph}) LIMIT ?",
        (*kinds, int(budget))).fetchall()
    done = 0
    for r in rows:
        asks = (_json.loads(r["meta"] or "{}")).get("asks") or []
        if not asks:
            continue
        s = summarize_chat(asks)
        if not s:
            continue
        with store.db:
            store.db.execute("UPDATE nodes SET understanding=?, status='live' WHERE id=?", (s, r["id"]))
            store.db.execute("DELETE FROM nodes_fts WHERE id=?", (r["id"],))
            store.db.execute("INSERT INTO nodes_fts(id,name,understanding) VALUES(?,?,?)",
                             (r["id"], r["name"], s))
        done += 1
    remaining = store.db.execute(
        f"SELECT count(*) c FROM nodes WHERE status='raw' AND kind IN ({ph})", kinds).fetchone()["c"]
    return {"summarized": done, "still_raw": remaining}
