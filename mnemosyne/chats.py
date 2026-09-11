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


def index_chats(store, projects_root="/root/.claude/projects", box="ARES"):
    hub_id = f"{box}:chats"
    store.upsert_node({
        "id": hub_id, "box": box, "kind": "folder", "path": HUB_PATH,
        "name": "Claude Code Chats",
        "understanding": "Indexed Claude Code conversation transcripts — searchable session context "
                         "across every project on this box.",
    })
    root_id = f"{box}:/mnt/nvme/PROMETHEUS"
    if store.get_node(root_id):
        store.add_edge(root_id, hub_id, "contains")

    files = glob.glob(os.path.join(projects_root, "*", "*.jsonl"))
    chats = linked = 0
    for path in files:
        sid = os.path.splitext(os.path.basename(path))[0]
        cwd, ts, asks, summary = _summarize(path)
        if not asks and not summary:
            continue
        date = _date(path, ts)
        first = (asks[0] if asks else summary) or "chat"
        name = (f"{date} · " if date else "") + first[:70]
        und = (summary or " · ".join(asks) or "")[:700]
        try:
            mt = int(os.path.getmtime(path))
        except Exception:
            mt = None
        cid = f"{box}:chat/{sid}"
        store.upsert_node({
            "id": cid, "box": box, "kind": "chat", "path": path, "name": name,
            "understanding": und, "mtime": mt,
            "meta": {"session": sid, "cwd": cwd or "", "turns": len(asks)},
        })
        store.add_edge(hub_id, cid, "contains")
        a = _anchor(store, box, cwd)
        if a and a != hub_id:
            store.add_edge(a, cid, "contains")
            linked += 1
        chats += 1
    store.db.commit()
    return {"chats": chats, "project_linked": linked, "hub": hub_id}
