"""Recover sessions whose transcripts are gone from Claude Code's prompt log.

`~/.claude/history.jsonl` records every prompt typed (display, project, sessionId,
timestamp) and is NOT pruned by cleanupPeriodDays, so it outlives transcripts that
Claude Code deleted after 30 days. For each session with no graph node yet, this
builds a `chat` node from its prompts and summarizes it (redacted OpenRouter).
Nodes built from real transcripts are never touched; a later transcript for the
same session replaces the history node (different path → index_chats re-reads).
"""
import json, time

NOTE = "[Only the human prompts of this session survive; the assistant replies were not kept.]"
MAX_ROWS = 2_000_000


def _load(path):
    by_sid = {}
    try:
        f = open(path, "r", errors="ignore")
    except OSError:
        return by_sid
    with f:
        for n, line in enumerate(f):
            if n >= MAX_ROWS:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            sid, ts = r.get("sessionId"), r.get("timestamp")
            if sid and isinstance(ts, (int, float)) and (r.get("display") or "").strip():
                by_sid.setdefault(sid, []).append(r)
    for rows in by_sid.values():
        rows.sort(key=lambda r: r["timestamp"])
    return by_sid


def index_history(store, path, box="ARES", summarize=True, summary_budget=None,
                  min_idle=900, workers=8, now=None, embed_fn=None):
    from .chats import _ensure_hub, _anchor, _sample, _summarize_batch
    from .transcript import digest as make_digest
    from .names import pick_title
    from . import understanding as U
    from . import embeddings as E
    embed_fn = embed_fn or E.embed
    now = now if now is not None else time.time()
    hub_id = _ensure_hub(store, box)
    added = unchanged = 0
    todo = []
    for sid, rows in _load(path).items():
        cid = f"{box}:chat/{sid}"
        prev = store.get_node(cid)
        src = f"{path}#{sid}"
        if prev and prev.get("path") != src:
            continue                                   # has a real transcript node → leave it
        last = rows[-1]["timestamp"] / 1000
        fp = f"h:{len(rows)}:{int(last)}"
        fresh = bool(prev) and prev.get("status") == "live"
        if prev and prev.get("fingerprint") == fp and (fresh or not summarize):
            unchanged += 1
            continue
        asks = [r["display"].strip().replace("\n", " ") for r in rows]
        cwd = rows[0].get("project") or ""
        date = time.strftime("%Y-%m-%d", time.localtime(rows[0]["timestamp"] / 1000))
        sampled = _sample(asks)
        gen = ((prev or {}).get("meta") or {}).get("gen_title") or ""
        node = {
            "id": cid, "box": box, "kind": "chat", "path": src,
            "name": f"{date} · {gen or pick_title(asks)}",
            "understanding": (prev["understanding"] if fresh else
                              " · ".join(a[:200] for a in sampled)[:1500]),
            "mtime": int(last), "fingerprint": fp, "status": "raw",
            "meta": {"session": sid, "cwd": cwd, "turns": len(asks), "asks": sampled,
                     "title": "", "source": "history", "gen_title": gen,
                     "last_ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(last))},
            "embedding": prev.get("embedding") if prev else None,
        }
        store.upsert_node(node)
        store.add_edge(hub_id, cid, "contains")
        a = _anchor(store, box, cwd)
        if a and a != hub_id:
            store.add_edge(a, cid, "contains")
        added += 0 if prev else 1
        if summarize and (now - last) >= min_idle and \
                (summary_budget is None or len(todo) < summary_budget):
            dig = NOTE + "\n" + make_digest([("USER", x[:700]) for x in asks])
            todo.append((node, dig, {"title": None, "cwd": cwd}))
    store.db.commit()
    summarized, broke = _summarize_batch(store, todo, box, U, embed_fn, workers)
    return {"added": added, "unchanged": unchanged, "summarized": summarized,
            "queued": len(todo), "out_of_credit": broke}
