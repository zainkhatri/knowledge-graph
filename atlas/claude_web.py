"""Index a claude.ai data export into the knowledge graph.

claude.ai web conversations are NOT stored locally (unlike Claude Code). Export them
from claude.ai → Settings → Privacy → "Export data"; Anthropic emails a zip containing
`conversations.json` — a list of {uuid, name, created_at, chat_messages:[{sender,text}]}.
Drop that under PERSONAL/CLAUDE (or anywhere) and run `kg index-claude-web --dir <path>`.

Mirrors the GPT indexer: name/understanding = title + your asks (FTS-searchable now),
Ollama summaries fill in via `summarize-pending`.
"""
import os, glob, json, time


def _date(s):
    if not s:
        return ""
    if isinstance(s, (int, float)):
        try:
            return time.strftime("%Y-%m-%d", time.localtime(s))
        except Exception:
            return ""
    return str(s)[:10]                      # ISO "2024-03-15T..." -> "2024-03-15"


def _text(m):
    t = m.get("text")
    if t:
        return t
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    if isinstance(c, str):
        return c
    return ""


def index_claude_web(store, root, box="ARES", summarize=True, summary_budget=None):
    hub_id = f"{box}:claudeweb"
    store.upsert_node({
        "id": hub_id, "box": box, "kind": "folder", "path": f"/{box}/CLAUDE-WEB",
        "name": "Claude.ai Chats",
        "understanding": "Indexed claude.ai web conversation history — searchable across the export.",
    })
    files = glob.glob(os.path.join(root, "**", "conversations*.json"), recursive=True)
    n = summarized = 0
    for fp in files:
        try:
            with open(fp, errors="ignore") as f:
                data = json.load(f)
        except Exception:
            continue
        convos = data if isinstance(data, list) else data.get("conversations", [])
        for convo in convos:
            cid = convo.get("uuid") or convo.get("id")
            if not cid:
                continue
            msgs = convo.get("chat_messages") or convo.get("messages") or []
            asks = [t.replace("\n", " ") for m in msgs
                    if (m.get("sender") or m.get("role")) in ("human", "user")
                    for t in [_text(m)] if t][:6]
            title = (convo.get("name") or (asks[0][:60] if asks else "conversation")).strip()
            if not asks and not title:
                continue
            date = _date(convo.get("created_at"))
            nid = f"{box}:claude/{cid}"
            name = (f"{date} · " if date else "") + title[:70]
            fpr = f"n{len(msgs)}"
            status = "raw"
            emb = None
            csum = (convo.get("summary") or "").strip()      # Claude's export ships its own summary
            und = (csum or title + (" · " + " · ".join(asks) if asks else ""))[:700]
            prev = store.get_node(nid)
            if csum:
                status = "live"                               # already summarized by Claude — no Ollama
                if prev and prev.get("fingerprint") == fpr and prev.get("understanding") == und:
                    emb = prev.get("embedding")                # same csum as last run — keep its embedding
            else:
                if prev and prev.get("fingerprint") == fpr and prev.get("status") == "live":
                    und = prev["understanding"]; status = "live"
                    emb = prev.get("embedding")
                elif summarize and (summary_budget is None or summarized < summary_budget):
                    from .understanding import summarize_chat
                    s = summarize_chat([title] + asks)
                    if s:
                        und = s; status = "live"; summarized += 1
            store.upsert_node({
                "id": nid, "box": box, "kind": "claude-chat",
                "path": os.path.join(root, str(cid)), "name": name,
                "understanding": und, "fingerprint": fpr, "status": status,
                "meta": {"title": title, "turns": len(asks), "asks": ([title] + asks)[:6]},
                "embedding": emb,
            })
            store.add_edge(hub_id, nid, "contains")
            n += 1
    store.db.commit()
    return {"claude_web_chats": n, "summarized": summarized, "hub": hub_id}
