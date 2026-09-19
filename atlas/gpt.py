"""Index a ChatGPT export archive into the knowledge graph.

Each conversation in `PERSONAL/GPT/conversations-*.json` becomes a `gpt-chat`
node: name = title + date, understanding = the user's questions (FTS-searchable)
or an Ollama summary. Linked under a "ChatGPT Archive" hub. Same progressive
summary budget as the Claude-chat indexer, so a 3.7k-convo backfill spreads
across nights instead of blocking.
"""
import os, glob, json, time

GPT_DIR = "/mnt/nvme/PROMETHEUS/PERSONAL/GPT"


def _messages(convo):
    out = []
    for _, node in (convo.get("mapping") or {}).items():
        msg = (node or {}).get("message")
        if not msg:
            continue
        role = (msg.get("author") or {}).get("role")
        parts = (msg.get("content") or {}).get("parts") or []
        text = " ".join(p for p in parts if isinstance(p, str)).strip()
        if role and text:
            out.append((msg.get("create_time") or 0, role, text))
    out.sort(key=lambda x: x[0] or 0)
    return out


def _asks(msgs, k=6):
    return [t.replace("\n", " ") for (_, role, t) in msgs if role == "user"][:k]


def index_gpt(store, gpt_dir=GPT_DIR, box="ARES", summarize=True, summary_budget=None):
    hub_id = f"{box}:gpt"
    store.upsert_node({
        "id": hub_id, "box": box, "kind": "folder", "path": f"/{box}/CHATGPT-ARCHIVE",
        "name": "ChatGPT Archive",
        "understanding": "Indexed ChatGPT conversation history — searchable across the whole archive.",
    })
    if store.get_node(f"{box}:{gpt_dir}"):
        store.add_edge(f"{box}:{gpt_dir}", hub_id, "contains")

    gpt = summarized = 0
    for fp in sorted(glob.glob(os.path.join(gpt_dir, "conversations-*.json"))):
        try:
            with open(fp, errors="ignore") as f:
                data = json.load(f)
        except Exception:
            continue
        convos = data if isinstance(data, list) else data.get("conversations", [])
        for convo in convos:
            cid = convo.get("conversation_id") or convo.get("id")
            if not cid:
                continue
            msgs = _messages(convo)
            asks = _asks(msgs)
            title = (convo.get("title") or (asks[0][:60] if asks else "conversation")).strip()
            if not asks and not title:
                continue
            ct = convo.get("create_time") or (msgs[0][0] if msgs else None)
            date = time.strftime("%Y-%m-%d", time.localtime(ct)) if ct else ""
            nid = f"{box}:gpt/{cid}"
            name = (f"{date} · " if date else "") + title[:70]
            und = (title + (" · " + " · ".join(asks) if asks else ""))[:700]   # fallback
            fpr = f"n{len(msgs)}"                                              # msg-count = change signal
            status = "raw"
            emb = None
            prev = store.get_node(nid)
            if prev and prev.get("fingerprint") == fpr and prev.get("status") == "live":
                und = prev["understanding"]; status = "live"
                emb = prev.get("embedding")
            elif summarize and (summary_budget is None or summarized < summary_budget):
                from .understanding import summarize_chat
                s = summarize_chat([title] + asks)
                if s:
                    und = s; status = "live"; summarized += 1
            store.upsert_node({
                "id": nid, "box": box, "kind": "gpt-chat",
                "path": os.path.join(gpt_dir, str(cid)), "name": name,
                "understanding": und, "mtime": int(ct) if ct else None,
                "fingerprint": fpr, "status": status,
                "meta": {"title": title, "turns": len(asks), "asks": ([title] + asks)[:6]},
                "embedding": emb,
            })
            store.add_edge(hub_id, nid, "contains")
            gpt += 1
    store.db.commit()
    return {"gpt_chats": gpt, "summarized": summarized, "hub": hub_id}
