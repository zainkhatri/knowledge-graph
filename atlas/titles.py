"""Short LLM titles for sessions that have no Claude Code ai-title.

Sessions rebuilt from prompt logs, headless `claude -p` runs and some older transcripts
carry no ai-title, so their name is their first prompt, which often describes only how
the session started. This asks the summarizer (redacted OpenRouter, via
understanding.llm) for a 3-8 word title from the existing summary plus a few prompts,
stores it in meta.gen_title and renames the node. Idempotent.
"""
import json, re
from concurrent.futures import ThreadPoolExecutor

TITLE_MAX = 80
_SKIP_CWD = "cc-proxy"          # thousands of near-identical automated classifier calls


def clean_title(text):
    t = " ".join((text or "").split())
    t = re.sub(r"^(title|session title)\s*:\s*", "", t, flags=re.I)
    t = t.strip(" \"'`*#").rstrip(".").strip()
    return t[:TITLE_MAX].rstrip() or None


def _prompt(summary, asks):
    joined = "\n- ".join(a[:200] for a in asks[:4])
    return ("Write a title of 3 to 8 words for this Claude Code session, like a git commit "
            "subject: specific, names the main system or task, no quotes, no trailing period. "
            "Reply with the title only.\n"
            f"Summary: {summary[:1200]}\n"
            f"First requests:\n- {joined}\n")


def gen_titles(store, budget=5000, workers=8):
    from . import understanding as U
    rows = store.db.execute(
        "SELECT id, name, understanding, meta FROM nodes WHERE kind='chat' AND status='live'"
        " AND coalesce(json_extract(meta,'$.title'),'')=''"
        " AND coalesce(json_extract(meta,'$.gen_title'),'')=''"
        " AND coalesce(json_extract(meta,'$.cwd'),'') NOT LIKE ?"
        " AND trim(coalesce(understanding,''))!='' LIMIT ?",
        (f"%{_SKIP_CWD}%", int(budget))).fetchall()
    state = {"broke": False}

    def work(r):
        if state["broke"]:
            return r, None
        asks = (json.loads(r["meta"] or "{}")).get("asks") or []
        try:
            return r, clean_title(U.llm(_prompt(r["understanding"], asks), max_tokens=30))
        except U.OutOfCredit:
            state["broke"] = True
            return r, None

    titled = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for r, title in ex.map(work, rows):
            if not title:
                continue
            date, sep, _ = (r["name"] or "").partition(" · ")
            name = (f"{date} · " if sep else "") + title
            with store.db:
                store.db.execute("UPDATE nodes SET name=?, meta=json_set(meta,'$.gen_title',?)"
                                 " WHERE id=?", (name, title, r["id"]))
                store.db.execute("DELETE FROM nodes_fts WHERE id=?", (r["id"],))
                store.db.execute("INSERT INTO nodes_fts(id,name,understanding) VALUES(?,?,?)",
                                 (r["id"], name, r["understanding"]))
            titled += 1
    return {"titled": titled, "candidates": len(rows), "out_of_credit": state["broke"]}
