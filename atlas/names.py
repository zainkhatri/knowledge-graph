"""Readable session names.

A session with no ai-title is named after its first prompt. When that prompt is a bare
slash command (`/resume`, `/model opus`) or injected skill text, the name says nothing
(495 such names on 2026-09-25, 428 of them `/resume`). pick_title() skips those and
falls back to the summary's first sentence; fix_names() repairs existing nodes.
"""
import re

TITLE_MAX = 90
_SKILL = "base directory for this skill"


def is_noise_prompt(text):
    t = (text or "").strip()
    if not t:
        return True
    if t.lower().startswith(_SKILL):
        return True
    words = t.split()
    # a bare command plus at most one argument; "/Users/..." paths are not commands
    return t.startswith("/") and len(words) <= 2 and "/" not in words[0][1:]


def _first_sentence(text):
    text = " ".join((text or "").split())
    m = re.search(r"(.+?[.!?])(\s|$)", text)
    return (m.group(1) if m else text)[:TITLE_MAX]


def pick_title(asks, summary=None):
    for a in asks or []:
        if not is_noise_prompt(a):
            return " ".join(a.split())[:TITLE_MAX]
    if summary and summary.strip():
        return _first_sentence(summary)
    return " ".join(((asks or [""])[0]).split())[:TITLE_MAX]


def fix_names(store):
    import json as _json
    rows = store.db.execute(
        "SELECT id, name, understanding, status, meta FROM nodes WHERE kind='chat'").fetchall()
    renamed = 0
    for r in rows:
        name = r["name"] or ""
        date, sep, title = name.partition(" · ")
        if not sep:
            date, title = "", name
        if not is_noise_prompt(title):
            continue
        asks = (_json.loads(r["meta"] or "{}")).get("asks") or []
        summary = r["understanding"] if r["status"] == "live" else None
        new_title = pick_title(asks, summary)
        new = (f"{date} · " if date else "") + new_title
        if new == name or is_noise_prompt(new_title):
            continue
        with store.db:
            store.db.execute("UPDATE nodes SET name=? WHERE id=?", (new, r["id"]))
            store.db.execute("DELETE FROM nodes_fts WHERE id=?", (r["id"],))
            store.db.execute("INSERT INTO nodes_fts(id,name,understanding) VALUES(?,?,?)",
                             (r["id"], new, r["understanding"] or ""))
        renamed += 1
    return {"renamed": renamed}
