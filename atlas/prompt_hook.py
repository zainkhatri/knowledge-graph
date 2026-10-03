"""UserPromptSubmit hook: search ATLAS with a session's FIRST prompt and put the matching
past work into context before Claude touches a file.

Why: on 2026-10-03 only 27% of ARES sessions (38/142 over 7 days) searched ATLAS before
their first Grep/Read/Bash, though CLAUDE.md asks for it. This makes "check ATLAS first"
automatic. Silent when nothing relevant matches (a hit must share 2+ content words with
the prompt) so unrelated chats get no noise. Logs each search to <state>/log.jsonl, which
ares-code's atlas_stats op counts as "checked first".
"""
import concurrent.futures
import math
import json
import os
import re
import time
from pathlib import Path

QUERY_MAX = 300
HITS_MAX = 5
OTHER_MAX = 2            # folders/files: at most this many — past sessions are the point
CHAT_KINDS = ("chat", "gpt-chat", "claude-chat")
SKIP_KINDS = frozenset({"file-content"})   # extracted docs match long prompts by sheer length
OVERLAP_FRAC = 0.3
SEARCH_POOL = 12
CLIP = 260
EMBED_TIMEOUT_S = 2.5
MIN_TOKENS = 2
STOPWORDS = frozenset("""
a about after again all also am an and any are as at be been but by can cant could did dont
do does doing done for from get got had has have how i if im in into is isnt it its just know
let like make me more my need no not now of off ok okay on one or our out please really see
should so some still than that thats the their them then there these they thing things this
those to too up us use very want was we were what when where which while who why will with
would yeah yes you your fix help work working check look go going
without using use used tool tools list given give tell show just answer any ids stop
then first again same new old way thanks thank
""".split())
_PASTED = re.compile(r"</?pasted_content[^>]*>")
_WORD = re.compile(r"[a-z0-9][a-z0-9_.-]*[a-z0-9]|[a-z0-9]")


def query_text(prompt: str) -> str:
    """The searchable part of a prompt: no leading /command, no paste wrapper tags, clipped."""
    text = _PASTED.sub(" ", prompt or "").strip()
    if text.startswith("/"):
        text = text.split(None, 1)[1] if len(text.split(None, 1)) > 1 else ""
    return " ".join(text.split())[:QUERY_MAX]


def content_tokens(text: str) -> list[str]:
    out: list[str] = []
    for w in _WORD.findall((text or "").lower()):
        if len(w) >= 3 and w not in STOPWORDS and w not in out:
            out.append(w)
    return out


def worth_searching(text: str) -> bool:
    return len(content_tokens(query_text(text))) >= MIN_TOKENS


def relevant(hit: dict, tokens: list[str], session_id: str) -> bool:
    """Shares enough content words with the prompt (30% of them, at least MIN_TOKENS),
    is a session or folder, and isn't this very session."""
    hid = str(hit.get("id") or "")
    if hit.get("kind") in SKIP_KINDS or (session_id and hid.endswith("/" + session_id)):
        return False
    text = f"{hit.get('name') or ''} {hit.get('understanding') or ''}".lower()
    need = min(len(tokens), max(MIN_TOKENS, math.ceil(len(tokens) * OVERLAP_FRAC)))
    return sum(1 for t in tokens if t in text) >= need


def first_prompt(state_dir: Path, session_id: str) -> bool:
    """True exactly once per session (atomic marker file)."""
    seen = Path(state_dir) / "seen"
    seen.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(seen / re.sub(r"[^A-Za-z0-9_-]", "", session_id or "none"),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def _bounded_embed(embed):
    """Semantic search needs Ollama; cold it took 2.5 s. Past the budget, go keyword-only."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def run(q):
        try:
            return pool.submit(embed, q).result(timeout=EMBED_TIMEOUT_S)
        except Exception:
            return None
    return run


def _clip(text: str) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= CLIP else text[:CLIP].rstrip() + "…"


def _key(hit: dict) -> str:
    """One session synced to several boxes (ARES:chat/X, ZEUS:chat/X) is one result."""
    hid = str(hit.get("id") or "")
    return hid.split(":", 1)[-1] if hit.get("kind") in CHAT_KINDS else hid


def pick(hits: list[dict]) -> list[dict]:
    """Past sessions first (search order kept, one per session), then up to OTHER_MAX folders."""
    seen: set[str] = set()
    unique = []
    for h in hits:
        if _key(h) not in seen:
            seen.add(_key(h))
            unique.append(h)
    hits = unique
    chats = [h for h in hits if h.get("kind") in CHAT_KINDS]
    other = [h for h in hits if h.get("kind") not in CHAT_KINDS][:OTHER_MAX]
    return (chats + other)[:HITS_MAX]


def render(hits: list[dict]) -> str:
    lines = ["ATLAS was searched automatically with this session's first prompt (homelab-kg). "
             "Past work that matches — read it before exploring files; kg_get <id> for the full "
             "summary. Ignore anything unrelated, and run kg_search yourself if these miss:"]
    for h in hits:
        where = f" [{h['path']}]" if h.get("path") and h.get("kind") not in CHAT_KINDS else ""
        lines.append(f"- {h.get('name')} — {h.get('id')}{where}\n  {_clip(h.get('understanding'))}")
    return "\n".join(lines)


def run(inp: dict, db: str, state_dir: Path, embed=None) -> dict | None:
    sid = str(inp.get("session_id") or "")
    text = query_text(str(inp.get("prompt") or ""))
    tokens = content_tokens(text)
    if len(tokens) < MIN_TOKENS or not first_prompt(state_dir, sid):
        return None
    from .store import Store
    if embed is None:
        from . import embeddings as E
        embed = E.embed
    rows = Store(db).search(" ".join(tokens), limit=SEARCH_POOL * 3, embed_fn=_bounded_embed(embed))
    hits = pick([r for r in rows if relevant(r, tokens, sid)])
    state = Path(state_dir)
    with (state / "log.jsonl").open("a") as log:
        log.write(json.dumps({"session": sid, "ts": time.time(), "hits": len(hits)}) + "\n")
    if not hits:
        return None
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": render(hits)}}
