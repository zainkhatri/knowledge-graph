"""UserPromptSubmit hook: search ATLAS with a session's FIRST prompt and put the matching
past work into context before Claude touches a file.

Why: on 2026-10-03 only 27% of ARES sessions (38/142 over 7 days) searched ATLAS before
their first Grep/Read/Bash, though CLAUDE.md asks for it. This makes "check ATLAS first"
automatic. Silent when nothing relevant matches (a hit must share 2+ content words with
the prompt) so unrelated chats get no noise. Logs each search to <state>/log.jsonl, which
ares-code's atlas_stats op counts as "checked first".
"""
import threading
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
SHORT_PROMPT = 6          # up to this many content words: all but one must match
PREFIX_MIN = 4            # "photo" matches "photos"; "add" must not match "address"
# Jev (OpenRouter decisions model) as a junk filter over the word-matched candidates.
# Measured 2026-10-03 on 5 real prompts: 0.1-0.2 s and ~2.3k input tokens per call; it
# rejected every off-topic ZEUS cold email (0.08-0.11) but is noisy on fine relevance, so
# it only DROPS clear junk (< JEV_DROP) and never ranks. Any failure → word match alone.
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"
JEV_DROP = 0.3
JEV_TIMEOUT_S = 2.0
JEV_CANDIDATES = 10
SEARCH_POOL = 12
CLIP = 260
EMBED_TIMEOUT_S = 1.0
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
    words = set(_WORD.findall(f"{hit.get('name') or ''} {hit.get('understanding') or ''}".lower()))
    need = min(len(tokens), max(MIN_TOKENS, math.ceil(len(tokens) * OVERLAP_FRAC)))
    if len(tokens) <= SHORT_PROMPT:
        need = max(need, len(tokens) - 1)   # "add the hook to zeus" must not match cold-email "hooks" on ZEUS
    return sum(1 for t in tokens if _has_word(words, t)) >= need


def _has_word(words: set[str], token: str) -> bool:
    if token in words:
        return True
    return len(token) >= PREFIX_MIN and any(w.startswith(token) for w in words)


def _jev_post(url: str, body: bytes, headers: dict, timeout: float) -> dict:
    import urllib.request
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def jev_filter(prompt: str, hits: list[dict], key: str | None, http=None) -> list[dict] | None:
    """Drops candidates Jev scores below JEV_DROP. None = Jev unavailable (use hits as is)."""
    if not hits:
        return []
    if not key:
        return None
    state = {"user_request": prompt,
             "candidates": {f"c{i}": f"{h.get('name')}: {_clip(h.get('understanding'))}" for i, h in enumerate(hits)}}
    questions = {f"c{i}": {"type": "noul",
                           "instructions": f"Would reading candidate c{i} (past work from the user's homelab notes) help with the user_request?",
                           "criteria": {"true": "It is about the same system, problem or task as the request",
                                        "false": "It is unrelated or only shares generic words"}}
                 for i in range(len(hits))}
    body = json.dumps({"model": JEV_MODEL, "state": state, "questions": questions}).encode()
    try:
        answers = (http or _jev_post)(JEV_URL, body, {"Authorization": f"Bearer {key}",
                                                      "Content-Type": "application/json"},
                                      JEV_TIMEOUT_S)["answers"]
        return [h for i, h in enumerate(hits) if float(answers[f"c{i}"]["noul"]) >= JEV_DROP]
    except Exception:
        return None


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


def _bounded_embed(embed, timeout: float = EMBED_TIMEOUT_S):
    """Semantic search needs Ollama: 0.3 s when it is free, but 45-60 s queued behind other
    Ollama work (measured 2026-10-05). Past the budget, go keyword-only. The call runs on a
    DAEMON thread: a ThreadPoolExecutor worker is joined at interpreter exit, so the hook
    process lived until Ollama answered and Claude sat on its 6 s hook timeout every time."""
    def run(q):
        box: list = []
        t = threading.Thread(target=lambda: box.append(embed(q)), daemon=True)
        t.start()
        t.join(timeout)
        return box[0] if box else None
    return run


def _clip(text: str) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= CLIP else text[:CLIP].rstrip() + "…"


def _key(hit: dict) -> str:
    """One session synced to several boxes (ARES:chat/X, ZEUS:chat/X) is one result."""
    hid = str(hit.get("id") or "")
    return hid.split(":", 1)[-1] if hit.get("kind") in CHAT_KINDS else hid


def pick(hits: list[dict]) -> list[dict]:
    """pick_all capped at HITS_MAX."""
    return pick_all(hits)[:HITS_MAX]


def pick_all(hits: list[dict]) -> list[dict]:
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
    return chats + other


def render(hits: list[dict]) -> str:
    lines = ["ATLAS was searched automatically with this session's first prompt (homelab-kg). "
             "Past work that matches — read it before exploring files; kg_get <id> for the full "
             "summary. Ignore anything unrelated, and run kg_search yourself if these miss:"]
    for h in hits:
        where = f" [{h['path']}]" if h.get("path") and h.get("kind") not in CHAT_KINDS else ""
        lines.append(f"- {h.get('name')} — {h.get('id')}{where}\n  {_clip(h.get('understanding'))}")
    return "\n".join(lines)


def _openrouter_key() -> str | None:
    try:
        from .understanding import openrouter_key
        return openrouter_key()
    except Exception:
        return None


def run(inp: dict, db: str, state_dir: Path, embed=None, jev_key=None, jev_http=None) -> dict | None:
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
    candidates = pick_all([r for r in rows if relevant(r, tokens, sid)])[:JEV_CANDIDATES]
    key = jev_key if jev_key is not None else _openrouter_key()
    judged = jev_filter(text, candidates, key, http=jev_http)
    hits = pick(candidates if judged is None else judged)
    state = Path(state_dir)
    with (state / "log.jsonl").open("a") as log:
        log.write(json.dumps({"session": sid, "ts": time.time(), "hits": len(hits),
                              "jev": "off" if judged is None else f"{len(judged)}/{len(candidates)}"}) + "\n")
    if not hits:
        return None
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": render(hits)}}
