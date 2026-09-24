"""Read a WHOLE Claude Code transcript into a bounded digest for summarization.

chats.py used to read only the first 400 lines / 6 asks, so anything a session did
after its opening turns was invisible. This walks every line (bounded) and keeps
all user asks plus short assistant replies, then fits them into `max_chars` by
keeping the head and tail and sampling the middle evenly.

Stdlib-only (it ships to EROS with the rest of the package).
"""
import json

MAX_LINES = 3_000_000          # hard bound; the largest transcript seen is ~70MB / ~10k lines
USER_CLIP = 700
CLAUDE_CLIP = 350


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "text")
    return ""


def _is_noise(txt):
    return (not txt or txt.startswith("<") or "tool_result" in txt[:40]
            or "[Request interrupted" in txt or txt.startswith("Caveat:"))


def read(path, max_lines=MAX_LINES):
    """Return dict(cwd, first_ts, last_ts, title, summary, asks, turns) for one transcript.
    turns = ordered [("USER"|"CLAUDE", text)]. Never raises."""
    out = {"cwd": None, "first_ts": None, "last_ts": None, "title": None,
           "summary": None, "asks": [], "turns": []}
    try:
        f = open(path, "r", errors="ignore")
    except Exception:
        return out
    with f:
        for n, line in enumerate(f):
            if n >= max_lines:
                break
            # cheap prefilter: skip huge tool-result/attachment lines before json parsing
            if '"user"' not in line and '"assistant"' not in line \
               and '"cwd"' not in line and 'itle"' not in line and '"summary"' not in line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("type")
            if out["cwd"] is None and d.get("cwd"):
                out["cwd"] = d["cwd"]
            ts = d.get("timestamp")
            if ts:
                out["first_ts"] = out["first_ts"] or ts
                out["last_ts"] = ts
            if t in ("ai-title", "custom-title"):
                out["title"] = d.get("customTitle") or d.get("aiTitle") or out["title"]
            elif t == "summary" and d.get("summary"):
                out["summary"] = out["summary"] or d["summary"]
            elif t == "user":
                txt = _text(d.get("message", {}).get("content")).strip().replace("\n", " ")
                if not _is_noise(txt):
                    out["asks"].append(txt)
                    out["turns"].append(("USER", txt[:USER_CLIP]))
            elif t == "assistant":
                txt = _text(d.get("message", {}).get("content")).strip().replace("\n", " ")
                if txt:
                    out["turns"].append(("CLAUDE", txt[:CLAUDE_CLIP]))
    return out


def digest(turns, max_chars=24000):
    """Fit ordered turns into max_chars: head + evenly-sampled middle + tail."""
    lines = [f"{who}: {txt}" for who, txt in turns]
    total = sum(len(l) + 1 for l in lines)
    if total <= max_chars:
        return "\n".join(lines)
    budget = max_chars // 3
    head, tail = [], []
    used = 0
    for l in lines:
        if used + len(l) > budget:
            break
        head.append(l); used += len(l) + 1
    used = 0
    for l in reversed(lines[len(head):]):
        if used + len(l) > budget:
            break
        tail.append(l); used += len(l) + 1
    tail.reverse()
    middle = lines[len(head):len(lines) - len(tail)]
    # prefer USER lines in the sampled middle — they carry the direction changes
    users = [l for l in middle if l.startswith("USER:")] or middle
    avg = max(1, sum(len(l) for l in users) // max(1, len(users)))
    k = max(1, budget // (avg + 1))
    step = max(1, len(users) // k)
    sampled, used = [], 0
    for l in users[::step]:
        if used + len(l) > budget:
            break
        sampled.append(l); used += len(l) + 1
    return "\n".join(head + [f"[… {len(middle)} middle turns, sampled …]"] + sampled
                     + ["[…]"] + tail)
