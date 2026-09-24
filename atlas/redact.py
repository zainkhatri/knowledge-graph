"""Strip secrets from text before it leaves the box (OpenRouter summaries).

Pattern-based and deliberately greedy on anything credential-shaped, while
keeping paths, hex hashes, UUIDs and ordinary prose intact. Stdlib-only.
"""
import re

R = "[REDACTED]"

_PEM = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END [A-Z ]*PRIVATE KEY-----|$)", re.S)
_TOKENS = re.compile(
    r"\b(?:sk-(?:or-|ant-|proj-)?[A-Za-z0-9_\-]{16,}"      # OpenAI / OpenRouter / Anthropic
    r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[abprs]-[A-Za-z0-9\-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_\-]{30,}"
    r"|eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})")
_BEARER = re.compile(r"(?i)\b(bearer|token)\s+[A-Za-z0-9._~+/=\-]{12,}")
_URL_CRED = re.compile(r"\b([a-z][a-z0-9+.\-]*://[^\s:/@]+:)[^\s@/]+@")
_ASSIGN = re.compile(
    r"""(?ix)
    (\b[A-Z0-9_.\-]*(?:key|token|secret|passw(?:or)?d?|pwd|cred(?:ential)?s?|auth)[A-Z0-9_.\-]*
     ["']?\s*[:=]\s*["']?)
    ([^\s"',;}]{4,})""")
_WORD = re.compile(r"[A-Za-z0-9_\-+/=]{32,}")


def _entropy_word(m):
    w = m.group(0)
    if "/" in w and not w.endswith("="):          # looks like a path
        return w
    if any(c.isupper() for c in w) and any(c.islower() for c in w) and any(c.isdigit() for c in w):
        return R
    return w


def redact(text):
    if not text:
        return ""
    t = _PEM.sub(R, text)
    t = _TOKENS.sub(R, t)
    t = _BEARER.sub(lambda m: f"{m.group(1)} {R}", t)
    t = _URL_CRED.sub(lambda m: f"{m.group(1)}{R}@", t)
    t = _ASSIGN.sub(lambda m: m.group(1) + (m.group(2) if m.group(2) == R else R), t)
    t = _WORD.sub(_entropy_word, t)
    return t
