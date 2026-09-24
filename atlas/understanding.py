import os, json, urllib.request, urllib.error
from .redact import redact

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.20.51:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
GPU_LOAN_FLAG = os.getenv("GPU_LOAN_FLAG",
                          "/mnt/nvme/PROMETHEUS/PROJECTS/ARES-DASHBOARD/.gpu-on-loan")

def gpu_on_loan():
    return os.path.exists(GPU_LOAN_FLAG)

def build_prompt(node, children_understandings):
    meta = node.get("meta") or {}
    ext = ", ".join(f"{k}:{v}" for k, v in
                    sorted((meta.get("ext") or {}).items(), key=lambda x: -x[1])[:6])
    kids = "; ".join((children_understandings or [])[:12])
    return (
        "You are cataloguing a homelab NAS for future search. In THREE to FIVE sentences, state "
        "what this folder is, what it holds, and anything notable (scale, purpose, how it's used). "
        "Mention specific subfolder or file-type names where relevant instead of staying generic. "
        "Be concrete and factual. No preamble, no bullet points.\n"
        f"Folder path: {node['path']}\n"
        f"Name: {node['name']}\n"
        f"Contains: {meta.get('n_dirs', 0)} subfolders, {meta.get('n_files', 0)} files. "
        f"File types: {ext or 'n/a'}\n"
        + (f"Notable subfolders: {kids}\n" if kids else "")
    )

def _http_post(url, data):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())

def build_chat_prompt(asks):
    joined = "\n- ".join(a[:220] for a in (asks or [])[:8])
    return (
        "You are cataloguing a homelab's Claude Code session history. In ONE or TWO sentences, "
        "summarize what this coding/ops session was about — the goal and the main topics worked on. "
        "Be concrete and factual. No preamble, no bullet points, no 'the user'.\n"
        f"The requests made in the session:\n- {joined}\n"
    )

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SUMMARY_MODEL = os.getenv("KG_SUMMARY_MODEL", "google/gemini-2.5-flash-lite")
SESSION_SUMMARY_MAX_CHARS = 1500


class OutOfCredit(Exception):
    """OpenRouter returned 402. Stop the run; do not retry."""


def openrouter_key():
    """OPENROUTER_API_KEY env, else the key in ~/.claude/settings.json (fast-jev-compaction)."""
    k = os.getenv("OPENROUTER_API_KEY")
    if k:
        return k
    for p in ("/root/.claude/settings.json", os.path.expanduser("~/.claude/settings.json")):
        try:
            with open(p) as f:
                k = json.load(f).get("env", {}).get("OPENROUTER_API_KEY")
            if k:
                return k
        except Exception:
            continue
    return None


def _openrouter_post(key, body, timeout=90):
    req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}",
        "X-Title": "atlas homelab-kg"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 402:
            raise OutOfCredit(str(e))
        raise


def llm(prompt, max_tokens=400, http=None):
    """One completion. The prompt is ALWAYS redacted first. OpenRouter when a key exists
    (remote, so the GPU loan does not apply), else local Ollama (skipped under .gpu-on-loan).
    Returns None on failure after 3 tries; raises OutOfCredit on HTTP 402."""
    prompt = redact(prompt)
    key = openrouter_key()
    if key:
        body = {"model": SUMMARY_MODEL, "max_tokens": max_tokens, "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}]}
        caller = http or _openrouter_post
        for _attempt in range(3):
            try:
                resp = caller(key, body)
                return (resp["choices"][0]["message"]["content"] or "").strip() or None
            except OutOfCredit:
                raise
            except Exception:
                continue
        return None
    if gpu_on_loan():
        return None
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt,
                          "stream": False, "options": {"temperature": 0.2}}).encode()
    try:
        resp = _http_post(f"{OLLAMA_HOST}/api/generate", payload)
        return (resp.get("response") or "").strip() or None
    except Exception:
        return None


def build_session_prompt(digest, title=None, cwd=None, box=None):
    return (
        "You are writing the permanent memory entry for one Claude Code session in a homelab "
        "(ARES = Proxmox host + dashboard, ZEUS = NAS + Docker, EROS = GPU + Ollama box, "
        "MAC = laptop). A future agent will search these entries to recover context, so be "
        "dense and specific. Write 3-6 sentences of plain prose, no preamble, no 'the user': "
        "the goal; what was done or decided; concrete names (files, services, hosts, commands, "
        "errors, branches); and how it ended (fixed, shipped, abandoned, open follow-ups). If "
        "the session is an automated one-shot prompt, say so in one sentence.\n"
        f"Box: {box or '?'}  Working dir: {cwd or '?'}  Title: {title or '?'}\n"
        "Transcript (USER = human, CLAUDE = assistant; long sessions are sampled):\n"
        f"{digest}\n"
    )


def summarize_session(digest, title=None, cwd=None, box=None, http=None):
    if not (digest or "").strip():
        return None
    s = llm(build_session_prompt(digest, title, cwd, box), max_tokens=450, http=http)
    return s[:SESSION_SUMMARY_MAX_CHARS] if s else None


def summarize_chat(asks, http=None):
    if not asks:
        return None
    s = llm(build_chat_prompt(asks), max_tokens=200, http=http)
    return s[:400] if s else None

# 800 (was 400): the folder prompt now asks for 3-5 sentences instead of 1-2, so the
# old cap was truncating the richer output mid-sentence.
UNDERSTANDING_MAX_CHARS = 800

def generate(node, children_understandings=None, http=None):
    if node.get("kind") == "vault":
        return "Encrypted vault — contents not indexed."
    if gpu_on_loan():
        return None
    prompt = build_prompt(node, children_understandings or [])
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0.1}}).encode()
    caller = http or _http_post
    try:
        resp = caller(f"{OLLAMA_HOST}/api/generate", payload)
        return ((resp.get("response") or "").strip()[:UNDERSTANDING_MAX_CHARS]) or None
    except Exception:
        return None
