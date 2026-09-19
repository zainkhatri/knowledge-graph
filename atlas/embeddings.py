import os, json, urllib.request

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.20.51:11434")
EMBED_MODEL = os.getenv("KG_EMBED_MODEL", "nomic-embed-text")
GPU_LOAN_FLAG = os.getenv("GPU_LOAN_FLAG",
                          "/mnt/nvme/PROMETHEUS/PROJECTS/ARES-DASHBOARD/.gpu-on-loan")

def gpu_on_loan():
    return os.path.exists(GPU_LOAN_FLAG)

def _http_post(url, data):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())

def embed(text, http=None):
    """Returns a 768-dim embedding as list[float], or None on any failure:
    empty/whitespace-only text, GPU on loan, Ollama unreachable, timeout, or
    a malformed response. Never raises — same fail-soft contract as
    understanding.generate()."""
    if not (text or "").strip():
        return None
    if gpu_on_loan():
        return None
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    caller = http or _http_post
    try:
        resp = caller(f"{OLLAMA_HOST}/api/embeddings", payload)
        vec = resp.get("embedding")
        return vec if vec else None
    except Exception:
        return None
