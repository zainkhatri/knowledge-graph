import os, json, urllib.request

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
        "You are cataloguing a homelab NAS. In ONE or TWO sentences, state what this folder is "
        "and what it holds. Be concrete and factual. No preamble, no bullet points.\n"
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
        return ((resp.get("response") or "").strip()[:400]) or None
    except Exception:
        return None
