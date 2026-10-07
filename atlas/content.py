import os, re, hashlib, json, base64, subprocess, shutil
from . import understanding as U
from .walker import IGNORE_DIRS, default_vault_pred, node_id

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
PDF_EXT = {".pdf"}
DOCX_EXT = {".docx"}
TEXT_EXT = {".txt", ".md"}
SUPPORTED_EXT = IMAGE_EXT | PDF_EXT | DOCX_EXT | TEXT_EXT
TEXT_CAP = 200_000  # ~200KB; past this, txt/md content is truncated
OCR_MIN_ALNUM = 8   # tesseract output below this is "no real text" -> try vision fallback
_WORD = re.compile(r"[A-Za-z]{3,}")
_ID = re.compile(r"[A-Za-z0-9]{8,}")
MIN_WORDS = 3       # below this, OCR is glyph noise ("a i <> x"), not searchable text


def has_text(text):
    """True if extracted text is worth a search entry: MIN_WORDS real words, or an
    ID-like token (VIN, policy or account number: 8+ alnum with a digit)."""
    text = text or ""
    if len(_WORD.findall(text)) >= MIN_WORDS:
        return True
    return any(any(c.isdigit() for c in t) for t in _ID.findall(text))


OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")


def file_fingerprint(st):
    return hashlib.sha256(f"{int(st.st_mtime)}:{st.st_size}".encode()).hexdigest()[:16]


def _tesseract_available():
    return shutil.which("tesseract") is not None


def _pdftotext_available():
    return shutil.which("pdftotext") is not None


def _docx_available():
    try:
        import docx  # noqa: F401
        return True
    except ImportError:
        return False


def _run_tesseract(path):
    r = subprocess.run(["tesseract", path, "stdout"], capture_output=True, text=True, timeout=30)
    return r.stdout or ""


def _run_vision(path):
    """Ask a local Ollama vision model to describe an image — the fallback for
    the common case (a personal photo with no real text). Returns None (no
    description, not an error) whenever the GPU is on loan to VM 200/300, the
    image can't be read, or the model/network call fails; never raises."""
    if U.gpu_on_loan():
        return None
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    except OSError:
        return None
    payload = json.dumps({
        "model": OLLAMA_VISION_MODEL,
        "prompt": "Describe this image in one or two factual sentences.",
        "images": [b64], "stream": False,
    }).encode()
    try:
        r = U._http_post(f"{U.OLLAMA_HOST}/api/generate", payload)
    except Exception:
        return None
    return (r.get("response") or "").strip() or None


def _run_pdftotext(path):
    r = subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True, timeout=60)
    return r.stdout or ""


def _run_docx(path):
    import docx
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def extract_image(path, tesseract=None, vision=None):
    """OCR first; if that finds no real text, fall back to a vision
    description. Returns (text, method)."""
    tesseract = tesseract or _run_tesseract
    text = (tesseract(path) or "").strip()
    alnum = sum(c.isalnum() for c in text)
    if alnum >= OCR_MIN_ALNUM:
        return text, "tesseract"
    vision = vision or _run_vision
    desc = vision(path)
    if desc:
        return desc, "ollama-vision"
    return text, "tesseract"


def _count_files(path):
    n = 0
    for _, _, files in os.walk(path):
        n += len(files)
    return n


def _read_text_capped(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = f.read(TEXT_CAP + 1)
    truncated = len(data) > TEXT_CAP
    return data[:TEXT_CAP], truncated


def index_content(store, root, box="ARES", vault_pred=None, budget=500,
                   tesseract=None, vision=None, pdftotext=None, docx_extract=None,
                   docs_only=False, guard=None):
    """Walk `root` and extract text from photos (OCR)/PDF/docx/txt/md into new
    `file-content` nodes, searchable via the existing FTS index (`understanding`
    is the extracted text). Folders are untouched — `walker.py` owns those.

    Vault-excluded paths (per `vault_pred`, same predicate `walker.py` uses)
    are never opened: a matching directory is pruned before descent, and a
    matching file is skipped even if reached another way. `budget` caps how
    many new/changed files are processed per call, since the first pass over
    a large photo library can't finish in one run; a per-file fingerprint
    (mtime:size) makes repeat calls skip anything already indexed.
    `docs_only` skips images, so a priority pass over PROJECTS/WORK is never
    starved by image backlog.
    """
    exts = SUPPORTED_EXT - IMAGE_EXT if docs_only else SUPPORTED_EXT
    vault_pred = vault_pred if vault_pred is not None else default_vault_pred()
    from .photo_vision import guard_files_under_photos      # lazy: photo_vision imports content
    vault_pred = guard_files_under_photos(vault_pred, guard=guard)
    tesseract = tesseract or _run_tesseract
    # Photos are described by atlas.photo_vision (vault-guarded, nightly); the
    # content pass itself never sends images to a vision model by default.
    vision = vision or (lambda _p: None)
    pdftotext = pdftotext or _run_pdftotext
    docx_extract = docx_extract or _run_docx

    have_tesseract = _tesseract_available()
    have_pdftotext = _pdftotext_available()
    have_docx = _docx_available()
    warned = set()

    stats = {"processed": 0, "changed": 0, "skipped_vault": 0,
             "skipped_unchanged": 0, "skipped_budget": 0, "errors": 0}

    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in dirnames:
            if d in IGNORE_DIRS:
                continue
            child = os.path.join(dirpath, d)
            if vault_pred(child):
                # Pruned here means os.walk never descends into it, so this is
                # the only chance to count what got skipped.
                stats["skipped_vault"] += _count_files(child)
                continue
            keep.append(d)
        dirnames[:] = sorted(keep)
        if vault_pred(dirpath):
            stats["skipped_vault"] += len(filenames)
            continue

        for fname in sorted(filenames):
            path = os.path.join(dirpath, fname)
            if vault_pred(path) or (guard is not None and not guard.allowed(path)):
                stats["skipped_vault"] += 1
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in exts:
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue

            nid = node_id(box, path)
            fp = file_fingerprint(st)
            prev = store.get_node(nid)
            if prev and prev.get("fingerprint") == fp:
                stats["skipped_unchanged"] += 1
                continue
            if stats["processed"] >= budget:
                stats["skipped_budget"] += 1
                continue
            stats["processed"] += 1

            if ext in IMAGE_EXT and not have_tesseract:
                if "tesseract" not in warned:
                    print("content: tesseract not installed — skipping image OCR this run")
                    warned.add("tesseract")
                continue
            if ext in PDF_EXT and not have_pdftotext:
                if "pdftotext" not in warned:
                    print("content: pdftotext not installed — skipping PDF extraction this run")
                    warned.add("pdftotext")
                continue
            if ext in DOCX_EXT and not have_docx:
                if "docx" not in warned:
                    print("content: python-docx not installed — skipping docx extraction this run")
                    warned.add("docx")
                continue

            try:
                truncated = False
                if ext in IMAGE_EXT:
                    text, method = extract_image(path, tesseract, vision)
                elif ext in PDF_EXT:
                    text, method = pdftotext(path), "pdftotext"
                elif ext in DOCX_EXT:
                    text, method = docx_extract(path), "docx"
                else:
                    text, truncated = _read_text_capped(path)
                    method = "raw"
            except Exception as e:
                store.upsert_node({
                    "id": nid, "box": box, "kind": "file-content", "path": path,
                    "name": fname, "understanding": None, "fingerprint": fp,
                    "size": st.st_size, "mtime": int(st.st_mtime), "status": "error",
                    "meta": {"error": str(e), "ext": ext},
                })
                stats["errors"] += 1
                continue

            store.upsert_node({
                "id": nid, "box": box, "kind": "file-content", "path": path,
                "name": fname, "understanding": text, "fingerprint": fp,
                "size": st.st_size, "mtime": int(st.st_mtime),
                # "empty" rows stay as fingerprint markers (no nightly re-OCR) but are
                # kept out of FTS and embedding by Store.upsert_node / embed_pending.
                "status": "live" if has_text(text) else "empty",
                "meta": {"method": method, "ext": ext, "truncated": truncated},
            })
            store.add_edge(node_id(box, dirpath), nid, "contains")
            stats["changed"] += 1

    return stats
