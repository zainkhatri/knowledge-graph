"""Local vision descriptions for every photo, with My Eyes Only kept out.

Each image `file-content` node under the photo root gets a one-or-two sentence
description from a local Ollama vision model (EROS), merged with any OCR text,
so agents can search photos by what is in them. Nothing goes to a cloud API.

The vault boundary is layered and fails closed (see the spec,
docs/superpowers/specs/2026-10-04-photo-descriptions-design.md):
  1. path rules: any dot-directory component (PHOTOS/.vault is the real vault),
     the walker's vault predicate, symlinks, on both the path and its realpath;
  2. the dashboard's own vault index: every vaulted item's original library
     path, and its file name (a copy or re-download under another folder);
  3. known content hashes of vaulted items, checked on the bytes actually read;
  4. checked again right after the model returns, so a photo vaulted mid-run
     is dropped; nodes the guard rejects are deleted, never just skipped;
  5. no vault index, or an empty one, means no run at all.
The vault index is read only to build these sets. Vault files are never opened.
"""
import base64, hashlib, io, json, os, re, shutil, subprocess, tempfile, time

from .walker import default_vault_pred
from . import understanding as U

VISION_MODEL = os.getenv("KG_VISION_MODEL", "gemma3:4b")
VISION_VERSION = 1          # bump to re-describe everything with a new model/prompt
MAX_SIDE = 768
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
DASH = "/mnt/nvme/PROMETHEUS/PROJECTS/ARES-DASHBOARD"
VAULT_INDEX = os.getenv("KG_VAULT_INDEX", f"{DASH}/ai_data/vault.json")
CONTENT_HASHES = os.getenv("KG_CONTENT_HASHES", f"{DASH}/content_hashes.json")
PHOTOS_ROOT = "/mnt/nvme/PROMETHEUS/PHOTOS"
ALIASES = ("/mnt/data/PHOTOS", PHOTOS_ROOT)   # LXC bind path and host path
PROMPT = ("Describe this photo for a personal photo search index in one or two factual "
          "sentences: the scene, setting, main subjects and objects, and any clearly "
          "visible text. Do not guess anyone's name. Reply with the description only, "
          "no introduction.")
_PREAMBLE = re.compile(r"^\s*(?:here(?:'|\u2019)?s|here is|sure[,!.]?)[^:\n]{0,120}:\s*", re.I)


def clean(desc):
    """Collapse whitespace and drop a chatty lead-in ("Here's a description ...:")."""
    return _PREAMBLE.sub("", " ".join((desc or "").split())).strip()


class VaultIndexError(RuntimeError):
    """The vault index could not be trusted, so nothing may be described."""


class VaultGuard:
    def __init__(self, vault_index=VAULT_INDEX, content_hashes=CONTENT_HASHES,
                 photos_root=PHOTOS_ROOT):
        self.root = photos_root
        self.pred = default_vault_pred()
        try:
            with open(vault_index) as f:
                items = json.load(f).get("items") or {}
        except (OSError, ValueError, AttributeError) as e:
            raise VaultIndexError(f"vault index unreadable: {type(e).__name__}") from None
        if not isinstance(items, dict) or not items:
            raise VaultIndexError("vault index empty or old format; refusing to run")
        paths = [self._norm(it.get("path", "")) for it in items.values() if isinstance(it, dict)]
        self.paths = {p for p in paths if p}
        if not all(paths) or len(paths) != len(items):
            raise VaultIndexError("vault index has items without a path; refusing to run")
        self.names = {os.path.basename(p).lower() for p in self.paths}
        self.hashes = self._known_hashes(content_hashes)

    def _norm(self, p):
        for a in ALIASES:
            if p == a or p.startswith(a + "/"):
                return self.root + p[len(a):]
        return p

    def _known_hashes(self, path):
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return set()
        return {h for h, fp in d.items() if isinstance(fp, str) and self._norm(fp) in self.paths}

    def _path_ok(self, p):
        parts = [x for x in p.split(os.sep) if x]
        if any(x.startswith(".") for x in parts) or self.pred(p):
            return False
        return p not in self.paths and os.path.basename(p).lower() not in self.names

    def allowed(self, path):
        """True only if `path` is a regular, non-symlink file clear of every vault rule."""
        try:
            if os.path.islink(path) or not os.path.isfile(path):
                return False
            real = os.path.realpath(path)
        except OSError:
            return False
        return self._path_ok(path) and self._path_ok(real)

    def allowed_bytes(self, data):
        return hashlib.sha256(data).hexdigest() not in self.hashes


def load_guard():
    """The production guard, or None when the vault index cannot be trusted."""
    try:
        return VaultGuard()
    except VaultIndexError:
        return None


def visible(node, guard):
    """Query-time filter: photo-derived nodes show only while the guard still allows
    their file. No guard means no photo nodes (fail closed); other kinds untouched."""
    if (node or {}).get("kind") != "file-content":
        return True
    path = node.get("path") or ""
    if not (path.startswith(PHOTOS_ROOT + "/") or (guard and path.startswith(guard.root + "/"))):
        return True
    return bool(guard) and guard.allowed(path)


def prepare_image(path, scratch):
    """Decode (HEIC via heif-convert), fix orientation, downscale to MAX_SIDE, return
    JPEG bytes. Intermediate files live in a private dir under `scratch`, always removed."""
    from PIL import Image, ImageOps
    tmp = tempfile.mkdtemp(prefix="kgv-", dir=scratch)
    try:
        src = path
        if os.path.splitext(path)[1].lower() in (".heic", ".heif"):
            src = os.path.join(tmp, "x.jpg")
            r = subprocess.run(["heif-convert", "-q", "90", path, src],
                               capture_output=True, timeout=120)
            if r.returncode != 0 or not os.path.exists(src):
                raise OSError("heif-convert failed")
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
        return buf.getvalue()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _ollama(url, payload):
    return U._http_post(url, json.dumps(payload).encode())


def _scratch_dir():
    base = "/dev/shm" if os.path.isdir("/dev/shm") else None
    return tempfile.mkdtemp(prefix="kg-vision-", dir=base)


def _delete(store, nid):
    store.db.execute("DELETE FROM nodes WHERE id=?", (nid,))
    store.db.execute("DELETE FROM nodes_fts WHERE id=?", (nid,))
    store.db.execute("DELETE FROM edges WHERE src=? OR dst=?", (nid, nid))


def _candidates(store, root, limit):
    ph = ",".join("?" * len(IMAGE_EXT))
    return store.db.execute(
        "SELECT * FROM nodes WHERE kind='file-content' AND path LIKE ? AND"
        f" lower(json_extract(meta,'$.ext')) IN ({ph}) AND"
        " coalesce(json_extract(meta,'$.vision_v'),0) < ? ORDER BY mtime DESC LIMIT ?",
        (root.rstrip("/") + "/%", *sorted(IMAGE_EXT), VISION_VERSION, limit)).fetchall()


def _merge(node, desc):
    ocr = (node.get("understanding") or "").strip() if node.get("status") == "live" else ""
    meta = dict(node.get("meta") or {})
    if ocr and "ocr" not in meta:
        meta["ocr"] = ocr[:4000]
    ocr = meta.get("ocr") or ""
    meta.update({"vision_model": VISION_MODEL, "vision_v": VISION_VERSION,
                 "described_at": int(time.time()), "method": "vision+ocr" if ocr else "vision"})
    text = desc + (f"\n\nText in photo: {ocr}" if ocr else "")
    return dict(node, understanding=text, status="live", meta=meta)


def _describe_one(store, node, guard, http, prepare, scratch):
    """Returns 'described', 'rejected' (node deleted) or 'failed'."""
    path = node["path"]
    if not guard.allowed(path):
        with store.db:
            _delete(store, node["id"])
        return "rejected"
    try:
        if not guard.allowed_bytes(_raw(path)):
            with store.db:
                _delete(store, node["id"])
            return "rejected"
        img = prepare(path, scratch)
    except Exception:
        return "failed"
    try:
        r = http(f"{U.OLLAMA_HOST}/api/generate", {
            "model": VISION_MODEL, "prompt": PROMPT, "stream": False,
            "images": [base64.b64encode(img).decode()],
            "options": {"num_predict": 120, "temperature": 0.2}})
        desc = clean(r.get("response"))
    except Exception:
        return "failed"
    if not guard.allowed(path):          # vaulted while the model ran
        with store.db:
            _delete(store, node["id"])
        return "rejected"
    if not desc:
        return "failed"
    store.upsert_node(_merge(node, desc))
    return "described"


def _raw(path):
    with open(path, "rb") as f:
        return f.read()


def describe_pending(store, root, guard, budget=500, minutes=None, http=None,
                     prepare=None, gpu_free=None):
    """Describe up to `budget` photos under `root` that lack a current description,
    newest first, stopping at the `minutes` deadline or when the GPU is not free."""
    assert guard is not None, "describe_pending needs a VaultGuard"
    assert budget >= 0
    http = http or _ollama
    prepare = prepare or prepare_image
    # The vision model runs on EROS (via ARES's :11434 proxy), not the ARES 3080, so the
    # ARES GPU-loan flag does not apply; callers can pass a check for other setups.
    gpu_free = gpu_free or (lambda: True)
    deadline = time.time() + minutes * 60 if minutes else None
    stats = {"described": 0, "rejected": 0, "failed": 0, "stopped": None}
    scratch = _scratch_dir()
    try:
        rows = _candidates(store, root, budget + 200)
        for r in rows:
            if stats["described"] >= budget:
                break
            if deadline and time.time() > deadline:
                stats["stopped"] = "deadline"
                break
            if not gpu_free():
                stats["stopped"] = "gpu-busy"
                break
            stats[_describe_one(store, store._row(r), guard, http, prepare, scratch)] += 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return stats


def prune_photos(store, root, guard):
    """Delete photo-derived nodes whose file is gone or now fails the guard (e.g. moved
    into the vault after it was indexed), then merge FTS segments so text is not kept."""
    assert guard is not None, "prune_photos needs a VaultGuard"
    rows = store.db.execute(
        "SELECT id, path FROM nodes WHERE kind IN ('file-content','file-cluster') AND path LIKE ?",
        (root.rstrip("/") + "/%",)).fetchall()
    doomed = [r["id"] for r in rows if not guard.allowed(r["path"])]
    if doomed:
        with store.db:
            for nid in doomed:
                _delete(store, nid)
        store.db.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('optimize')")
        store.db.commit()
    return {"pruned": len(doomed), "checked": len(rows)}
