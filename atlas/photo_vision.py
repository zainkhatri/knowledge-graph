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
import base64, hashlib, io, json, os, re, shutil, sqlite3, subprocess, tempfile, time

from .walker import default_vault_pred
from . import understanding as U

VISION_MODEL = os.getenv("KG_VISION_MODEL", "gemma3:4b")
VISION_VERSION = 4          # bump to re-describe everything (4: calibrated face profiles)
MAX_SIDE = 768
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
DASH = "/mnt/nvme/PROMETHEUS/PROJECTS/ARES-DASHBOARD"
VAULT_INDEX = os.getenv("KG_VAULT_INDEX", f"{DASH}/ai_data/vault.json")
CONTENT_HASHES = os.getenv("KG_CONTENT_HASHES", f"{DASH}/content_hashes.json")
# library copies that look like a vaulted photo (ARES-DASHBOARD system/vault_lookalike_sweep.py)
LOOKALIKES = os.getenv("KG_VAULT_LOOKALIKES", f"{DASH}/ai_data/vault_lookalikes.json")
FACE_CLUSTERS = os.getenv("KG_FACE_CLUSTERS", f"{DASH}/ai_data/face_clusters.json")
PHOTO_INDEX = os.getenv("KG_PHOTO_INDEX", f"{DASH}/photo_index.db")
FACE_INDEX = os.getenv("KG_FACE_INDEX", f"{DASH}/ai_data/face_index.json")
FACE_EMB = os.getenv("KG_FACE_EMB", f"{DASH}/ai_data/face_embeddings.npy")
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
                 photos_root=PHOTOS_ROOT, lookalikes=LOOKALIKES):
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
        try:                                   # look-alike copies are vaulted too (paths only;
            with open(lookalikes) as f:        # their names are ordinary, so not added to names)
                self.paths |= {self._norm(p) for p in json.load(f).values() if p}
        except (OSError, ValueError, AttributeError):
            pass
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


class FaceIndex:
    """High-confidence people per photo, from the dashboard's face data. A name is used
    only when ALL hold for some face in the photo: detector score >= MIN_DET, distance to
    that person's profile <= MAX_DIST, a lead >= MIN_MARGIN over the next person, and the
    cluster also tags the photo. Missing/unreadable data means no names, never looser.

    Calibrated 2026-10-04 (no human labels exist: 0 seed/excluded photos):
    - Profiles are a trimmed centroid of each cluster's own faces (`emb_indices`, keep
      the closest 70%, 3 passes). The stored 5 `exemplars` are not usable: for Zain,
      Hamza and Bronny their centroid sits ~1.0 from the cluster's real faces, and two
      profiles share an identical exemplar (bronny/zaeem, mohsin/bholat).
    - With these profiles, genuine anchor faces sit at median 0.52 and the 0.01%
      impostor quantile is 1.067; MAX_DIST 1.00 sits below it. ~11.5k photos get names.
    - Clusters within ALIAS_DIST whose names share a word (omar / omar saleem) are one
      person: one name (the bigger cluster's), no margin contest. Close pairs WITHOUT a
      shared word (bronny / zayd) stay separate, so faces between them get no name.
    The dashboard's own expand step (nearest exemplar, 1.05 / 0.02 lead) is much looser,
    so its tags alone are never used."""
    MIN_DET = float(os.getenv("KG_FACE_MIN_DET", "0.7"))
    MAX_DIST = float(os.getenv("KG_FACE_MAX_DIST", "1.00"))
    MIN_MARGIN = float(os.getenv("KG_FACE_MIN_MARGIN", "0.10"))
    ALIAS_DIST = 0.6
    MIN_FACES = 5
    EXEMPLAR_OK = 0.8

    def __init__(self, clusters=FACE_CLUSTERS, photo_index=PHOTO_INDEX, faces=FACE_INDEX,
                 embeddings=FACE_EMB, photos_root=PHOTOS_ROOT):
        self.root, self.key_by_path, self.loaded, self.groups = photos_root, {}, False, []
        try:
            import numpy as np
            self.np = np
            self.embs = np.load(embeddings, mmap_mode="r")
            self._load_clusters(clusters)
            with open(faces) as f:
                self.faces = json.load(f)
            self.key_by_path = self._paths(photo_index)
            self.loaded = len(self.groups) >= 2
        except (OSError, ValueError, KeyError, TypeError, IndexError, ImportError, sqlite3.Error):
            self.key_by_path = {}

    def _profile(self, idx):
        np = self.np
        E = np.asarray(self.embs[sorted(idx)], dtype="float32")
        c = E.mean(axis=0)
        for _ in range(3):
            c = c / np.linalg.norm(c)
            d = np.linalg.norm(E - c, axis=1)
            c = E[d <= np.quantile(d, 0.7)].mean(axis=0)
        return c / np.linalg.norm(c)

    def _load_clusters(self, path):
        with open(path) as f:
            clusters = json.load(f)
        self.tags, profs = {}, []
        for c in clusters.values():
            name = (c.get("name") or "").strip() if isinstance(c, dict) else ""
            if not name or name.isdigit():
                continue
            name = name.title()
            gone = set(c.get("excluded_hashes") or [])
            for k in c.get("photo_hashes") or []:
                if k not in gone:
                    self.tags.setdefault(k, set()).add(name)
            idx = {int(i) for i in c.get("emb_indices") or [] if 0 <= int(i) < len(self.embs)}
            if len(idx) >= self.MIN_FACES:
                cen = self._profile(idx)
                profs.append((name, cen, int(c.get("photo_count") or 0),
                              self._prototypes(cen, c.get("exemplars"))))
        self.groups = self._alias_groups(profs)

    def _prototypes(self, cen, exemplars):
        """The centroid plus stored exemplars that agree with the person's own faces
        (<= EXEMPLAR_OK): strong hand-built models count, stale ones are ignored."""
        np, out = self.np, [cen]
        for ex in exemplars or []:
            v = np.asarray(ex, dtype="float32")
            if v.shape == cen.shape and np.linalg.norm(v) > 0:
                v = v / np.linalg.norm(v)
                if float(np.linalg.norm(v - cen)) <= self.EXEMPLAR_OK:
                    out.append(v)
        return np.stack(out)

    def _alias_groups(self, profs):
        """[(display_name, member_names, centroid, prototypes)]; merges near profiles that
        share a word (their prototypes are pooled)."""
        np, groups = self.np, []
        for name, cen, _, protos in sorted(profs, key=lambda p: -p[2]):   # biggest names the group
            words = set(name.lower().split())
            for k, g in enumerate(groups):
                same_word = any(words & set(m.lower().split()) for m in g[1])
                if same_word and float(np.linalg.norm(g[2] - cen)) < self.ALIAS_DIST:
                    g[1].add(name)
                    groups[k] = (g[0], g[1], g[2], np.concatenate([g[3], protos]))
                    break
            else:
                groups.append((name, {name}, cen, protos))
        return groups

    def _paths(self, db_path):
        if not os.path.isfile(db_path):
            raise OSError("photo index missing")
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            out = {}
            for path, item in con.execute("SELECT path, item FROM photos"):
                key = (json.loads(item).get("thumb") or "").rsplit("/", 1)[-1].replace(".jpg", "")
                real = self._resolve(path) if key in self.tags else None
                if real:
                    out[real] = key
            return out
        finally:
            con.close()

    def _norm(self, p):
        for a in ALIASES:
            if p.startswith(a + "/"):
                return self.root + p[len(a):]
        return p

    def _resolve(self, p):
        """The file an index path refers to today. Most rows still use an old layout
        with a doubled PHOTOS segment (PHOTOS/PHOTOS/FUJI/x -> PHOTOS/FUJI/x), so the
        exact path is tried first, then the collapsed one; neither existing means skip."""
        exact = self._norm(p)
        if os.path.isfile(exact):
            return exact
        nested = self.root + "/PHOTOS/"
        if exact.startswith(nested):
            moved = self.root + "/" + exact[len(nested):]
            if os.path.isfile(moved):
                return moved
        return None

    def _confident(self, face, tagged):
        if float(face.get("det_score") or 0) < self.MIN_DET:
            return None
        e = self.embs[int(face["emb_idx"])]
        d = sorted((float(self.np.min(self.np.linalg.norm(protos - e, axis=1))), i)
                   for i, (_, _, _, protos) in enumerate(self.groups))
        (best, i), (second, _) = d[0], d[1]
        name, members, _, _ = self.groups[i]
        if best <= self.MAX_DIST and second - best >= self.MIN_MARGIN and members & tagged:
            return name
        return None

    def names(self, path):
        key = self.key_by_path.get(path)
        if not key:
            return []
        tagged = self.tags.get(key, set())
        found = {self._confident(f, tagged) for f in self.faces.get(key) or []}
        return sorted(n for n in found if n)


def load_guard():
    """The production guard, or None when the vault index cannot be trusted."""
    try:
        return VaultGuard()
    except VaultIndexError:
        return None


def guard_files_under_photos(base_pred, guard=None, loader=None):
    """Wrap a walker/content vault predicate so every FILE under PHOTOS_ROOT must also pass
    the VaultGuard, whatever root the walk started from. Directories are not judged (the
    guard only rules on files; dot-dir vaults are already caught by base_pred). The guard
    loads lazily, once; if it cannot be trusted every file under PHOTOS is excluded.
    Why: 2026-10-07 the pool-wide content pass (root /mnt/nvme/PROMETHEUS, no guard) OCR'd
    ~300 vaulted originals into the graph every day."""
    state = {"g": guard, "loaded": guard is not None}

    def pred(path):
        if base_pred(path):
            return True
        root = PHOTOS_ROOT
        if not path.startswith(root + "/") or not os.path.isfile(path):
            return False
        if not state["loaded"]:
            state["g"], state["loaded"] = (loader or load_guard)(), True
        return state["g"] is None or not state["g"].allowed(path)
    return pred


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


def _prompt(people):
    if not people:
        return PROMPT
    return (PROMPT + f" Face recognition identified these people in the photo: {', '.join(people)}."
            " Refer to them by these names where it fits; do not add any other names.")


def _merge(node, desc, people=()):
    ocr = (node.get("understanding") or "").strip() if node.get("status") == "live" else ""
    meta = dict(node.get("meta") or {})
    if ocr and "ocr" not in meta:
        meta["ocr"] = ocr[:4000]
    ocr = meta.get("ocr") or ""
    meta["people"] = list(people)
    meta.update({"vision_model": VISION_MODEL, "vision_v": VISION_VERSION,
                 "described_at": int(time.time()), "method": "vision+ocr" if ocr else "vision"})
    text = desc + (f"\n\nPeople: {', '.join(people)}" if people else "")
    text += f"\n\nText in photo: {ocr}" if ocr else ""
    return dict(node, understanding=text, status="live", meta=meta)


def _describe_one(store, node, guard, http, prepare, scratch, faces=None):
    """Returns 'described', 'rejected' (node deleted) or 'failed'."""
    path = node["path"]
    if not guard.allowed(path):
        with store.db:
            _delete(store, node["id"])
        return "rejected"
    people = faces.names(path) if faces else []
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
            "model": VISION_MODEL, "prompt": _prompt(people), "stream": False,
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
    store.upsert_node(_merge(node, desc, people))
    return "described"


def _raw(path):
    with open(path, "rb") as f:
        return f.read()


def describe_pending(store, root, guard, budget=500, minutes=None, http=None,
                     prepare=None, gpu_free=None, faces=None):
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
            stats[_describe_one(store, store._row(r), guard, http, prepare, scratch, faces)] += 1
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
    # the guard rules on files only; an existing folder node (walker file-cluster/folder)
    # is not a leak, and deleting it made the walker re-add it every day (366/night churn)
    doomed = [r["id"] for r in rows
              if not os.path.isdir(r["path"]) and not guard.allowed(r["path"])]
    if doomed:
        with store.db:
            for nid in doomed:
                _delete(store, nid)
        store.db.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('optimize')")
        store.db.commit()
    return {"pruned": len(doomed), "checked": len(rows)}
