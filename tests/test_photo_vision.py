import hashlib
import json
import os

import pytest

from atlas.store import Store
from atlas.content import index_content
from atlas import photo_vision as PV


def _write(root, rel, data=b"img"):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    return p


def _vault_index(tmp_path, paths, extra=None):
    items = {f"k{i}": {"path": p, "type": "photo"} for i, p in enumerate(paths)}
    vi = tmp_path / "vault.json"
    vi.write_text(json.dumps({"items": items}))
    hashes = tmp_path / "content_hashes.json"
    hashes.write_text(json.dumps(extra or {}))
    return str(vi), str(hashes)


def _guard(tmp_path, root, vaulted=(), extra=None):
    vi, ch = _vault_index(tmp_path, [os.path.join(root, ".vault", v) for v in vaulted] +
                          [os.path.join(root, v) for v in vaulted], extra)
    return PV.VaultGuard(vi, ch, photos_root=root)


def _node(st, path, text="", status="empty"):
    st.upsert_node({"id": "ARES:" + path, "box": "ARES", "kind": "file-content", "path": path,
                    "name": os.path.basename(path), "understanding": text, "status": status,
                    "mtime": int(os.path.getmtime(path)) if os.path.exists(path) else 0,
                    "meta": {"method": "tesseract", "ext": os.path.splitext(path)[1].lower()}})


class SpyHTTP:
    def __init__(self, reply="A dog on a beach at sunset."):
        self.calls, self.reply = [], reply

    def __call__(self, url, payload):
        self.calls.append(payload)
        return {"response": self.reply}


def _ident(path, tmpdir):            # stand-in for decode+downscale: raw bytes
    with open(path, "rb") as f:
        return f.read()


# --- guard -----------------------------------------------------------------

def test_guard_fails_closed_without_vault_index(tmp_path):
    with pytest.raises(PV.VaultIndexError):
        PV.VaultGuard(str(tmp_path / "missing.json"), str(tmp_path / "nope.json"))


def test_guard_fails_closed_on_empty_vault_index(tmp_path):
    vi, ch = _vault_index(tmp_path, [])
    with pytest.raises(PV.VaultIndexError):
        PV.VaultGuard(vi, ch)


def test_guard_rejects_every_vault_shape(tmp_path):
    root = str(tmp_path / "PHOTOS")
    ok = _write(root, "2024/beach.jpg")
    dot = _write(root, ".vault/2024/secret.jpg")
    hidden = _write(root, "2024/.private/x.jpg")
    named = _write(root, "My Eyes Only/y.jpg")
    same_name = _write(root, "2019/IMG_0042.JPG")
    link = os.path.join(root, "2024", "link.jpg")
    os.symlink(dot, link)
    g = _guard(tmp_path, root, vaulted=["old/IMG_0042.jpg"])
    assert g.allowed(ok)
    for bad in (dot, hidden, named, same_name, link):
        assert not g.allowed(bad), bad


def test_guard_rejects_known_vault_content_hash(tmp_path):
    root = str(tmp_path / "PHOTOS")
    copy = _write(root, "2024/copy.jpg", b"secret-bytes")
    h = hashlib.sha256(b"secret-bytes").hexdigest()
    g = _guard(tmp_path, root, vaulted=["a/orig.jpg"], extra={h: os.path.join(root, "a/orig.jpg")})
    assert not g.allowed_bytes(b"secret-bytes")
    assert g.allowed_bytes(b"other")
    st = Store(str(tmp_path / "kg.db"))
    _node(st, copy)
    http = SpyHTTP()
    PV.describe_pending(st, root, g, http=http, prepare=_ident, gpu_free=lambda: True)
    assert http.calls == []
    st.close()


# --- describing ------------------------------------------------------------

def test_describe_pending_describes_and_merges_ocr(tmp_path):
    root = str(tmp_path / "PHOTOS")
    a = _write(root, "2024/a.jpg")
    b = _write(root, "2024/b.png")
    st = Store(str(tmp_path / "kg.db"))
    _node(st, a)
    _node(st, b, text="TOTAL $42.10 RECEIPT", status="live")
    g = _guard(tmp_path, root, vaulted=["x/zzz.jpg"])
    stats = PV.describe_pending(st, root, g, http=SpyHTTP(), prepare=_ident, gpu_free=lambda: True)
    assert stats["described"] == 2
    na, nb = st.get_node("ARES:" + a), st.get_node("ARES:" + b)
    assert na["status"] == "live" and "dog on a beach" in na["understanding"]
    assert "dog on a beach" in nb["understanding"] and "RECEIPT" in nb["understanding"]
    assert na["meta"]["vision_model"] == PV.VISION_MODEL
    assert [r["id"] for r in st.search("beach")]
    st.close()


def test_describe_pending_is_idempotent_and_budgeted(tmp_path):
    root = str(tmp_path / "PHOTOS")
    st = Store(str(tmp_path / "kg.db"))
    for i in range(5):
        _node(st, _write(root, f"2024/p{i}.jpg"))
    g = _guard(tmp_path, root, vaulted=["x/zzz.jpg"])
    http = SpyHTTP()
    assert PV.describe_pending(st, root, g, budget=3, http=http, prepare=_ident,
                               gpu_free=lambda: True)["described"] == 3
    assert PV.describe_pending(st, root, g, budget=10, http=http, prepare=_ident,
                               gpu_free=lambda: True)["described"] == 2
    assert len(http.calls) == 5
    st.close()


def test_describe_pending_stops_when_gpu_busy(tmp_path):
    root = str(tmp_path / "PHOTOS")
    st = Store(str(tmp_path / "kg.db"))
    _node(st, _write(root, "2024/a.jpg"))
    g = _guard(tmp_path, root, vaulted=["x/zzz.jpg"])
    http = SpyHTTP()
    stats = PV.describe_pending(st, root, g, http=http, prepare=_ident, gpu_free=lambda: False)
    assert stats["described"] == 0 and stats["stopped"] == "gpu-busy" and http.calls == []
    st.close()


def test_result_discarded_if_photo_vaulted_mid_describe(tmp_path):
    root = str(tmp_path / "PHOTOS")
    a = _write(root, "2024/a.jpg")
    st = Store(str(tmp_path / "kg.db"))
    _node(st, a)
    g = _guard(tmp_path, root, vaulted=["x/zzz.jpg"])

    def http(url, payload):           # the photo moves into the vault while the model runs
        os.makedirs(os.path.join(root, ".vault", "2024"), exist_ok=True)
        os.replace(a, os.path.join(root, ".vault", "2024", "a.jpg"))
        return {"response": "A private photo."}

    stats = PV.describe_pending(st, root, g, http=http, prepare=_ident, gpu_free=lambda: True)
    assert stats["described"] == 0
    assert st.get_node("ARES:" + a) is None
    assert not st.search("private")
    st.close()


def test_prepare_cleans_up_its_temp_files(tmp_path):
    root = str(tmp_path / "PHOTOS")
    from PIL import Image
    p = os.path.join(root, "2024", "big.png")
    os.makedirs(os.path.dirname(p))
    Image.new("RGB", (2000, 1000), "red").save(p)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    data = PV.prepare_image(p, str(scratch))
    assert data[:2] == b"\xff\xd8"           # JPEG
    assert os.listdir(scratch) == []


# --- pruning, query filter, canary -----------------------------------------

def test_prune_removes_missing_and_newly_vaulted_photos(tmp_path):
    root = str(tmp_path / "PHOTOS")
    keep = _write(root, "2024/keep.jpg")
    gone = _write(root, "2024/gone.jpg")
    vaulted = _write(root, "2024/later.jpg")
    st = Store(str(tmp_path / "kg.db"))
    for p in (keep, gone, vaulted):
        _node(st, p, text=f"words about {os.path.basename(p)}", status="live")
    os.remove(gone)
    g = _guard(tmp_path, root, vaulted=["2024/later.jpg"])
    assert PV.prune_photos(st, root, g)["pruned"] == 2
    assert st.get_node("ARES:" + keep)
    assert st.get_node("ARES:" + gone) is None and st.get_node("ARES:" + vaulted) is None
    assert not st.search("later")
    st.close()


def test_visible_hides_photo_nodes_the_guard_rejects(tmp_path):
    root = str(tmp_path / "PHOTOS")
    ok = _write(root, "2024/ok.jpg")
    g = _guard(tmp_path, root, vaulted=["2024/later.jpg"])
    mk = lambda p: {"kind": "file-content", "path": p}
    assert PV.visible(mk(ok), g)
    assert not PV.visible(mk(os.path.join(root, "2024/missing.jpg")), g)
    assert not PV.visible(mk(os.path.join(root, ".vault/2024/later.jpg")), g)
    assert not PV.visible(mk(PV.PHOTOS_ROOT + "/2024/ok.jpg"), None)  # no guard -> fail closed
    assert PV.visible({"kind": "chat", "path": "/x"}, None)  # non-photo nodes untouched


def test_canary_vault_photo_never_reaches_model_or_graph(tmp_path):
    root = str(tmp_path / "PHOTOS")
    canary = b"CANARY-7f3a9-NEVER-INDEX"
    _write(root, ".vault/2025/canary.jpg", canary)
    _write(root, "2025/canary.jpg", canary)          # a stray copy outside the vault
    _write(root, "2025/fine.jpg", b"fine")
    g = _guard(tmp_path, root, vaulted=["2025/canary.jpg"])
    st = Store(str(tmp_path / "kg.db"))
    index_content(st, root, box="ARES", tesseract=lambda p: "", budget=100, guard=g)
    http = SpyHTTP(reply="A plain photo.")
    PV.describe_pending(st, root, g, http=http, prepare=_ident, gpu_free=lambda: True)
    import base64
    sent = [base64.b64decode(img) for c in http.calls for img in c["images"]]
    assert sent and all(canary not in s for s in sent)
    rows = st.db.execute("SELECT kind, path FROM nodes WHERE path LIKE '%/canary.jpg'").fetchall()
    assert [tuple(r) for r in rows] == []
    assert len(http.calls) == 1                       # only fine.jpg was described
    st.close()


def test_clean_drops_chatty_preamble():
    raw = "Here’s a description of the photo suitable for a personal photo search index:\n\nA dog  on a beach."
    assert PV.clean(raw) == "A dog on a beach."
    assert PV.clean("Here is a description: A cat.") == "A cat."
    assert PV.clean("A man at a cafe: reading.") == "A man at a cafe: reading."
