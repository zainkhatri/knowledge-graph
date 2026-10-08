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


# --- faces -----------------------------------------------------------------

def _unit(i, dim=8):
    import numpy as np
    v = np.zeros(dim, dtype="float32"); v[i] = 1.0
    return v


def _faces(tmp_path, root, photos, profiles=None):
    """photos: {relpath: [(tagged_names, face_vec, det_score), ...]} -> FaceIndex.
    profiles: {name: [member face vecs]} (default Zain = axis 0, Hamza = 1, Haadi = 2,
    five members each). A cluster's profile is built from its member faces."""
    import sqlite3
    import numpy as np
    profiles = profiles or {n: [_unit(i)] * 5 for i, n in enumerate(("zain", "hamza", "haadi"))}
    embs, clusters = [], {}
    for n, members in profiles.items():
        idx = list(range(len(embs), len(embs) + len(members)))
        embs.extend(np.asarray(m, dtype="float32") for m in members)
        clusters[n] = {"name": n, "photo_hashes": [], "excluded_hashes": [], "emb_indices": idx,
                       "photo_count": len(members)}
    clusters["unnamed"] = {"name": "", "photo_hashes": [], "emb_indices": []}
    rows, face_index = [], {}
    for i, (rel, faces) in enumerate(photos.items()):
        key = f"{i:032x}"
        disk = rel[len("PHOTOS/"):] if rel.startswith("PHOTOS/") else rel   # old doubled layout
        if not os.path.exists(os.path.join(root, disk)):
            _write(root, disk)
        rows.append(("/mnt/data/PHOTOS/" + rel, json.dumps({"thumb": f"/static/thumbs/{key}.jpg"})))
        face_index[key] = []
        for tags, vec, det in faces:
            for n in tags:
                clusters[n]["photo_hashes"].append(key)
            face_index[key].append({"emb_idx": len(embs), "det_score": det, "bbox": [0, 0, 1, 1]})
            embs.append(np.asarray(vec, dtype="float32"))
    (tmp_path / "face_clusters.json").write_text(json.dumps({str(i): c for i, c in enumerate(clusters.values())}))
    (tmp_path / "face_index.json").write_text(json.dumps(face_index))
    np.save(tmp_path / "face_embeddings.npy", np.stack(embs))
    db = tmp_path / "photo_index.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE photos (path TEXT PRIMARY KEY, item TEXT NOT NULL)")
    con.executemany("INSERT INTO photos VALUES (?,?)", rows)
    con.commit(); con.close()
    return PV.FaceIndex(str(tmp_path / "face_clusters.json"), str(db), str(tmp_path / "face_index.json"),
                        str(tmp_path / "face_embeddings.npy"), photos_root=root)


def _near(i, j=None, w=0.15):
    """A face embedding close to profile i (optionally pulled toward profile j)."""
    import numpy as np
    v = _unit(i) + (w * _unit(j) if j is not None else w * _unit(5))
    return v / np.linalg.norm(v)


def test_face_index_names_only_high_confidence_faces(tmp_path):
    import numpy as np
    root = str(tmp_path / "PHOTOS")
    midway = (_unit(0) + _unit(1)) / np.linalg.norm(_unit(0) + _unit(1))
    fx = _faces(tmp_path, root, {
        "2024/clear.jpg": [(["zain"], _near(0), 0.9), (["hamza"], _near(1), 0.85)],
        "2024/blurry.jpg": [(["zain"], _near(0), 0.4)],                 # weak detection
        "2024/ambiguous.jpg": [(["zain"], midway, 0.9)],                # equally Zain/Hamza
        "2024/mislabeled.jpg": [(["haadi"], _near(0), 0.9)],            # face is Zain, tag says Haadi
        "2024/untagged.jpg": [([], _near(0), 0.9)],                     # profile match, no cluster tag
        "2024/far.jpg": [(["zain"], _unit(4), 0.9)],                    # nobody's face
    })
    assert fx.loaded
    p = lambda r: os.path.join(root, r)
    assert fx.names(p("2024/clear.jpg")) == ["Hamza", "Zain"]
    for r in ("blurry", "ambiguous", "mislabeled", "untagged", "far"):
        assert fx.names(p(f"2024/{r}.jpg")) == [], r
    assert fx.names(p("2024/none.jpg")) == []


def test_face_index_missing_files_means_no_names(tmp_path):
    fx = PV.FaceIndex(*(str(tmp_path / n) for n in ("a.json", "b.db", "c.json", "d.npy")))
    assert fx.names("/x.jpg") == [] and fx.loaded is False


def test_people_go_into_prompt_text_and_meta(tmp_path):
    root = str(tmp_path / "PHOTOS")
    a = _write(root, "2024/a.jpg")
    st = Store(str(tmp_path / "kg.db"))
    _node(st, a)
    g = _guard(tmp_path, root, vaulted=["x/zzz.jpg"])
    fx = _faces(tmp_path, root, {"2024/a.jpg": [(["zain"], _near(0), 0.9), (["hamza"], _near(1), 0.9)]})
    http = SpyHTTP(reply="Two friends at a cafe.")
    PV.describe_pending(st, root, g, faces=fx, http=http, prepare=_ident, gpu_free=lambda: True)
    assert "Hamza, Zain" in http.calls[0]["prompt"]
    n = st.get_node("ARES:" + a)
    assert n["meta"]["people"] == ["Hamza", "Zain"]
    assert "People: Hamza, Zain" in n["understanding"]
    assert [r["id"] for r in st.search("hamza cafe")] == ["ARES:" + a]
    st.close()


def test_contaminated_member_does_not_steal_a_name(tmp_path):
    # One of Hamza's cluster faces is really Zain's (the real data has shared/contaminated
    # members). The trimmed profile ignores it, so Zain's face still gets Zain's name.
    root = str(tmp_path / "PHOTOS")
    prof = {"zain": [_unit(0)] * 5, "hamza": [_unit(1)] * 4 + [_near(0)], "haadi": [_unit(2)] * 5}
    fx = _faces(tmp_path, root, {"2024/a.jpg": [(["zain", "hamza"], _near(0), 0.9)]}, profiles=prof)
    assert fx.names(os.path.join(root, "2024/a.jpg")) == ["Zain"]


def test_split_clusters_of_one_person_share_a_name(tmp_path):
    root = str(tmp_path / "PHOTOS")
    prof = {"zain": [_unit(0)] * 5, "hamza": [_unit(1)] * 5,
            "haadi": [_unit(2)] * 9, "haadi k": [_near(2, 5, 0.05)] * 5}
    fx = _faces(tmp_path, root, {"2024/a.jpg": [(["haadi k"], _near(2), 0.9)]}, profiles=prof)
    assert fx.names(os.path.join(root, "2024/a.jpg")) == ["Haadi"]


def test_close_profiles_with_different_names_stay_separate(tmp_path):
    root = str(tmp_path / "PHOTOS")
    prof = {"zain": [_unit(0)] * 5, "bronny": [_unit(1)] * 5, "zayd": [_near(1, 5, 0.05)] * 5}
    fx = _faces(tmp_path, root, {"2024/a.jpg": [(["zayd"], _near(1, 5, 0.03), 0.9)]}, profiles=prof)
    assert len(fx.groups) == 3
    assert fx.names(os.path.join(root, "2024/a.jpg")) == []          # too close to call


def test_face_index_resolves_stale_nested_photos_paths(tmp_path):
    # The dashboard's photo index still lists most photos under an old layout,
    # /mnt/data/PHOTOS/PHOTOS/<dir>/<file>, while the file now lives at PHOTOS/<dir>/<file>.
    root = str(tmp_path / "PHOTOS")
    real = _write(root, "FUJI/DSCF1.JPG")
    fx = _faces(tmp_path, root, {"PHOTOS/FUJI/DSCF1.JPG": [(["zain"], _near(0), 0.9)]})
    assert fx.names(real) == ["Zain"]
    assert fx.names(os.path.join(root, "PHOTOS/FUJI/DSCF1.JPG")) == []    # no such file: no key


def test_strong_stored_exemplar_counts_and_stale_one_is_ignored(tmp_path):
    root = str(tmp_path / "PHOTOS")
    side = _near(0, 6, 0.55)                    # a real other-angle face of Zain
    probe = _near(0, 6, 2.0)                    # too far from Zain's frontal centroid alone
    fx = _faces(tmp_path, root, {"2024/side.jpg": [(["zain"], probe, 0.9)]})
    assert fx.names(os.path.join(root, "2024/side.jpg")) == []
    cl = json.loads((tmp_path / "face_clusters.json").read_text())
    for c in cl.values():
        if c["name"] == "zain":
            c["exemplars"] = [side.tolist(), _unit(4).tolist()]       # one strong, one stale
    (tmp_path / "face_clusters.json").write_text(json.dumps(cl))
    fx = PV.FaceIndex(str(tmp_path / "face_clusters.json"), str(tmp_path / "photo_index.db"),
                      str(tmp_path / "face_index.json"), str(tmp_path / "face_embeddings.npy"), photos_root=root)
    assert fx.names(os.path.join(root, "2024/side.jpg")) == ["Zain"]
    assert len(fx.groups[[g[0] for g in fx.groups].index("Zain")][3]) == 2


# --- 2026-10-07 leak: pool-wide content pass walked into PHOTOS with no guard ---------
def _pool(tmp_path):
    pool = str(tmp_path / "POOL")
    photos = os.path.join(pool, "PHOTOS")
    ok = _write(photos, "2024/ok.txt", b"harbour trip notes")
    vaulted = _write(photos, "2024/later.txt", b"private words")
    other = _write(pool, "docs/readme.txt", b"pool readme words")
    return pool, photos, ok, vaulted, other


def test_pool_wide_content_pass_never_indexes_vaulted_files(tmp_path, monkeypatch):
    pool, photos, ok, vaulted, other = _pool(tmp_path)
    g = _guard(tmp_path, photos, vaulted=["2024/later.txt"])
    monkeypatch.setattr(PV, "PHOTOS_ROOT", photos)
    monkeypatch.setattr(PV, "load_guard", lambda: g)
    st = Store(str(tmp_path / "kg.db"))
    index_content(st, pool, "ARES", budget=50)                  # no guard passed: the old leak
    assert st.get_node("ARES:" + vaulted) is None
    assert st.get_node("ARES:" + ok) and st.get_node("ARES:" + other)
    assert not st.search("private")
    st.close()


def test_no_trusted_guard_means_nothing_under_photos_is_read(tmp_path, monkeypatch):
    pool, photos, ok, vaulted, other = _pool(tmp_path)
    monkeypatch.setattr(PV, "PHOTOS_ROOT", photos)
    monkeypatch.setattr(PV, "load_guard", lambda: None)         # vault index unreadable
    st = Store(str(tmp_path / "kg.db"))
    index_content(st, pool, "ARES", budget=50)
    assert st.get_node("ARES:" + ok) is None and st.get_node("ARES:" + vaulted) is None
    assert st.get_node("ARES:" + other)                         # outside PHOTOS still indexed
    st.close()


def test_prune_keeps_existing_folder_nodes(tmp_path):
    root = str(tmp_path / "PHOTOS")
    _write(root, "2024/a.jpg")
    os.makedirs(os.path.join(root, "gone_dir"))
    st = Store(str(tmp_path / "kg.db"))
    for p, kind in ((os.path.join(root, "2024"), "file-cluster"),
                    (os.path.join(root, "gone_dir"), "file-cluster")):
        st.upsert_node({"id": "ARES:" + p, "box": "ARES", "kind": kind, "path": p,
                        "name": os.path.basename(p), "understanding": "photos"})
    os.rmdir(os.path.join(root, "gone_dir"))
    res = PV.prune_photos(st, root, _guard(tmp_path, root, vaulted=["2023/other.jpg"]))
    assert st.get_node("ARES:" + os.path.join(root, "2024"))    # real folder kept (was churned nightly)
    assert st.get_node("ARES:" + os.path.join(root, "gone_dir")) is None
    assert res["pruned"] == 1
    st.close()


def test_guard_treats_vault_lookalikes_as_vaulted(tmp_path):
    root = str(tmp_path / "PHOTOS")
    twin = _write(root, "GOOGLE/2024/takeout_copy.jpg")
    other = _write(root, "GOOGLE/2024/fine.jpg")
    vi, ch = _vault_index(tmp_path, [os.path.join(root, ".vault", "iPhone/x.jpg")])
    la = tmp_path / "vault_lookalikes.json"
    la.write_text(json.dumps({"k1": twin}))
    g = PV.VaultGuard(vi, ch, photos_root=root, lookalikes=str(la))
    assert not g.allowed(twin) and g.allowed(other)
    g2 = PV.VaultGuard(vi, ch, photos_root=root, lookalikes=str(tmp_path / "missing.json"))
    assert g2.allowed(twin)                                   # no sweep yet: unchanged behaviour


def test_photos_go_to_ares_12b_when_up_else_eros_4b(tmp_path, monkeypatch):
    from atlas import understanding as U
    root = str(tmp_path / "PHOTOS")
    a = _write(root, "2024/a.jpg"); b = _write(root, "2024/b.jpg")
    st = Store(str(tmp_path / "kg.db"))
    _node(st, a); _node(st, b)
    g = _guard(tmp_path, root, vaulted=["x/zzz.jpg"])
    seen = []
    def spy(url, payload):
        seen.append((url, payload["model"])); return {"response": "A dog on a beach at sunset."}
    up = iter([True, False])
    monkeypatch.setattr(U, "_ares_up", lambda: next(up, False))
    PV.describe_pending(st, root, g, http=spy, prepare=_ident, gpu_free=lambda: True)
    assert seen[0] == (U.LOCAL_LLM_HOST + "/api/generate", PV.LOCAL_VISION_MODEL)
    assert seen[1] == (U.OLLAMA_HOST + "/api/generate", PV.VISION_MODEL)
    models = sorted(st.get_node("ARES:" + p)["meta"]["vision_model"] for p in (a, b))
    assert models == sorted([PV.LOCAL_VISION_MODEL, PV.VISION_MODEL])
    st.close()
