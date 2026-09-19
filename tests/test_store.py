import numpy as np
from atlas.store import Store

def test_upsert_get_and_search(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    st.upsert_node({
        "id": "ARES:/x/photos", "box": "ARES", "kind": "dataset",
        "path": "/x/photos", "name": "photos",
        "understanding": "2015 photo library from Snapchat and camera roll",
        "fingerprint": "abc123", "size": 0, "mtime": 100, "status": "live", "meta": {"n_files": 4000},
    })
    n = st.get_node("ARES:/x/photos")
    assert n["name"] == "photos" and n["kind"] == "dataset"
    assert st.get_fingerprint("ARES:/x/photos") == "abc123"
    hits = st.search("snapchat")
    assert any(h["id"] == "ARES:/x/photos" for h in hits)
    st.close()

def test_edges_and_neighbors(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    for nid in ("ARES:/x", "ARES:/x/photos"):
        st.upsert_node({"id": nid, "box": "ARES", "kind": "folder", "path": nid.split(":")[1],
                        "name": nid.rsplit("/", 1)[-1], "understanding": "", "fingerprint": "f",
                        "size": 0, "mtime": 1, "status": "live", "meta": {}})
    st.add_edge("ARES:/x", "ARES:/x/photos", "contains")
    assert st.children("ARES:/x")[0]["id"] == "ARES:/x/photos"
    assert st.neighbors("ARES:/x/photos")[0]["type"] == "contains"
    st.close()

def test_embedding_roundtrip(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    vec = [0.1, 0.2, 0.3, 0.4]
    blob = st.vec_to_blob(vec)
    assert isinstance(blob, bytes)
    back = st.blob_to_vec(blob)
    assert np.allclose(back, vec, atol=1e-6)
    assert st.vec_to_blob(None) is None
    assert st.blob_to_vec(None) is None
    st.close()

def test_upsert_node_stores_and_retrieves_embedding(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    blob = st.vec_to_blob([1.0, 2.0, 3.0])
    st.upsert_node({
        "id": "ARES:/x/photos", "box": "ARES", "kind": "dataset",
        "path": "/x/photos", "name": "photos", "understanding": "photo library",
        "fingerprint": "abc123", "size": 0, "mtime": 100, "status": "live",
        "meta": {}, "embedding": blob,
    })
    n = st.get_node("ARES:/x/photos")
    assert st.blob_to_vec(n["embedding"]).tolist() == [1.0, 2.0, 3.0]
    st.close()

def test_upsert_node_without_embedding_key_stores_null(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    st.upsert_node({
        "id": "ARES:/x", "box": "ARES", "kind": "folder", "path": "/x",
        "name": "x", "understanding": "", "fingerprint": "f", "size": 0,
        "mtime": 1, "status": "live", "meta": {},
    })
    n = st.get_node("ARES:/x")
    assert n["embedding"] is None
    st.close()

def test_all_embedded_returns_ids_and_matrix(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    st.upsert_node({"id": "ARES:/a", "box": "ARES", "kind": "folder", "path": "/a",
                    "name": "a", "understanding": "u", "fingerprint": "f1", "size": 0,
                    "mtime": 1, "status": "live", "meta": {},
                    "embedding": st.vec_to_blob([1.0, 0.0])})
    st.upsert_node({"id": "ARES:/b", "box": "ARES", "kind": "folder", "path": "/b",
                    "name": "b", "understanding": "u", "fingerprint": "f2", "size": 0,
                    "mtime": 1, "status": "live", "meta": {}})  # no embedding
    ids, matrix = st.all_embedded()
    assert ids == ["ARES:/a"]
    assert matrix.shape == (1, 2)
    assert matrix[0].tolist() == [1.0, 0.0]
    st.close()
