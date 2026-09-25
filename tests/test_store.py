import os, subprocess, sys
import numpy as np
from atlas.store import Store

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_store_works_without_numpy_installed(tmp_path):
    """EROS (and any other box atlas gets shipped to for a plain filesystem
    reindex — see kg-sync-eros.sh's 'stdlib only' design) doesn't have numpy.
    Store/upsert_node/basic search must keep working there; only the
    embedding-specific calls should need numpy, and only when actually used.
    Runs in a subprocess since sys.modules poisoning must happen before
    atlas.store is first imported anywhere in the process."""
    db_path = str(tmp_path / "kg.db")
    code = f'''
import sys
sys.modules["numpy"] = None   # simulates a box with no numpy installed
sys.path.insert(0, {_REPO_ROOT!r})
from atlas.store import Store
st = Store({db_path!r})
st.upsert_node({{"id": "ARES:/x", "box": "ARES", "kind": "folder", "path": "/x",
                "name": "x", "understanding": "hello", "fingerprint": "f",
                "size": 0, "mtime": 1, "status": "live", "meta": {{}}}})
n = st.get_node("ARES:/x")
assert n["name"] == "x"
hits = st.search("hello")
assert any(h["id"] == "ARES:/x" for h in hits)
# embedding calls degrade gracefully instead of crashing the whole module
assert st.vec_to_blob([1.0, 2.0]) is None
assert st.blob_to_vec(b"whatever") is None
ids, matrix = st.all_embedded()
assert ids == []
print("OK")
'''
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "OK" in r.stdout

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

def test_semantic_fallback_finds_match_with_no_shared_words(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    # "photo library" and "image archive" share zero words but should be
    # semantically close under a real embedding model — here we fake the
    # embedder with hand-picked vectors so the test is deterministic.
    st.upsert_node({"id": "ARES:/photos", "box": "ARES", "kind": "dataset",
                    "path": "/photos", "name": "photos", "understanding": "photo library",
                    "fingerprint": "f1", "size": 0, "mtime": 1, "status": "live", "meta": {},
                    "embedding": st.vec_to_blob([1.0, 0.0, 0.0])})
    st.upsert_node({"id": "ARES:/unrelated", "box": "ARES", "kind": "dataset",
                    "path": "/unrelated", "name": "unrelated", "understanding": "totally unrelated",
                    "fingerprint": "f2", "size": 0, "mtime": 1, "status": "live", "meta": {},
                    "embedding": st.vec_to_blob([0.0, 1.0, 0.0])})
    def fake_embed(text, http=None):
        return [0.9, 0.1, 0.0]   # close to /photos, far from /unrelated
    hits = st.search("image archive", embed_fn=fake_embed)
    assert len(hits) == 1
    assert hits[0]["id"] == "ARES:/photos"
    st.close()

def test_semantic_fallback_returns_empty_when_embed_fails(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    st.upsert_node({"id": "ARES:/x", "box": "ARES", "kind": "dataset", "path": "/x",
                    "name": "x", "understanding": "something", "fingerprint": "f", "size": 0,
                    "mtime": 1, "status": "live", "meta": {}, "embedding": st.vec_to_blob([1.0, 0.0])})
    def failing_embed(text, http=None):
        return None
    assert st.search("nonsense query words", embed_fn=failing_embed) == []
    st.close()

def _node(st, nid, und, vec=None):
    st.upsert_node({"id": nid, "box": "ARES", "kind": "chat", "path": nid, "name": nid,
                    "understanding": und, "fingerprint": "f", "size": 0, "mtime": 1,
                    "status": "live", "meta": {}, "embedding": st.vec_to_blob(vec) if vec else None})


def test_hybrid_returns_keyword_and_semantic_hits_exact_first(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    _node(st, "kw", "zeus backup vault", [0.0, 1.0, 0.0])          # exact keyword hit, far vector
    _node(st, "sem", "nightly horcrux pull", [1.0, 0.0, 0.0])      # no shared words, close vector
    _node(st, "far", "banana bread recipe", [0.0, 0.0, 1.0])       # neither
    hits = [h["id"] for h in st.search("zeus backup", embed_fn=lambda t, http=None: [0.95, 0.05, 0.0])]
    assert hits[0] == "kw"                 # exact keyword match stays on top
    assert "sem" in hits                   # semantic neighbour is no longer suppressed
    assert "far" not in hits
    st.close()


def test_semantic_floor_drops_weak_neighbours(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    _node(st, "weak", "something else", [0.5, 0.866, 0.0])         # cos 0.5 to the query
    assert st.search("image archive", embed_fn=lambda t, http=None: [1.0, 0.0, 0.0]) == []
    st.close()


def test_keyword_hits_survive_embed_failure(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    _node(st, "kw", "zeus backup vault", [1.0, 0.0])
    hits = st.search("zeus backup", embed_fn=lambda t, http=None: None)
    assert [h["id"] for h in hits] == ["kw"]
    st.close()


def test_embedding_matrix_cache_sees_new_vectors(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    _node(st, "a", "alpha", [1.0, 0.0])
    q = lambda t, http=None: [0.0, 1.0]
    assert st.search("zzz qqq", embed_fn=q) == []
    _node(st, "b", "beta", [0.0, 1.0])                              # new vector after first search
    assert [h["id"] for h in st.search("zzz qqq", embed_fn=q)] == ["b"]
    st.close()


def test_long_query_matches_on_two_content_words(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    _node(st, "hit", "moonlight streaming lag fixed on vm200")
    q = "why is moonlight so laggy when i stream games from the windows vm to my mac tonight"
    assert [h["id"] for h in st.search(q, embed_fn=lambda t, http=None: None)] == ["hit"]
    st.close()
