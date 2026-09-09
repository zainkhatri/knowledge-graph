from mnemosyne.store import Store

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
