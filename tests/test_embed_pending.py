from atlas.store import Store
from atlas.embed_pending import embed_pending


def _n(st, nid, kind="chat", und="summary text", status="live", emb=None):
    st.upsert_node({"id": nid, "box": "ARES", "kind": kind, "path": nid, "name": nid,
                    "understanding": und, "status": status, "embedding": emb})


def test_fills_only_missing_live_vectors(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    _n(st, "a"); _n(st, "b", kind="gpt-chat"); _n(st, "c", kind="claude-chat")
    _n(st, "done", emb=st.vec_to_blob([9.0]))
    _n(st, "raw", status="raw"); _n(st, "empty", und="")
    seen = []
    def fake(text, http=None):
        seen.append(text); return [1.0, 2.0]
    res = embed_pending(st, budget=100, embed_fn=fake, workers=2)
    assert res["embedded"] == 3 and res["still_missing"] == 0
    assert st.blob_to_vec(st.get_node("b")["embedding"]).tolist() == [1.0, 2.0]
    assert st.blob_to_vec(st.get_node("done")["embedding"]).tolist() == [9.0]
    assert st.get_node("raw")["embedding"] is None
    st.close()


def test_budget_and_failures(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    for i in range(5):
        _n(st, f"n{i}")
    res = embed_pending(st, budget=3, embed_fn=lambda t, http=None: None, workers=2)
    assert res["embedded"] == 0 and res["failed"] == 3 and res["still_missing"] == 5
    st.close()
