import json
from atlas.store import Store
from atlas.chats import summarize_pending


def _seed_raw_chat(st, node_id, asks):
    st.upsert_node({
        "id": node_id, "box": "ARES", "kind": "chat", "path": node_id,
        "name": "a chat", "understanding": None, "fingerprint": "f",
        "size": 0, "mtime": 1, "status": "raw",
        "meta": {"asks": asks},
    })


def test_summarize_pending_stores_embedding(tmp_path, monkeypatch):
    st = Store(str(tmp_path / "kg.db"))
    _seed_raw_chat(st, "ARES:chat/1", ["fix the caddy config"])
    monkeypatch.setattr("atlas.understanding.gpu_on_loan", lambda: False)
    monkeypatch.setattr("atlas.understanding.summarize_chat", lambda asks, http=None: "Fixed Caddy config.")
    def embed_fn(text, http=None): return [1.0, 2.0, 3.0]
    res = summarize_pending(st, budget=10, embed_fn=embed_fn)
    assert res["summarized"] == 1
    n = st.get_node("ARES:chat/1")
    assert st.blob_to_vec(n["embedding"]).tolist() == [1.0, 2.0, 3.0]
    st.close()


def test_summarize_pending_gpu_on_loan_skips_entirely(tmp_path, monkeypatch):
    st = Store(str(tmp_path / "kg.db"))
    _seed_raw_chat(st, "ARES:chat/1", ["fix the caddy config"])
    monkeypatch.setattr("atlas.understanding.gpu_on_loan", lambda: True)
    called = {"n": 0}
    def embed_fn(text, http=None):
        called["n"] += 1; return [1.0]
    res = summarize_pending(st, budget=10, embed_fn=embed_fn)
    assert res == {"summarized": 0, "note": "gpu-on-loan"}
    assert called["n"] == 0
    st.close()
