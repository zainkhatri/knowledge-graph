import json, os
from atlas.store import Store
from atlas.chats import summarize_pending, index_chats


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


def _write_session(projects_root, project_dir, sid, cwd, ask):
    d = os.path.join(projects_root, project_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{sid}.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({"cwd": cwd, "timestamp": "2026-01-01T00:00:00Z"}) + "\n")
        f.write(json.dumps({"type": "user", "message": {"content": ask}}) + "\n")
    return path


def test_index_chats_reuses_cached_embedding_when_unchanged(tmp_path):
    projects_root = str(tmp_path / "projects")
    _write_session(projects_root, "projA", "sess1", "/some/proj", "fix the caddy config")
    st = Store(str(tmp_path / "kg.db"))

    index_chats(st, projects_root=projects_root, box="ARES", summarize=False)
    node = st.get_node("ARES:chat/sess1")
    assert node is not None
    # simulate summarize_pending having already computed a live summary + embedding
    emb = st.vec_to_blob([1.0, 2.0, 3.0])
    with st.db:
        st.db.execute("UPDATE nodes SET status='live', embedding=? WHERE id=?",
                      (emb, "ARES:chat/sess1"))

    # re-run with the exact same source file (fingerprint unchanged) — must not wipe embedding
    index_chats(st, projects_root=projects_root, box="ARES", summarize=False)
    node2 = st.get_node("ARES:chat/sess1")
    assert node2["status"] == "live"
    assert st.blob_to_vec(node2["embedding"]).tolist() == [1.0, 2.0, 3.0]
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
