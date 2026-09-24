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
    monkeypatch.setattr("atlas.understanding.openrouter_key", lambda: None)   # local Ollama path
    called = {"n": 0}
    def embed_fn(text, http=None):
        called["n"] += 1; return [1.0]
    res = summarize_pending(st, budget=10, embed_fn=embed_fn)
    assert res == {"summarized": 0, "note": "gpu-on-loan"}
    assert called["n"] == 0
    st.close()


def _session(root, sid, entries, project="projA"):
    d = os.path.join(root, project); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{sid}.jsonl")
    with open(p, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return p


def _u(txt):
    return {"type": "user", "cwd": "/some/proj", "timestamp": "2026-09-01T00:00:00Z",
            "message": {"content": txt}}


def _capture(monkeypatch, reply="Summary."):
    seen = []
    def fake(digest, title=None, cwd=None, box=None, http=None):
        seen.append({"digest": digest, "title": title, "cwd": cwd, "box": box}); return reply
    monkeypatch.setattr("atlas.understanding.summarize_session", fake)
    return seen


def test_index_keeps_session_with_no_asks(tmp_path):
    root = str(tmp_path / "p")
    _session(root, "empty1", [{"type": "assistant", "cwd": "/x",
                                "message": {"content": [{"type": "text", "text": "hi"}]}}])
    st = Store(str(tmp_path / "kg.db"))
    index_chats(st, projects_root=root, box="ARES", summarize=False)
    n = st.get_node("ARES:chat/empty1")
    assert n is not None and "session empty1" in n["name"]
    st.close()


def test_summary_sees_whole_transcript_and_title(tmp_path, monkeypatch):
    root = str(tmp_path / "p")
    entries = [_u("first ask")] + [{"type": "assistant", "message": {"content": "x"}}] * 500
    entries += [_u("late ask about tailscale acl"), {"type": "ai-title", "aiTitle": "ACL work"}]
    _session(root, "s1", entries)
    seen = _capture(monkeypatch)
    st = Store(str(tmp_path / "kg.db"))
    res = index_chats(st, projects_root=root, box="ZEUS", min_idle=0, embed_fn=lambda t: None)
    assert res["summarized"] == 1
    assert "late ask about tailscale acl" in seen[0]["digest"]
    assert seen[0]["title"] == "ACL work" and seen[0]["box"] == "ZEUS"
    n = st.get_node("ZEUS:chat/s1")
    assert n["understanding"] == "Summary." and n["status"] == "live"
    st.close()


def test_active_session_not_summarized_until_idle(tmp_path, monkeypatch):
    root = str(tmp_path / "p")
    _session(root, "s1", [_u("working on it")])
    seen = _capture(monkeypatch)
    st = Store(str(tmp_path / "kg.db"))
    index_chats(st, projects_root=root, box="ARES", min_idle=900, embed_fn=lambda t: None)
    assert seen == [] and st.get_node("ARES:chat/s1")["status"] == "raw"
    import time
    index_chats(st, projects_root=root, box="ARES", min_idle=900, embed_fn=lambda t: None,
                now=time.time() + 3600)                  # file unchanged, now idle
    assert len(seen) == 1 and st.get_node("ARES:chat/s1")["status"] == "live"
    st.close()


def test_unchanged_live_session_is_not_resummarized(tmp_path, monkeypatch):
    root = str(tmp_path / "p")
    _session(root, "s1", [_u("fix caddy")])
    seen = _capture(monkeypatch)
    st = Store(str(tmp_path / "kg.db"))
    index_chats(st, projects_root=root, box="ARES", min_idle=0, embed_fn=lambda t: None)
    index_chats(st, projects_root=root, box="ARES", min_idle=0, embed_fn=lambda t: None)
    assert len(seen) == 1
    st.close()


def test_budget_limits_summaries(tmp_path, monkeypatch):
    root = str(tmp_path / "p")
    for i in range(5):
        _session(root, f"s{i}", [_u(f"ask {i}")])
    seen = _capture(monkeypatch)
    st = Store(str(tmp_path / "kg.db"))
    res = index_chats(st, projects_root=root, box="ARES", min_idle=0, summary_budget=2,
                      embed_fn=lambda t: None)
    assert res["summarized"] == 2 and len(seen) == 2 and res["chats"] == 5
    st.close()


def test_out_of_credit_stops_cleanly(tmp_path, monkeypatch):
    from atlas import understanding as U
    root = str(tmp_path / "p")
    for i in range(3):
        _session(root, f"s{i}", [_u(f"ask {i}")])
    def broke(*a, **k):
        raise U.OutOfCredit("402")
    monkeypatch.setattr("atlas.understanding.summarize_session", broke)
    st = Store(str(tmp_path / "kg.db"))
    res = index_chats(st, projects_root=root, box="ARES", min_idle=0, embed_fn=lambda t: None)
    assert res["out_of_credit"] is True and res["summarized"] == 0 and res["chats"] == 3
    assert st.get_node("ARES:chat/s0")["status"] == "raw"
    st.close()


def test_summarize_pending_ignores_gpu_loan_when_openrouter_key(tmp_path, monkeypatch):
    st = Store(str(tmp_path / "kg.db"))
    _seed_raw_chat(st, "ARES:chat/1", ["fix the caddy config"])
    monkeypatch.setattr("atlas.understanding.gpu_on_loan", lambda: True)
    monkeypatch.setattr("atlas.understanding.openrouter_key", lambda: "sk-or-x")
    monkeypatch.setattr("atlas.understanding.summarize_chat", lambda asks, http=None: "Did it.")
    res = summarize_pending(st, budget=10, embed_fn=lambda t, http=None: None)
    assert res["summarized"] == 1
    st.close()


def test_old_summary_version_is_redone_and_path_updated(tmp_path, monkeypatch):
    root = str(tmp_path / "p")
    path = _session(root, "s1", [_u("fix caddy")])
    st = Store(str(tmp_path / "kg.db"))
    s = os.stat(path)
    st.upsert_node({"id": "ZEUS:chat/s1", "box": "ZEUS", "kind": "chat", "path": "/home/zain/x.jsonl",
                    "name": "old", "understanding": "old ollama summary", "status": "live",
                    "fingerprint": f"{int(s.st_mtime)}:{s.st_size}", "meta": {"asks": ["fix caddy"]}})
    seen = _capture(monkeypatch)
    index_chats(st, projects_root=root, box="ZEUS", min_idle=0, embed_fn=lambda t: None)
    n = st.get_node("ZEUS:chat/s1")
    assert len(seen) == 1 and n["understanding"] == "Summary." and n["path"] == path
    assert n["meta"]["sv"] == 2
    st.close()


def test_summarize_pending_leaves_chats_with_live_source_to_indexer(tmp_path, monkeypatch):
    st = Store(str(tmp_path / "kg.db"))
    src = tmp_path / "live.jsonl"; src.write_text("{}\n")
    st.upsert_node({"id": "ARES:chat/live", "box": "ARES", "kind": "chat", "path": str(src),
                    "name": "l", "status": "raw", "meta": {"asks": ["a"]}})
    _seed_raw_chat(st, "ARES:chat/gone", ["b"])          # path does not exist on disk
    monkeypatch.setattr("atlas.understanding.openrouter_key", lambda: "sk-or-x")
    monkeypatch.setattr("atlas.understanding.summarize_chat", lambda asks, http=None: "S")
    res = summarize_pending(st, budget=10, embed_fn=lambda t, http=None: None)
    assert res["summarized"] == 1
    assert st.get_node("ARES:chat/live")["status"] == "raw"
    assert st.get_node("ARES:chat/gone")["status"] == "live"
    st.close()
