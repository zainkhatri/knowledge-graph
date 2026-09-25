import json
from atlas.store import Store
from atlas.history import index_history

T0 = 1_750_000_000_000          # ms


def _hist(tmp_path, rows):
    p = tmp_path / "history.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    return str(p)


def _row(sid, text, i, project="/mnt/nvme/PROMETHEUS/PROJECTS/ARES-DASHBOARD"):
    return {"display": text, "pastedContents": {}, "project": project,
            "sessionId": sid, "timestamp": T0 + i * 1000}


def _capture(monkeypatch):
    seen = []
    def fake(digest, title=None, cwd=None, box=None, http=None):
        seen.append({"digest": digest, "cwd": cwd, "box": box}); return "Old session summary."
    monkeypatch.setattr("atlas.understanding.summarize_session", fake)
    return seen


def test_creates_node_for_session_missing_from_graph(tmp_path, monkeypatch):
    path = _hist(tmp_path, [_row("old1", "fix the photo grid", 0),
                            _row("old1", "now make it faster", 1)])
    seen = _capture(monkeypatch)
    st = Store(str(tmp_path / "kg.db"))
    res = index_history(st, path, box="ARES", min_idle=0, embed_fn=lambda t: None)
    n = st.get_node("ARES:chat/old1")
    assert res["added"] == 1 and res["summarized"] == 1
    assert n["understanding"] == "Old session summary." and n["status"] == "live"
    assert n["meta"]["source"] == "history" and n["meta"]["turns"] == 2
    assert n["meta"]["cwd"] == "/mnt/nvme/PROMETHEUS/PROJECTS/ARES-DASHBOARD"
    d = seen[0]["digest"]
    assert "USER: fix the photo grid" in d and "USER: now make it faster" in d
    assert "replies were not kept" in d
    st.close()


def test_never_touches_a_session_that_has_a_transcript_node(tmp_path, monkeypatch):
    st = Store(str(tmp_path / "kg.db"))
    st.upsert_node({"id": "ARES:chat/s1", "box": "ARES", "kind": "chat",
                    "path": "/archive/ARES/p/s1.jsonl", "name": "real", "understanding": "real",
                    "status": "live", "meta": {"session": "s1"}})
    path = _hist(tmp_path, [_row("s1", "hello", 0)])
    seen = _capture(monkeypatch)
    res = index_history(st, path, box="ARES", min_idle=0, embed_fn=lambda t: None)
    assert res["added"] == 0 and seen == []
    assert st.get_node("ARES:chat/s1")["understanding"] == "real"
    st.close()


def test_unchanged_history_session_is_skipped(tmp_path, monkeypatch):
    path = _hist(tmp_path, [_row("old1", "a", 0)])
    seen = _capture(monkeypatch)
    st = Store(str(tmp_path / "kg.db"))
    index_history(st, path, box="ARES", min_idle=0, embed_fn=lambda t: None)
    res = index_history(st, path, box="ARES", min_idle=0, embed_fn=lambda t: None)
    assert len(seen) == 1 and res["unchanged"] == 1
    st.close()


def test_recent_session_waits_for_idle(tmp_path, monkeypatch):
    import time
    now_ms = int(time.time() * 1000)
    rows = [{"display": "live one", "project": "/x", "sessionId": "new1", "timestamp": now_ms}]
    path = _hist(tmp_path, rows)
    seen = _capture(monkeypatch)
    st = Store(str(tmp_path / "kg.db"))
    index_history(st, path, box="ARES", min_idle=900, embed_fn=lambda t: None)
    assert seen == [] and st.get_node("ARES:chat/new1")["status"] == "raw"
    st.close()
