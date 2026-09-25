import json
import mcp_server as M
from atlas.store import Store


def _seed(tmp_path, monkeypatch, n=12):
    st = Store(str(tmp_path / "kg.db"))
    for i in range(n):
        st.upsert_node({"id": f"ARES:chat/s{i}", "box": "ARES", "kind": "chat",
                        "path": f"/archive/ARES/p/s{i}.jsonl", "name": f"caddy session {i}",
                        "understanding": "fixed caddy config " + "x" * 1400, "status": "live",
                        "fingerprint": "1:2", "size": 9,
                        "meta": {"session": f"s{i}", "cwd": "/p", "asks": ["a" * 900] * 12},
                        "embedding": st.vec_to_blob([0.5] * 768)})
    st.upsert_node({"id": "ARES:/mnt/x", "box": "ARES", "kind": "folder", "path": "/mnt/x",
                    "name": "x caddy", "understanding": "caddy folder"})
    monkeypatch.setattr(M, "_store", st)
    monkeypatch.setattr(st, "search", lambda q, limit=20, embed_fn=None:
                        [dict(st.get_node(f"ARES:chat/s{i}")) for i in range(min(limit, 12))])
    return st


def test_search_default_is_compact(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    r = M.kg_search("caddy")
    assert len(r) == 8
    for x in r:
        assert len(x["understanding"]) <= M.SEARCH_CLIP + 1
        assert "embedding" not in x and "path" not in x          # chats: id is enough
    assert len(json.dumps(r)) < 4000


def test_search_limit_still_overridable(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    assert len(M.kg_search("caddy", limit=12)) == 12


def test_get_drops_embedding_and_bookkeeping(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    n = M.kg_get("ARES:chat/s1")
    assert "embedding" not in n and "fingerprint" not in n
    assert n["understanding"].startswith("fixed caddy config")    # full text kept here
    assert len(json.dumps(n, default=str)) < 6000


def test_get_missing_is_none(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    assert M.kg_get("nope") is None
