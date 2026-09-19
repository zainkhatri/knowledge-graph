from atlas.store import Store

def _seed(st, box, paths):
    for p in paths:
        st.upsert_node({"id": f"{box}:{p}", "box": box, "kind": "folder", "path": p,
                        "name": p.rsplit("/", 1)[-1] or p, "understanding": "u " + p,
                        "fingerprint": "f", "size": 0, "mtime": 1, "status": "live", "meta": {}})

def test_merge_adds_prunes_and_leaves_other_boxes(tmp_path):
    central = Store(str(tmp_path / "c.db"))
    _seed(central, "ARES", ["/a", "/a/b"])
    _seed(central, "EROS", ["/stale"])          # old EROS node that should be pruned
    other = Store(str(tmp_path / "e.db"))
    _seed(other, "EROS", ["/x", "/x/y"])
    other.add_edge("EROS:/x", "EROS:/x/y", "contains")
    other.close()

    res = central.merge_from(str(tmp_path / "e.db"), "EROS")
    assert res == {"merged": 2, "pruned": 1}
    assert central.get_node("EROS:/x") is not None          # new EROS merged in
    assert central.get_node("EROS:/stale") is None          # stale EROS pruned
    assert central.get_node("ARES:/a") is not None          # ARES untouched
    assert any(e["type"] == "contains" for e in central.neighbors("EROS:/x"))  # edge merged
    central.close()

def test_merge_is_idempotent(tmp_path):
    central = Store(str(tmp_path / "c.db"))
    other = Store(str(tmp_path / "e.db"))
    _seed(other, "EROS", ["/x"])
    other.close()
    central.merge_from(str(tmp_path / "e.db"), "EROS")
    res2 = central.merge_from(str(tmp_path / "e.db"), "EROS")
    assert res2 == {"merged": 1, "pruned": 0}
    central.close()

def test_merge_from_preserves_embedding(tmp_path):
    central = Store(str(tmp_path / "c.db"))
    other = Store(str(tmp_path / "e.db"))
    other.upsert_node({"id": "EROS:/x", "box": "EROS", "kind": "folder", "path": "/x",
                       "name": "x", "understanding": "u /x", "fingerprint": "f", "size": 0,
                       "mtime": 1, "status": "live", "meta": {},
                       "embedding": other.vec_to_blob([1.0, 2.0, 3.0])})
    other.close()

    central.merge_from(str(tmp_path / "e.db"), "EROS")
    merged = central.get_node("EROS:/x")
    assert merged is not None
    assert central.blob_to_vec(merged["embedding"]).tolist() == [1.0, 2.0, 3.0]
    central.close()

def test_cli_merge(tmp_path, monkeypatch, capsys):
    from atlas import cli
    other = Store(str(tmp_path / "e.db")); _seed(other, "EROS", ["/x"]); other.close()
    monkeypatch.setenv("KG_DB", str(tmp_path / "central.db"))
    cli.main(["merge", str(tmp_path / "e.db"), "--box", "EROS"])
    out = capsys.readouterr().out
    assert "'merged': 1" in out
