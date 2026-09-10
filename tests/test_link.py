from mnemosyne.store import Store
from mnemosyne.link import link_boxes, ZEUS_ID

def test_link_boxes_creates_hub_and_backup_edges(tmp_path):
    st = Store(str(tmp_path / "c.db"))
    for box, root in [("ARES", "/mnt/nvme/PROMETHEUS"), ("EROS", "/srv")]:
        st.upsert_node({"id": f"{box}:{root}", "box": box, "kind": "folder", "path": root,
                        "name": root.rsplit("/", 1)[-1], "understanding": "", "fingerprint": "f",
                        "size": 0, "mtime": 1, "status": "live", "meta": {}})
    res = link_boxes(st)
    hub = st.get_node(ZEUS_ID)
    assert hub is not None and hub["kind"] == "box"
    assert res["linked"] == 2
    edges = st.neighbors(ZEUS_ID)
    assert sum(1 for e in edges if e["type"] == "backs_up") == 2
    assert any(e["src"] == "ARES:/mnt/nvme/PROMETHEUS" and e["dst"] == ZEUS_ID for e in edges)
    st.close()
