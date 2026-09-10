"""Join each box's graph to a synthetic ZEUS backup hub so the no-filter view is the union."""
ZEUS_ID = "ZEUS:/"

def link_boxes(store):
    store.upsert_node({"id": ZEUS_ID, "box": "ZEUS", "kind": "box", "path": "/", "name": "ZEUS",
                       "understanding": "Backup vault — holds nightly copies of ARES + EROS.",
                       "fingerprint": "hub", "size": 0, "mtime": 0, "status": "live", "meta": {}})
    boxes = [r["box"] for r in store.db.execute(
        "SELECT DISTINCT box FROM nodes WHERE box!='ZEUS'").fetchall()]
    linked = 0
    for b in boxes:
        root = store.db.execute(
            "SELECT id FROM nodes WHERE box=? ORDER BY (length(path)-length(replace(path,'/',''))) ASC, id LIMIT 1",
            (b,)).fetchone()
        if root:
            store.add_edge(root["id"], ZEUS_ID, "backs_up")
            linked += 1
    return {"hub": ZEUS_ID, "linked": linked}
