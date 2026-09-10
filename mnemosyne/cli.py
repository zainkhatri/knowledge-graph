import argparse, os, json
from .store import Store

def _db_path():
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "homelab_kg.db")
    return os.getenv("KG_DB", default)

def _tree(st, nid, depth, lvl):
    n = st.get_node(nid)
    if not n:
        return
    print("  " * lvl + f"{n['name']} [{n['kind']}] — {(n.get('understanding') or '')[:80]}")
    if lvl < depth:
        for c in st.children(nid):
            _tree(st, c["id"], depth, lvl + 1)

def main(argv=None):
    ap = argparse.ArgumentParser(prog="kg")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-n", type=int, default=15)
    g = sub.add_parser("get"); g.add_argument("id")
    nb = sub.add_parser("neighbors"); nb.add_argument("id"); nb.add_argument("--type")
    tr = sub.add_parser("tree"); tr.add_argument("id"); tr.add_argument("--depth", type=int, default=2)
    rx = sub.add_parser("reindex"); rx.add_argument("root"); rx.add_argument("--box", default="ARES")
    mg = sub.add_parser("merge"); mg.add_argument("other"); mg.add_argument("--box", required=True)
    sub.add_parser("stat")
    a = ap.parse_args(argv)
    st = Store(_db_path())
    try:
        if a.cmd == "search":
            for n in st.search(a.query, a.n):
                print(f"{n['kind']:12} {n['id']}\n   {n.get('understanding') or ''}")
        elif a.cmd == "get":
            n = st.get_node(a.id); print(json.dumps(n, indent=2) if n else "not found")
        elif a.cmd == "neighbors":
            for e in st.neighbors(a.id, a.type):
                print(f"{e['type']:10} {e['src']} -> {e['dst']}")
        elif a.cmd == "tree":
            _tree(st, a.id, a.depth, 0)
        elif a.cmd == "reindex":
            from .collect import collect
            print(collect(st, a.root, a.box))
        elif a.cmd == "merge":
            print(st.merge_from(a.other, a.box))
        elif a.cmd == "stat":
            c = st.db.execute("SELECT count(*) c FROM nodes").fetchone()["c"]
            e = st.db.execute("SELECT count(*) c FROM edges").fetchone()["c"]
            print(f"nodes {c} edges {e} db {_db_path()}")
    finally:
        st.close()

if __name__ == "__main__":
    main()
