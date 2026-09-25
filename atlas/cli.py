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
    rx.add_argument("--retry-budget", type=int, default=500)
    mg = sub.add_parser("merge"); mg.add_argument("other"); mg.add_argument("--box", required=True)
    sub.add_parser("link-boxes")
    ic = sub.add_parser("index-chats"); ic.add_argument("--box", default="ARES")
    ic.add_argument("--root", default="/root/.claude/projects")
    ic.add_argument("--summary-budget", type=int, default=None)
    ic.add_argument("--min-idle", type=int, default=900, help="seconds a session must be idle before summarizing")
    ic.add_argument("--workers", type=int, default=8)
    ic.add_argument("--no-summarize", action="store_true")
    ih = sub.add_parser("index-history"); ih.add_argument("--box", default="ARES")
    ih.add_argument("--file", required=True, help="a ~/.claude/history.jsonl (prompt log)")
    ih.add_argument("--summary-budget", type=int, default=None)
    ih.add_argument("--min-idle", type=int, default=900)
    ih.add_argument("--workers", type=int, default=8)
    ig = sub.add_parser("index-gpt"); ig.add_argument("--box", default="ARES")
    ig.add_argument("--dir", default="/mnt/nvme/PROMETHEUS/PERSONAL/GPT")
    ig.add_argument("--summary-budget", type=int, default=None)
    iw = sub.add_parser("index-claude-web"); iw.add_argument("--box", default="ARES")
    iw.add_argument("--dir", required=True); iw.add_argument("--summary-budget", type=int, default=None)
    ie = sub.add_parser("index-env"); ie.add_argument("--box", default="ARES")
    ie.add_argument("--home", default="/root/.claude"); ie.add_argument("--config", default="/root/.claude.json")
    sub.add_parser("link-workdirs")
    ep = sub.add_parser("embed-pending"); ep.add_argument("--budget", type=int, default=5000)
    ep.add_argument("--workers", type=int, default=4)
    ur = sub.add_parser("usage-report"); ur.add_argument("--days", type=int, default=7)
    ur.add_argument("--archive", default="/mnt/nvme/PROMETHEUS/PERSONAL/CLAUDE-CODE-SESSIONS")
    ur.add_argument("--log", default="/var/log/kg-usage.jsonl", help="append one JSON line here ('' = don't)")
    sm = sub.add_parser("summarize-pending"); sm.add_argument("--budget", type=int, default=500)
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
            print(collect(st, a.root, a.box, retry_budget=a.retry_budget))
        elif a.cmd == "merge":
            print(st.merge_from(a.other, a.box))
        elif a.cmd == "link-boxes":
            from .link import link_boxes
            print(link_boxes(st))
        elif a.cmd == "index-chats":
            from .chats import index_chats
            print(index_chats(st, a.root, a.box, summarize=not a.no_summarize,
                              summary_budget=a.summary_budget, min_idle=a.min_idle,
                              workers=a.workers))
        elif a.cmd == "index-history":
            from .history import index_history
            print(index_history(st, a.file, a.box, summary_budget=a.summary_budget,
                                min_idle=a.min_idle, workers=a.workers))
        elif a.cmd == "index-gpt":
            from .gpt import index_gpt
            print(index_gpt(st, a.dir, a.box, summary_budget=a.summary_budget))
        elif a.cmd == "index-claude-web":
            from .claude_web import index_claude_web
            print(index_claude_web(st, a.dir, a.box, summary_budget=a.summary_budget))
        elif a.cmd == "index-env":
            from .env import index_env
            print(index_env(st, a.box, [a.home, "/mnt/nvme/PROMETHEUS/.claude"], a.config))
        elif a.cmd == "embed-pending":
            from .embed_pending import embed_pending
            print(embed_pending(st, a.budget, workers=a.workers))
        elif a.cmd == "link-workdirs":
            from .workdirs import link_workdirs
            print(link_workdirs(st))
        elif a.cmd == "usage-report":
            from .usage import report
            r = report(a.archive, a.days)
            if a.log:
                with open(a.log, "a") as f:
                    f.write(json.dumps(r) + "\n")
            print(json.dumps(r, indent=2))
        elif a.cmd == "summarize-pending":
            from .chats import summarize_pending
            print(summarize_pending(st, a.budget))
        elif a.cmd == "stat":
            c = st.db.execute("SELECT count(*) c FROM nodes").fetchone()["c"]
            e = st.db.execute("SELECT count(*) c FROM edges").fetchone()["c"]
            print(f"nodes {c} edges {e} db {_db_path()}")
    finally:
        st.close()

if __name__ == "__main__":
    main()
