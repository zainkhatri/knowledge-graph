"""One-off verification: replay real historical kg_search queries (extracted
from local Claude Code session transcripts) through the current 3-tier
store.search(), and report success rate + flag any query that returns a
result with similarity/relevance the operator should sanity-check by eye.

Usage: python3 scripts/replay_search_queries.py
"""
import json, glob, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atlas.store import Store

DB = os.environ.get("KG_DB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "homelab_kg.db"))

def extract_historical_queries():
    files = glob.glob(os.path.expanduser("~/.claude/projects/**/*.jsonl"), recursive=True)
    queries = []
    for f in files:
        try:
            with open(f, errors="ignore") as fh:
                lines = fh.readlines()
        except Exception:
            continue
        for line in lines:
            if "mcp__homelab-kg__kg_search" not in line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            content = d.get("message", {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "mcp__homelab-kg__kg_search":
                    q = (block.get("input") or {}).get("query", "")
                    if q:
                        queries.append(q)
    return queries

def main():
    queries = extract_historical_queries()
    st = Store(DB)
    ok = empty = 0
    for q in queries:
        hits = st.search(q, limit=5)
        if hits:
            ok += 1
        else:
            empty += 1
    n = len(queries)
    print(f"total historical queries replayed: {n}")
    print(f"  returned something: {ok} ({ok/n*100:.0f}%)" if n else "  no queries found")
    print(f"  empty: {empty} ({empty/n*100:.0f}%)" if n else "")
    st.close()

if __name__ == "__main__":
    main()
