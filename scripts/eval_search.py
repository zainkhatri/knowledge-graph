"""Known-item retrieval eval for Store.search.

Query = a session's opening prompt (first ~15 words); target = that session. Only
sessions with an LLM summary from the full transcript (meta.sv=2) and an ai-title name,
so the target text paraphrases the query instead of containing it verbatim.
Reports hit@1, hit@8 (kg_search's default limit), MRR, empty rate, and latency.

Usage: KG_DB=... OLLAMA_HOST=... python3 scripts/eval_search.py [N]
"""
import json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atlas.store import Store

DB = os.environ.get("KG_DB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          "data", "homelab_kg.db"))


def eval_set(st, n=150, seed=7):
    rows = st.db.execute(
        "SELECT id, name, meta FROM nodes WHERE kind='chat' AND json_extract(meta,'$.sv')=2"
        " AND coalesce(json_extract(meta,'$.title'),'')!=''"
        " AND json_extract(meta,'$.cwd') NOT LIKE '%cc-proxy%'").fetchall()
    items = []
    for r in rows:
        asks = json.loads(r["meta"] or "{}").get("asks") or []
        if not asks:
            continue
        q = " ".join(asks[0].split()[:15])
        if len(q.split()) < 4 or q.lower().startswith(("you are", "base directory", "/")):
            continue
        items.append((q, r["id"]))
    random.Random(seed).shuffle(items)
    return items[:n]


def run(st, items, search, limit=8):
    h1 = h8 = empty = 0; rr = 0.0; t0 = time.time()
    for q, target in items:
        ids = [r["id"] for r in search(q, limit)]
        empty += not ids
        if target in ids:
            k = ids.index(target) + 1
            rr += 1.0 / k; h1 += k == 1; h8 += 1
    n = len(items)
    return {"n": n, "hit@1": round(h1 / n, 3), "hit@8": round(h8 / n, 3), "mrr": round(rr / n, 3),
            "empty": round(empty / n, 3), "ms/query": round(1000 * (time.time() - t0) / n)}


if __name__ == "__main__":
    st = Store(DB)
    items = eval_set(st, int(sys.argv[1]) if len(sys.argv) > 1 else 150)
    print(json.dumps(run(st, items, lambda q, k: st.search(q, limit=k))))
