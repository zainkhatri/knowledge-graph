import os
from .walker import walk, default_vault_pred
from . import understanding as U

def collect(store, root, box="ARES", vault_pred=None, gen=None, retry_budget=500):
    """retry_budget caps how many previously-failed (fingerprint-unchanged but
    understanding-empty) nodes get retried per run. Without this cap, fixing the
    empty-understanding bug below would regenerate the entire backlog in one
    pass — for a large existing backlog that's thousands of Ollama calls and
    blows the reindex timeout. Budgeted, it self-heals over many nightly runs
    instead, the same pattern summarize-pending already uses for chat/gpt nodes."""
    vault_pred = vault_pred if vault_pred is not None else default_vault_pred()
    gen = gen or U.generate
    nodes = list(walk(root, box, vault_pred=vault_pred))
    by_id = {n["id"]: n for n in nodes}
    children = {}
    for n in nodes:
        if n["parent"]:
            children.setdefault(n["parent"], []).append(n["id"])
    order = sorted(nodes, key=lambda n: n["path"].count(os.sep), reverse=True)  # deepest first
    understandings = {}
    changed = 0
    retried = 0
    for n in order:
        prev = store.get_node(n["id"])
        kids = [understandings.get(c) for c in children.get(n["id"], [])]
        kids = [k for k in kids if k]
        fp_same = prev and prev.get("fingerprint") == n["fingerprint"]
        # Bug this fixes: a failed/skipped generation (Ollama down, GPU on loan) used
        # to be cached as permanent success — since the folder's fingerprint never
        # changes again on its own, that null understanding was never retried. Now:
        # trust the cache only if it actually holds text, or the retry budget is spent.
        if fp_same and (prev.get("understanding") or retried >= retry_budget):
            u = prev.get("understanding")
        elif "understanding" in n:            # e.g. vault node carries fixed text
            u = n["understanding"]; changed += 1
        else:
            u = gen(n, kids); changed += 1
            if fp_same:
                retried += 1   # retry of a previously-failed node, not a genuine content change
        understandings[n["id"]] = u
        rec = {k: v for k, v in n.items() if k != "parent"}
        rec["understanding"] = u
        rec["status"] = "live"
        store.upsert_node(rec)
        if n["parent"]:
            store.add_edge(n["parent"], n["id"], "contains")
    return {"nodes": len(nodes), "changed": changed, "retried": retried}
