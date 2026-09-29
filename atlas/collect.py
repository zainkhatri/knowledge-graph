import os, time
from .walker import walk, default_vault_pred
from . import understanding as U
from . import embeddings as E

def collect(store, root, box="ARES", vault_pred=None, gen=None, embed_fn=None, retry_budget=500,
            deadline=None, now=time.monotonic):
    """retry_budget caps how many previously-failed (fingerprint-unchanged but
    understanding-empty) nodes get retried per run. Without this cap, fixing the
    empty-understanding bug below would regenerate the entire backlog in one
    pass — for a large existing backlog that's thousands of Ollama calls and
    blows the reindex timeout. Budgeted, it self-heals over many nightly runs
    instead, the same pattern summarize-pending already uses for chat/gpt nodes.

    embed_fn computes a semantic-search embedding for the same understanding
    text, riding the same retry_budget (one extra Ollama call per already-
    budgeted unit of work, not a new uncapped cost). Vault nodes and any node
    that failed to generate understanding never get an embedding — nothing
    meaningful to embed.

    deadline (seconds) caps wall-clock time spent generating. Past it, a node
    that needs generation is deferred: it keeps its old text, and its stored
    fingerprint is cleared (changed/new node) or kept (failed-retry node) so the
    next run picks it up the same way. Added 2026-09-28 after GPU contention on
    EROS tripled per-call time and the deep backfill hit its systemd timeout
    every night."""
    vault_pred = vault_pred if vault_pred is not None else default_vault_pred()
    gen = gen or U.generate
    embed_fn = embed_fn or E.embed
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
    deferred = 0
    t0 = now()
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
            emb = prev.get("embedding")   # reuse cached embedding alongside cached understanding
            if u and prev.get("status", "live") == "live":
                # Unchanged and cached: the row, FTS entry and contains-edge already
                # exist. Rewriting them costs a scan of nodes_fts per node (~8 min
                # per 16K-folder run, 2026-09-28), so skip the write entirely.
                understandings[n["id"]] = u
                continue
        elif "understanding" in n:            # e.g. vault node carries fixed text
            u = n["understanding"]; changed += 1
            emb = None                        # fixed placeholder text, nothing to embed
        elif deadline is not None and now() - t0 >= deadline:
            deferred += 1
            u = prev.get("understanding") if prev else None
            emb = prev.get("embedding") if prev else None
            if fp_same:
                understandings[n["id"]] = u      # deferred retry: stored row is already right
                continue
            n = dict(n, fingerprint=None)       # changed/new: force regeneration next run
        else:
            u = gen(n, kids); changed += 1
            emb = store.vec_to_blob(embed_fn(u)) if u else None
            if fp_same:
                retried += 1   # retry of a previously-failed node, not a genuine content change
        understandings[n["id"]] = u
        rec = {k: v for k, v in n.items() if k != "parent"}
        rec["understanding"] = u
        rec["embedding"] = emb
        rec["status"] = "live"
        store.upsert_node(rec)
        if n["parent"]:
            store.add_edge(n["parent"], n["id"], "contains")
    return {"nodes": len(nodes), "changed": changed, "retried": retried, "deferred": deferred}
