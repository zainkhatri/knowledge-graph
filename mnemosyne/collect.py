import os
from .walker import walk, default_vault_pred
from . import understanding as U

def collect(store, root, box="ARES", vault_pred=None, gen=None):
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
    for n in order:
        prev = store.get_node(n["id"])
        kids = [understandings.get(c) or (by_id[c].get("understanding")) for c in children.get(n["id"], [])]
        kids = [k for k in kids if k]
        if prev and prev.get("fingerprint") == n["fingerprint"]:
            u = prev.get("understanding")
        elif "understanding" in n:            # e.g. vault node carries fixed text
            u = n["understanding"]; changed += 1
        else:
            u = gen(n, kids); changed += 1
        understandings[n["id"]] = u
        rec = {k: v for k, v in n.items() if k != "parent"}
        rec["understanding"] = u
        rec["status"] = "live"
        store.upsert_node(rec)
        if n["parent"]:
            store.add_edge(n["parent"], n["id"], "contains")
    return {"nodes": len(nodes), "changed": changed}
