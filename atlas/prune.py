"""Remove graph nodes that current indexing rules would no longer create.

Walker and content exclusions only stop NEW nodes; rows written before a rule
existed stay searchable forever. This pass applies today's rules to the rows
already in the DB:
  - folder / file-cluster / file-content nodes under an excluded path (IGNORE_DIRS
    component, or the vault predicate, which includes every dot-directory) are
    deleted with their FTS row and edges. Chats are never touched: their path is
    the transcript in the session archive by design.
  - live file-content nodes whose text fails content.has_text() are marked
    status='empty' and dropped from FTS. The row stays as a fingerprint marker so
    the nightly OCR pass does not redo them.
"""
import os
from .walker import IGNORE_DIRS, default_vault_pred
from .content import has_text

FS_KINDS = ("folder", "file-cluster", "file-content")


def _excluded(path, vault_pred):
    parts = [p for p in (path or "").split(os.sep) if p]
    return any(p in IGNORE_DIRS for p in parts) or vault_pred(path)


def prune_excluded(store, vault_pred=None, dry_run=False):
    vault_pred = vault_pred or default_vault_pred()
    ph = ",".join("?" * len(FS_KINDS))
    rows = store.db.execute(
        f"SELECT id, kind, path, status, understanding FROM nodes WHERE kind IN ({ph})",
        FS_KINDS).fetchall()
    doomed = [r["id"] for r in rows if r["path"] and _excluded(r["path"], vault_pred)]
    gone = set(doomed)
    emptied = [r["id"] for r in rows
               if r["kind"] == "file-content" and r["id"] not in gone
               and (r["status"] or "live") == "live" and not has_text(r["understanding"])]
    if not dry_run:
        with store.db:
            for nid in doomed:
                store.db.execute("DELETE FROM nodes WHERE id=?", (nid,))
                store.db.execute("DELETE FROM nodes_fts WHERE id=?", (nid,))
                store.db.execute("DELETE FROM edges WHERE src=? OR dst=?", (nid, nid))
            for nid in emptied:
                store.db.execute("UPDATE nodes SET status='empty', embedding=NULL WHERE id=?", (nid,))
                store.db.execute("DELETE FROM nodes_fts WHERE id=?", (nid,))
    return {"pruned": len(doomed), "emptied": len(emptied), "dry_run": dry_run}
