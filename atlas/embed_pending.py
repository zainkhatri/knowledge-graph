"""Backfill semantic-search vectors for summarized nodes that have none.

Vectors are normally computed right after a summary is written, but that step is
skipped whenever Ollama is unreachable or the GPU is on loan (.gpu-on-loan), which
leaves summarized chats that keyword search finds but semantic search cannot.
Bounded by budget; embeds in threads (HTTP only), writes on the calling thread.
"""
from concurrent.futures import ThreadPoolExecutor

KINDS = ("chat", "gpt-chat", "claude-chat")


PHOTO_ONLY = " AND json_extract(meta,'$.vision_v')>=1"   # described photos, not every OCR'd file


def _missing_count(store, ph, kinds, extra=""):
    return store.db.execute(
        f"SELECT count(*) FROM nodes WHERE status='live' AND embedding IS NULL"
        f" AND trim(coalesce(understanding,''))!='' AND kind IN ({ph}){extra}", kinds).fetchone()[0]


def embed_pending(store, budget=5000, kinds=KINDS, embed_fn=None, workers=4, photos=False):
    """photos=True: vectors for vision-described photos (file-content with meta.vision_v)."""
    from . import embeddings as E
    embed_fn = embed_fn or E.embed
    if photos:
        kinds = ("file-content",)
    extra = PHOTO_ONLY if photos else ""
    ph = ",".join("?" * len(kinds))
    rows = store.db.execute(
        f"SELECT id, understanding FROM nodes WHERE status='live' AND embedding IS NULL"
        f" AND trim(coalesce(understanding,''))!='' AND kind IN ({ph}){extra} LIMIT ?",
        (*kinds, int(budget))).fetchall()
    done = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        vecs = ex.map(lambda r: (r["id"], embed_fn(r["understanding"])), rows)
        for nid, vec in vecs:
            blob = store.vec_to_blob(vec) if vec else None
            if blob is None:
                failed += 1
                continue
            with store.db:
                store.db.execute("UPDATE nodes SET embedding=? WHERE id=?", (blob, nid))
            done += 1
    return {"embedded": done, "failed": failed,
            "still_missing": _missing_count(store, ph, kinds, extra)}
