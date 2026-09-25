import sqlite3, json

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes(
  id TEXT PRIMARY KEY, box TEXT, kind TEXT, path TEXT, name TEXT,
  understanding TEXT, fingerprint TEXT, size INTEGER, mtime INTEGER,
  status TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS edges(
  src TEXT, dst TEXT, type TEXT, meta TEXT, PRIMARY KEY(src,dst,type));
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(id UNINDEXED, name, understanding);
"""

class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=60000")   # wait, don't fail, on concurrent writers
        self.db.execute("PRAGMA journal_mode=WAL")      # readers don't block the writer
        self.db.executescript(SCHEMA)
        try:
            self.db.execute("ALTER TABLE nodes ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass  # column already exists (every run after the first on a given db file)

    @staticmethod
    def _row(r):
        d = dict(r)
        if "meta" in d:
            d["meta"] = json.loads(d["meta"] or "{}")
        return d

    @staticmethod
    def vec_to_blob(vec):
        """Serialize an embedding vector to the BLOB format stored in
        nodes.embedding. None in, None out — 'no embedding' is NULL, never
        a zero-vector (which would be a false similarity match). numpy is
        imported lazily, after the None-check, so a box with no embedding
        to store (the common case on EROS — see kg-sync-eros.sh's 'stdlib
        only' design, and no numpy there) never needs numpy at all; a box
        that also lacks numpy but somehow gets a real vector degrades to
        None (embedding silently skipped) instead of crashing the caller."""
        if vec is None:
            return None
        try:
            import numpy as np
        except ImportError:
            return None
        return np.asarray(vec, dtype="float32").tobytes()

    @staticmethod
    def blob_to_vec(blob):
        """Inverse of vec_to_blob. None in, None out."""
        if blob is None:
            return None
        try:
            import numpy as np
        except ImportError:
            return None
        return np.frombuffer(blob, dtype="float32")

    def upsert_node(self, node):
        meta = json.dumps(node.get("meta") or {})
        with self.db:
            self.db.execute(
                "INSERT INTO nodes(id,box,kind,path,name,understanding,fingerprint,size,mtime,status,meta,embedding)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET box=excluded.box,kind=excluded.kind,path=excluded.path,"
                "name=excluded.name,understanding=excluded.understanding,fingerprint=excluded.fingerprint,"
                "size=excluded.size,mtime=excluded.mtime,status=excluded.status,meta=excluded.meta,"
                "embedding=excluded.embedding",
                (node["id"], node["box"], node["kind"], node["path"], node["name"],
                 node.get("understanding"), node.get("fingerprint"), node.get("size"),
                 node.get("mtime"), node.get("status", "live"), meta, node.get("embedding")))
            self.db.execute("DELETE FROM nodes_fts WHERE id=?", (node["id"],))
            self.db.execute("INSERT INTO nodes_fts(id,name,understanding) VALUES(?,?,?)",
                            (node["id"], node.get("name") or "", node.get("understanding") or ""))

    def get_node(self, node_id):
        r = self.db.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not r:
            return None
        return self._row(r)

    def get_fingerprint(self, node_id):
        r = self.db.execute("SELECT fingerprint FROM nodes WHERE id=?", (node_id,)).fetchone()
        return r["fingerprint"] if r else None

    def add_edge(self, src, dst, type_, meta=None):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO edges(src,dst,type,meta) VALUES(?,?,?,?)",
                            (src, dst, type_, json.dumps(meta or {})))

    def neighbors(self, node_id, etype=None):
        q = "SELECT * FROM edges WHERE (src=? OR dst=?)"; args = [node_id, node_id]
        if etype is not None:
            q += " AND type=?"; args.append(etype)
        return [dict(r) for r in self.db.execute(q, args).fetchall()]

    def children(self, node_id):
        rows = self.db.execute(
            "SELECT n.* FROM edges e JOIN nodes n ON n.id=e.dst"
            " WHERE e.src=? AND e.type='contains' ORDER BY n.name", (node_id,)).fetchall()
        return [self._row(r) for r in rows]

    def all_embedded(self, box=None):
        """Return (ids, matrix) for every node with a non-null embedding —
        the corpus for semantic similarity search. matrix.shape is
        (n_nodes, 768); ids[i] corresponds to matrix row i. Returns
        ([], empty (0,0) array) if nothing is embedded yet, or if numpy
        isn't installed on this box (see vec_to_blob's docstring — a box
        with no numpy also never has real embeddings to serve, so this is
        just the same empty-corpus case)."""
        try:
            import numpy as np
        except ImportError:
            return [], None
        q = "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL"
        args = []
        if box:
            q += " AND box=?"; args.append(box)
        rows = self.db.execute(q, args).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype="float32")
        ids = [r["id"] for r in rows]
        matrix = np.stack([self.blob_to_vec(r["embedding"]) for r in rows])
        return ids, matrix

    @staticmethod
    def _fts_query(query, joiner=" "):
        """Quote each whitespace token as an FTS5 phrase so punctuation like
        '-' is treated as literal text, not a MATCH operator (which raises
        OperationalError on queries such as 'kg-nightly'). Internal double
        quotes are doubled per FTS5 escaping. joiner=" " gives implicit AND
        (precise); joiner=" OR " gives OR (broader recall). Returns '' for an
        empty query."""
        tokens = [t for t in (query or "").split() if t]
        return joiner.join('"' + t.replace('"', '""') + '"' for t in tokens)

    # Excluded from the fallback's quorum count (and from the OR candidate fetch)
    # so a stopword-heavy off-topic sentence can't rack up a fake quorum just by
    # coincidence — with CLAUDE.md now calling kg_search on every message, this
    # is the difference between "silent on unrelated turns" and "surfaces junk".
    _STOPWORDS = {
        "a","an","and","are","as","at","be","by","for","from","how","i","in","into",
        "is","it","its","many","of","on","or","that","the","there","this","to","was",
        "were","what","when","where","which","who","why","will","with","you","your",
    }

    # Hybrid ranking, tuned on scripts/eval_search.py (known-item retrieval, 2026-09-25):
    # exact-AND x3 + quorum-OR + semantic (cos >= 0.60), fused by reciprocal rank.
    # vs the old "semantic only if keywords find nothing" tiers: title queries hit@1
    # 0.873 -> 0.913, natural-language prompts hit@8 0.42 -> 0.66.
    RRF_K = 60
    POOL = 50
    SIM_FLOOR = 0.60          # off-topic queries top out ~0.51-0.61; homelab ones ~0.69+
    WEIGHTS = (3.0, 1.0, 1.0)  # (exact AND, quorum OR, semantic)
    EMB_TTL = 600
    QUORUM_CAP = 2

    def _fts_ids(self, match, k):
        return [r[0] for r in self.db.execute(
            "SELECT f.id FROM nodes_fts f WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?", (match, k))]

    def _quorum_or_ids(self, query, k):
        """OR over content tokens, kept only if min(half of them, QUORUM_CAP) appear. The
        cap matters for long natural-language queries: requiring 8 of 15 words threw away
        good matches (eval hit@8 0.57 -> 0.66 with cap 2); 1 lets single-word noise in."""
        tokens = [t for t in (query or "").split() if t and t.lower() not in self._STOPWORDS]
        if len(tokens) < 2:
            return []
        rows = self.db.execute(
            "SELECT n.id, n.name, n.understanding FROM nodes_fts f JOIN nodes n ON n.id=f.id"
            " WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?",
            (self._fts_query(" ".join(tokens), joiner=" OR "), max(k, 60))).fetchall()
        needed = max(1, min(-(-len(tokens) // 2), self.QUORUM_CAP))
        low = [t.lower() for t in tokens]
        scored = []
        for i, r in enumerate(rows):
            text = f"{r['name'] or ''} {r['understanding'] or ''}".lower()
            overlap = sum(1 for t in low if t in text)
            if overlap >= needed:
                scored.append((-overlap, i, r["id"]))
        scored.sort()
        return [nid for _, _, nid in scored[:k]]

    def _emb_matrix(self):
        """(ids, row-normalized matrix), cached; rebuilt when the embedded count changes
        or after EMB_TTL seconds (catches in-place re-embeds)."""
        import time
        n = self.db.execute("SELECT count(*) FROM nodes WHERE embedding IS NOT NULL").fetchone()[0]
        c = getattr(self, "_emb_cache", None)
        if c and c[0] == n and time.time() - c[1] < self.EMB_TTL:
            return c[2], c[3]
        ids, matrix = self.all_embedded()
        if not ids:
            self._emb_cache = (n, time.time(), [], None)
            return [], None
        import numpy as np
        mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        self._emb_cache = (n, time.time(), ids, mn)
        return ids, mn

    def _semantic_ids(self, query, k, embed_fn):
        from . import embeddings as E
        qvec = (embed_fn or E.embed)(query)
        if qvec is None:
            return []
        ids, mn = self._emb_matrix()
        if not ids:
            return []           # also covers "numpy not installed"
        import numpy as np
        qv = np.asarray(qvec, dtype="float32")
        if qv.shape[0] != mn.shape[1]:
            return []
        sims = mn @ (qv / (np.linalg.norm(qv) or 1.0))
        order = np.argsort(-sims)[:k]
        return [ids[i] for i in order if sims[i] >= self.SIM_FLOOR]

    def search(self, query, limit=20, embed_fn=None):
        """Hybrid search: exact AND, quorum OR and semantic neighbours, merged by weighted
        reciprocal-rank fusion. Exact matches dominate (weight 3); semantic hits are no
        longer suppressed when keywords match. Degrades to keyword-only when Ollama/numpy
        is unavailable."""
        and_match = self._fts_query(query)
        if not and_match:
            return []
        lists = (self._fts_ids(and_match, self.POOL),
                 self._quorum_or_ids(query, self.POOL),
                 self._semantic_ids(query, self.POOL, embed_fn))
        score = {}
        for w, ids in zip(self.WEIGHTS, lists):
            for rank, nid in enumerate(ids):
                score[nid] = score.get(nid, 0.0) + w / (self.RRF_K + rank + 1)
        top = sorted(score, key=lambda nid: -score[nid])[:int(limit)]
        if not top:
            return []
        ph = ",".join("?" * len(top))
        rows = self.db.execute(f"SELECT * FROM nodes WHERE id IN ({ph})", top).fetchall()
        by_id = {r["id"]: self._row(r) for r in rows}
        return [by_id[i] for i in top if i in by_id]

    def merge_from(self, other_path, box):
        src = sqlite3.connect(f"file:{other_path}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        keep = set()
        try:
            for r in src.execute("SELECT id,box,kind,path,name,understanding,fingerprint,"
                                  "size,mtime,status,meta,embedding FROM nodes").fetchall():
                self.upsert_node({"id": r["id"], "box": r["box"], "kind": r["kind"], "path": r["path"],
                                  "name": r["name"], "understanding": r["understanding"],
                                  "fingerprint": r["fingerprint"], "size": r["size"], "mtime": r["mtime"],
                                  "status": r["status"] or "live", "meta": json.loads(r["meta"] or "{}"),
                                  "embedding": r["embedding"]})
                keep.add(r["id"])
            for e in src.execute("SELECT src,dst,type,meta FROM edges").fetchall():
                self.add_edge(e["src"], e["dst"], e["type"], json.loads(e["meta"] or "{}"))
        finally:
            src.close()
        stale = [r["id"] for r in self.db.execute("SELECT id FROM nodes WHERE box=?", (box,)).fetchall()
                 if r["id"] not in keep]
        with self.db:
            for nid in stale:
                self.db.execute("DELETE FROM nodes WHERE id=?", (nid,))
                self.db.execute("DELETE FROM nodes_fts WHERE id=?", (nid,))
                self.db.execute("DELETE FROM edges WHERE src=? OR dst=?", (nid, nid))
        return {"merged": len(keep), "pruned": len(stale)}

    def close(self):
        self.db.close()
