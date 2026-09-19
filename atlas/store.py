import sqlite3, json
import numpy as np

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
        a zero-vector (which would be a false similarity match)."""
        if vec is None:
            return None
        return np.asarray(vec, dtype="float32").tobytes()

    @staticmethod
    def blob_to_vec(blob):
        """Inverse of vec_to_blob. None in, None out."""
        if blob is None:
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
        ([], empty (0,0) array) if nothing is embedded yet."""
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

    def search(self, query, limit=20, embed_fn=None):
        """Three tiers, stopping at the first with results: AND (exact) ->
        quorum-OR (at least half the content tokens) -> semantic (cosine
        similarity over embeddings, via Ollama). Tier 3 only runs when 1 and
        2 both come back empty, so it adds zero latency to the common case."""
        and_match = self._fts_query(query)
        if not and_match:
            return []
        rows = self.db.execute(
            "SELECT n.* FROM nodes_fts f JOIN nodes n ON n.id=f.id"
            " WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?", (and_match, limit)).fetchall()
        if rows:
            return [self._row(r) for r in rows]
        content_tokens = [t for t in (query or "").split() if t and t.lower() not in self._STOPWORDS]
        if len(content_tokens) >= 2:
            or_match = self._fts_query(" ".join(content_tokens), joiner=" OR ")
            pool_size = max(limit * 4, 60)
            candidates = self.db.execute(
                "SELECT n.* FROM nodes_fts f JOIN nodes n ON n.id=f.id"
                " WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?", (or_match, pool_size)).fetchall()
            needed = max(1, -(-len(content_tokens) // 2))   # ceil(n/2)
            lowered = [t.lower() for t in content_tokens]
            scored = []
            for i, r in enumerate(candidates):
                text = f"{r['name'] or ''} {r['understanding'] or ''}".lower()
                overlap = sum(1 for t in lowered if t in text)
                if overlap >= needed:
                    scored.append((-overlap, i, r))
            scored.sort()
            if scored:
                return [self._row(r) for _, _, r in scored[:limit]]
        # tier 3: semantic
        from . import embeddings as E
        embed_fn = embed_fn or E.embed
        qvec = embed_fn(query)
        if qvec is None:
            return []
        ids, matrix = self.all_embedded()
        if not ids:
            return []
        qv = np.asarray(qvec, dtype="float32")
        qn = qv / (np.linalg.norm(qv) or 1.0)
        mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        sims = mn @ qn
        order = np.argsort(-sims)
        SIM_FLOOR = 0.5   # starting point — tune against the historical-query
                          # replay harness (see docs/plans/2026-09-19-atlas-semantic-search.md Task 6)
        top_ids = [ids[i] for i in order[:limit] if sims[i] >= SIM_FLOOR]
        if not top_ids:
            return []
        ph = ",".join("?" * len(top_ids))
        rows = self.db.execute(f"SELECT * FROM nodes WHERE id IN ({ph})", top_ids).fetchall()
        by_id = {r["id"]: self._row(r) for r in rows}
        return [by_id[i] for i in top_ids if i in by_id]

    def merge_from(self, other_path, box):
        src = sqlite3.connect(f"file:{other_path}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        keep = set()
        try:
            for r in src.execute("SELECT id,box,kind,path,name,understanding,fingerprint,"
                                  "size,mtime,status,meta FROM nodes").fetchall():
                self.upsert_node({"id": r["id"], "box": r["box"], "kind": r["kind"], "path": r["path"],
                                  "name": r["name"], "understanding": r["understanding"],
                                  "fingerprint": r["fingerprint"], "size": r["size"], "mtime": r["mtime"],
                                  "status": r["status"] or "live", "meta": json.loads(r["meta"] or "{}")})
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
