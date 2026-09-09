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
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    @staticmethod
    def _row(r):
        d = dict(r)
        if "meta" in d:
            d["meta"] = json.loads(d["meta"] or "{}")
        return d

    def upsert_node(self, node):
        meta = json.dumps(node.get("meta") or {})
        with self.db:
            self.db.execute(
                "INSERT INTO nodes(id,box,kind,path,name,understanding,fingerprint,size,mtime,status,meta)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET box=excluded.box,kind=excluded.kind,path=excluded.path,"
                "name=excluded.name,understanding=excluded.understanding,fingerprint=excluded.fingerprint,"
                "size=excluded.size,mtime=excluded.mtime,status=excluded.status,meta=excluded.meta",
                (node["id"], node["box"], node["kind"], node["path"], node["name"],
                 node.get("understanding"), node.get("fingerprint"), node.get("size"),
                 node.get("mtime"), node.get("status", "live"), meta))
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

    def search(self, query, limit=20):
        rows = self.db.execute(
            "SELECT n.* FROM nodes_fts f JOIN nodes n ON n.id=f.id"
            " WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?", (query, limit)).fetchall()
        return [self._row(r) for r in rows]

    def close(self):
        self.db.close()
