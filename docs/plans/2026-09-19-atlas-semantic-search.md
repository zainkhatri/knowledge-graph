# ATLAS Semantic Search Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a semantic (embedding-based) search tier to ATLAS's `store.search()` that only fires when today's lexical AND/quorum-OR tiers return nothing, so natural-language queries that don't share exact words with a node's summary can still find it.

**Architecture:** New `atlas/embeddings.py` module wraps Ollama's `/api/embeddings` endpoint (model `nomic-embed-text`, already pulled) behind the same fail-soft, injectable-`http` contract as the existing `understanding.py`. `store.py` gains a nullable `embedding BLOB` column, storage/retrieval helpers, and a third search tier that does brute-force cosine similarity (numpy) over all embedded nodes. `collect.py` and `chats.py`'s `summarize_pending` compute an embedding any time they (re)generate understanding text, riding their existing per-run budgets — no new backfill job.

**Tech Stack:** Python stdlib + numpy (already installed) + Ollama HTTP API (already running). No new package installs.

## Global Constraints

- Zero new pip installs — numpy and the `nomic-embed-text` Ollama model are both already present on the box (verified: `numpy.__version__` 2.4.6; `ollama list` shows `nomic-embed-text:latest`).
- Embedding vectors are 768-dim float32 (verified live: `curl /api/embeddings` for `nomic-embed-text` returns a 768-length `embedding` array).
- All new Ollama-calling code must follow the existing fail-soft contract: any exception, GPU-on-loan, or unreachable Ollama returns `None`/`[]`, never raises.
- Semantic tier must not change behavior for any query that already succeeds under AND or quorum-OR — it only runs when both of those return empty.
- No new cron/systemd timer — embedding generation rides the existing budgeted `collect()` and `summarize_pending()` passes.
- Every new/modified function that talks to Ollama takes an injectable parameter (`http=` or `embed_fn=`) so tests never make real network calls, matching `atlas/understanding.py`'s existing pattern.

---

### Task 1: `atlas/embeddings.py` — the embedding client

**Files:**
- Create: `atlas/embeddings.py`
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Produces: `embed(text: str, http=None) -> list[float] | None` — used by Tasks 2-5.
- Produces: `gpu_on_loan() -> bool` (mirrors `understanding.gpu_on_loan`, separate copy so this module has no import-time dependency on `understanding.py`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embeddings.py
from atlas import embeddings as E

def test_embed_uses_injected_http():
    seen = {}
    def fake_http(url, data):
        seen["url"] = url; seen["data"] = data
        return {"embedding": [0.1, 0.2, 0.3]}
    out = E.embed("photos folder with 4000 images", http=fake_http)
    assert out == [0.1, 0.2, 0.3]
    assert seen["url"].endswith("/api/embeddings")
    assert b"nomic-embed-text" in seen["data"]

def test_empty_text_returns_none_without_calling_http():
    called = {"n": 0}
    def fake_http(url, data):
        called["n"] += 1; return {"embedding": [1.0]}
    assert E.embed("", http=fake_http) is None
    assert E.embed("   ", http=fake_http) is None
    assert called["n"] == 0

def test_http_failure_returns_none():
    def boom(url, data):
        raise RuntimeError("ollama down")
    assert E.embed("some text", http=boom) is None

def test_missing_embedding_key_returns_none():
    def fake_http(url, data):
        return {}
    assert E.embed("some text", http=fake_http) is None

def test_gpu_loan_skips(monkeypatch):
    monkeypatch.setattr(E, "gpu_on_loan", lambda: True)
    called = {"n": 0}
    def fake_http(url, data):
        called["n"] += 1; return {"embedding": [1.0]}
    assert E.embed("some text", http=fake_http) is None
    assert called["n"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && python3 -m pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'atlas.embeddings'`

- [ ] **Step 3: Write the implementation**

```python
# atlas/embeddings.py
import os, json, urllib.request

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.20.51:11434")
EMBED_MODEL = os.getenv("KG_EMBED_MODEL", "nomic-embed-text")
GPU_LOAN_FLAG = os.getenv("GPU_LOAN_FLAG",
                          "/mnt/nvme/PROMETHEUS/PROJECTS/ARES-DASHBOARD/.gpu-on-loan")

def gpu_on_loan():
    return os.path.exists(GPU_LOAN_FLAG)

def _http_post(url, data):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())

def embed(text, http=None):
    """Returns a 768-dim embedding as list[float], or None on any failure:
    empty/whitespace-only text, GPU on loan, Ollama unreachable, timeout, or
    a malformed response. Never raises — same fail-soft contract as
    understanding.generate()."""
    if not (text or "").strip():
        return None
    if gpu_on_loan():
        return None
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    caller = http or _http_post
    try:
        resp = caller(f"{OLLAMA_HOST}/api/embeddings", payload)
        vec = resp.get("embedding")
        return vec if vec else None
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && python3 -m pytest tests/test_embeddings.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas"
git add atlas/embeddings.py tests/test_embeddings.py
git commit -m "feat: add embeddings.embed() client for semantic search"
```

---

### Task 2: `atlas/store.py` — embedding storage

**Files:**
- Modify: `atlas/store.py` (schema migration in `__init__`, `upsert_node`, new `vec_to_blob`/`blob_to_vec`/`all_embedded` methods)
- Test: `tests/test_store.py` (add to existing file)

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `Store.vec_to_blob(vec: list[float] | None) -> bytes | None`, `Store.blob_to_vec(blob: bytes | None) -> numpy.ndarray | None`, `Store.all_embedded(box: str | None = None) -> tuple[list[str], numpy.ndarray]`, and `upsert_node()` now accepts an optional `"embedding"` key (bytes or None) in the node dict. Used by Tasks 3, 4, 5.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_store.py
import numpy as np
from atlas.store import Store

def test_embedding_roundtrip(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    vec = [0.1, 0.2, 0.3, 0.4]
    blob = st.vec_to_blob(vec)
    assert isinstance(blob, bytes)
    back = st.blob_to_vec(blob)
    assert np.allclose(back, vec, atol=1e-6)
    assert st.vec_to_blob(None) is None
    assert st.blob_to_vec(None) is None
    st.close()

def test_upsert_node_stores_and_retrieves_embedding(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    blob = st.vec_to_blob([1.0, 2.0, 3.0])
    st.upsert_node({
        "id": "ARES:/x/photos", "box": "ARES", "kind": "dataset",
        "path": "/x/photos", "name": "photos", "understanding": "photo library",
        "fingerprint": "abc123", "size": 0, "mtime": 100, "status": "live",
        "meta": {}, "embedding": blob,
    })
    n = st.get_node("ARES:/x/photos")
    assert st.blob_to_vec(n["embedding"]).tolist() == [1.0, 2.0, 3.0]
    st.close()

def test_upsert_node_without_embedding_key_stores_null(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    st.upsert_node({
        "id": "ARES:/x", "box": "ARES", "kind": "folder", "path": "/x",
        "name": "x", "understanding": "", "fingerprint": "f", "size": 0,
        "mtime": 1, "status": "live", "meta": {},
    })
    n = st.get_node("ARES:/x")
    assert n["embedding"] is None
    st.close()

def test_all_embedded_returns_ids_and_matrix(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    st.upsert_node({"id": "ARES:/a", "box": "ARES", "kind": "folder", "path": "/a",
                    "name": "a", "understanding": "u", "fingerprint": "f1", "size": 0,
                    "mtime": 1, "status": "live", "meta": {},
                    "embedding": st.vec_to_blob([1.0, 0.0])})
    st.upsert_node({"id": "ARES:/b", "box": "ARES", "kind": "folder", "path": "/b",
                    "name": "b", "understanding": "u", "fingerprint": "f2", "size": 0,
                    "mtime": 1, "status": "live", "meta": {}})  # no embedding
    ids, matrix = st.all_embedded()
    assert ids == ["ARES:/a"]
    assert matrix.shape == (1, 2)
    assert matrix[0].tolist() == [1.0, 0.0]
    st.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && python3 -m pytest tests/test_store.py -v -k embedding or all_embedded`
Expected: FAIL — `AttributeError: 'Store' object has no attribute 'vec_to_blob'`

- [ ] **Step 3: Write the implementation**

Modify `atlas/store.py` — add `numpy` import at the top:

```python
import sqlite3, json
import numpy as np
```

Add the migration at the end of `__init__` (after `self.db.executescript(SCHEMA)`):

```python
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
```

Add two static methods (anywhere in the class, e.g. right after `_row`):

```python
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
```

Replace `upsert_node` to add the `embedding` column:

```python
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
```

Add `all_embedded` (e.g. right after `children`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && python3 -m pytest tests/test_store.py -v`
Expected: all passed (existing + 4 new)

- [ ] **Step 5: Commit**

```bash
cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas"
git add atlas/store.py tests/test_store.py
git commit -m "feat: add embedding column, storage helpers, and all_embedded() to Store"
```

---

### Task 3: `atlas/store.py` — semantic search tier

**Files:**
- Modify: `atlas/store.py` (`search()` method only)
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `Store.all_embedded()`, `Store.blob_to_vec()` (Task 2); `embeddings.embed()` (Task 1) as the default `embed_fn`.
- Produces: `Store.search(query, limit=20, embed_fn=None)` — same signature plus one new optional param, fully backward compatible (existing callers that don't pass `embed_fn` get the real `embeddings.embed`).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_store.py
def test_semantic_fallback_finds_match_with_no_shared_words(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    # "photo library" and "image archive" share zero words but should be
    # semantically close under a real embedding model — here we fake the
    # embedder with hand-picked vectors so the test is deterministic.
    st.upsert_node({"id": "ARES:/photos", "box": "ARES", "kind": "dataset",
                    "path": "/photos", "name": "photos", "understanding": "photo library",
                    "fingerprint": "f1", "size": 0, "mtime": 1, "status": "live", "meta": {},
                    "embedding": st.vec_to_blob([1.0, 0.0, 0.0])})
    st.upsert_node({"id": "ARES:/unrelated", "box": "ARES", "kind": "dataset",
                    "path": "/unrelated", "name": "unrelated", "understanding": "totally unrelated",
                    "fingerprint": "f2", "size": 0, "mtime": 1, "status": "live", "meta": {},
                    "embedding": st.vec_to_blob([0.0, 1.0, 0.0])})
    def fake_embed(text, http=None):
        return [0.9, 0.1, 0.0]   # close to /photos, far from /unrelated
    hits = st.search("image archive", embed_fn=fake_embed)
    assert len(hits) == 1
    assert hits[0]["id"] == "ARES:/photos"
    st.close()

def test_semantic_fallback_returns_empty_when_embed_fails(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    st.upsert_node({"id": "ARES:/x", "box": "ARES", "kind": "dataset", "path": "/x",
                    "name": "x", "understanding": "something", "fingerprint": "f", "size": 0,
                    "mtime": 1, "status": "live", "meta": {}, "embedding": st.vec_to_blob([1.0, 0.0])})
    def failing_embed(text, http=None):
        return None
    assert st.search("nonsense query words", embed_fn=failing_embed) == []
    st.close()

def test_and_match_still_wins_without_calling_embed_fn(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    st.upsert_node({"id": "ARES:/x", "box": "ARES", "kind": "dataset", "path": "/x",
                    "name": "x", "understanding": "zeus backup vault", "fingerprint": "f",
                    "size": 0, "mtime": 1, "status": "live", "meta": {}})
    called = {"n": 0}
    def fake_embed(text, http=None):
        called["n"] += 1; return [1.0]
    hits = st.search("zeus backup", embed_fn=fake_embed)
    assert len(hits) == 1
    assert called["n"] == 0   # AND already matched — semantic tier never invoked
    st.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && python3 -m pytest tests/test_store.py -v -k semantic`
Expected: FAIL — `TypeError: search() got an unexpected keyword argument 'embed_fn'`

- [ ] **Step 3: Write the implementation**

Replace the `search` method in `atlas/store.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && python3 -m pytest tests/test_store.py -v`
Expected: all passed (existing + Task 2's 4 + these 3)

- [ ] **Step 5: Commit**

```bash
cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas"
git add atlas/store.py tests/test_store.py
git commit -m "feat: add semantic search tier as fallback after AND/quorum-OR"
```

---

### Task 4: `atlas/collect.py` — embed during folder backfill

**Files:**
- Modify: `atlas/collect.py`
- Test: `tests/test_collect.py`

**Interfaces:**
- Consumes: `Store.vec_to_blob()` (Task 2), an injectable `embed_fn` defaulting to `embeddings.embed` (Task 1).
- Produces: `collect(store, root, box="ARES", vault_pred=None, gen=None, embed_fn=None, retry_budget=500)` — same return shape `{"nodes", "changed", "retried"}`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_collect.py
def test_collect_stores_embedding_alongside_understanding(tmp_path):
    build_tree(str(tmp_path))
    st = Store(str(tmp_path / "kg.db"))
    def gen(node, kids): return f"understood:{node['name']}"
    def embed_fn(text, http=None): return [1.0, 2.0]
    collect(st, str(tmp_path), "ARES", gen=gen, embed_fn=embed_fn)
    proj = st.get_node("ARES:" + os.path.join(str(tmp_path), "projA"))
    assert st.blob_to_vec(proj["embedding"]).tolist() == [1.0, 2.0]

def test_collect_reuses_cached_embedding_when_unchanged(tmp_path):
    build_tree(str(tmp_path))
    st = Store(str(tmp_path / "kg.db"))
    calls = {"n": 0}
    def gen(node, kids): return "u"
    def embed_fn(text, http=None):
        calls["n"] += 1; return [float(calls["n"])]
    collect(st, str(tmp_path), "ARES", gen=gen, embed_fn=embed_fn)
    first_calls = calls["n"]
    collect(st, str(tmp_path), "ARES", gen=gen, embed_fn=embed_fn)   # nothing changed
    assert calls["n"] == first_calls   # embed_fn not called again, cache reused

def test_collect_vault_nodes_never_embedded(tmp_path):
    v = tmp_path / "My Eyes Only"
    (v / "secret").mkdir(parents=True)
    (v / "secret" / "d.txt").write_text("x")
    (tmp_path / "readme.txt").write_text("hi")
    st = Store(str(tmp_path / "kg.db"))
    from atlas import walker
    def gen(node, kids): return "u"
    def embed_fn(text, http=None): return [1.0]
    collect(st, str(tmp_path), "ARES", vault_pred=walker.default_vault_pred(), gen=gen, embed_fn=embed_fn)
    vnode = st.get_node("ARES:" + str(v))
    assert vnode["embedding"] is None
    st.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && python3 -m pytest tests/test_collect.py -v -k embed`
Expected: FAIL — `TypeError: collect() got an unexpected keyword argument 'embed_fn'`

- [ ] **Step 3: Write the implementation**

Replace `atlas/collect.py` in full:

```python
import os
from .walker import walk, default_vault_pred
from . import understanding as U
from . import embeddings as E

def collect(store, root, box="ARES", vault_pred=None, gen=None, embed_fn=None, retry_budget=500):
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
    meaningful to embed."""
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
        elif "understanding" in n:            # e.g. vault node carries fixed text
            u = n["understanding"]; changed += 1
            emb = None                        # fixed placeholder text, nothing to embed
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
    return {"nodes": len(nodes), "changed": changed, "retried": retried}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && python3 -m pytest tests/test_collect.py -v`
Expected: all passed (existing 4 + these 3)

- [ ] **Step 5: Commit**

```bash
cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas"
git add atlas/collect.py tests/test_collect.py
git commit -m "feat: compute+cache embeddings during folder backfill"
```

---

### Task 5: `atlas/chats.py` — embed during chat/gpt-archive backfill

**Files:**
- Modify: `atlas/chats.py` (`summarize_pending` only)
- Test: `tests/test_chats.py` (create — no existing test file for this module)

**Interfaces:**
- Consumes: `Store.vec_to_blob()` (Task 2), injectable `embed_fn` defaulting to `embeddings.embed` (Task 1).
- Produces: `summarize_pending(store, budget=500, kinds=(...), embed_fn=None)` — same return shape `{"summarized", "still_raw"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chats.py
import json
from atlas.store import Store
from atlas.chats import summarize_pending

def _seed_raw_chat(st, node_id, asks):
    st.upsert_node({
        "id": node_id, "box": "ARES", "kind": "chat", "path": node_id,
        "name": "a chat", "understanding": None, "fingerprint": "f",
        "size": 0, "mtime": 1, "status": "raw",
        "meta": {"asks": asks},
    })

def test_summarize_pending_stores_embedding(tmp_path, monkeypatch):
    st = Store(str(tmp_path / "kg.db"))
    _seed_raw_chat(st, "ARES:chat/1", ["fix the caddy config"])
    monkeypatch.setattr("atlas.chats.gpu_on_loan", lambda: False)
    monkeypatch.setattr("atlas.understanding.summarize_chat", lambda asks, http=None: "Fixed Caddy config.")
    def embed_fn(text, http=None): return [1.0, 2.0, 3.0]
    res = summarize_pending(st, budget=10, embed_fn=embed_fn)
    assert res["summarized"] == 1
    n = st.get_node("ARES:chat/1")
    assert st.blob_to_vec(n["embedding"]).tolist() == [1.0, 2.0, 3.0]
    st.close()

def test_summarize_pending_gpu_on_loan_skips_entirely(tmp_path, monkeypatch):
    st = Store(str(tmp_path / "kg.db"))
    _seed_raw_chat(st, "ARES:chat/1", ["fix the caddy config"])
    monkeypatch.setattr("atlas.chats.gpu_on_loan", lambda: True)
    called = {"n": 0}
    def embed_fn(text, http=None):
        called["n"] += 1; return [1.0]
    res = summarize_pending(st, budget=10, embed_fn=embed_fn)
    assert res == {"summarized": 0, "note": "gpu-on-loan"}
    assert called["n"] == 0
    st.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && python3 -m pytest tests/test_chats.py -v`
Expected: FAIL — `TypeError: summarize_pending() got an unexpected keyword argument 'embed_fn'`

- [ ] **Step 3: Write the implementation**

Replace `summarize_pending` in `atlas/chats.py`:

```python
def summarize_pending(store, budget=500, kinds=("chat", "gpt-chat", "claude-chat"), embed_fn=None):
    """Generate Ollama summaries for chat/gpt nodes still marked status='raw', from the
    asks stored in meta — so it works for ANY box (ZEUS chats included) without the source
    file. Bounded by budget; run nightly to spread a big backfill across days. Also computes
    a semantic-search embedding for each freshly-generated summary, riding the same budget."""
    import json as _json
    from .understanding import summarize_chat, gpu_on_loan
    from . import embeddings as E
    embed_fn = embed_fn or E.embed
    if gpu_on_loan():
        return {"summarized": 0, "note": "gpu-on-loan"}
    ph = ",".join("?" * len(kinds))
    rows = store.db.execute(
        f"SELECT id, name, meta FROM nodes WHERE status='raw' AND kind IN ({ph}) LIMIT ?",
        (*kinds, int(budget))).fetchall()
    done = 0
    for r in rows:
        asks = (_json.loads(r["meta"] or "{}")).get("asks") or []
        if not asks:
            continue
        s = summarize_chat(asks)
        if not s:
            continue
        emb = store.vec_to_blob(embed_fn(s))
        with store.db:
            store.db.execute("UPDATE nodes SET understanding=?, status='live', embedding=? WHERE id=?",
                             (s, emb, r["id"]))
            store.db.execute("DELETE FROM nodes_fts WHERE id=?", (r["id"],))
            store.db.execute("INSERT INTO nodes_fts(id,name,understanding) VALUES(?,?,?)",
                             (r["id"], r["name"], s))
        done += 1
    remaining = store.db.execute(
        f"SELECT count(*) c FROM nodes WHERE status='raw' AND kind IN ({ph})", kinds).fetchone()["c"]
    return {"summarized": done, "still_raw": remaining}
```

Note: `gpu_on_loan` is imported into the `atlas.chats` module namespace via `from .understanding import ... gpu_on_loan`, which is why the tests monkeypatch `"atlas.chats.gpu_on_loan"` (the name as it exists in this module after the import), not `"atlas.understanding.gpu_on_loan"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && python3 -m pytest tests/test_chats.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas"
git add atlas/chats.py tests/test_chats.py
git commit -m "feat: compute+cache embeddings during chat/gpt-archive backfill"
```

---

### Task 6: Verification — historical query replay, threshold tuning, no-regression check

**Files:**
- Create: `scripts/replay_search_queries.py` (one-off verification tool, not a pytest test — depends on live `~/.claude` session transcripts outside the repo)

**Interfaces:**
- Consumes: `Store.search()` (Tasks 1-3, now with the semantic tier live).
- Produces: a printed report; no other code depends on this script.

This task requires real backfilled embeddings to be meaningful — it must run against the live production DB (`/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas/data/homelab_kg.db`) after at least one real backfill pass (Tasks 4/5's code, run via the existing nightly timers or manually) has populated some embeddings. If run before any embeddings exist, `all_embedded()` returns empty and every query silently stays at 2-tier behavior — not a bug, just nothing to verify yet.

- [ ] **Step 1: Write the replay script**

```python
# scripts/replay_search_queries.py
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
```

- [ ] **Step 2: Run it and record the baseline**

Run: `cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && KG_DB=data/homelab_kg.db python3 scripts/replay_search_queries.py`

Record the "returned something" percentage. Compare against the 59% success rate measured earlier today with the 2-tier (AND + quorum-OR) system, before this plan's changes existed. If embeddings haven't been backfilled yet (Tasks 4/5's code hasn't had a real nightly run), this number will match the 2-tier baseline exactly — that's expected, not a failure; re-run after the next nightly pass.

- [ ] **Step 3: Spot-check the known false-positive case**

```bash
cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && KG_DB=data/homelab_kg.db python3 -c "
from atlas.store import Store
st = Store('data/homelab_kg.db')
for q in ['can you help me write a poem about the ocean',
          'what is the capital of France and how many people live there']:
    hits = st.search(q, limit=5)
    print(q, '->', len(hits), [h.get('name') for h in hits])
"
```

Expected: 0 or very few hits, and any hits should be plausibly homelab-related, not "Ocean's 11 remake" or "Leadership in Process Management" style coincidental matches. If the semantic tier still surfaces obviously irrelevant results here, raise `SIM_FLOOR` in `atlas/store.py`'s `search()` (Task 3) — e.g. to 0.6 or 0.7 — and re-run this check.

- [ ] **Step 4: Verify zero regressions on queries that already worked**

```bash
cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas" && KG_DB=data/homelab_kg.db python3 -c "
from atlas.store import Store
st = Store('data/homelab_kg.db')
for q in ['EROS', 'zeus backup', 'dongle receiver', 'disk usage pve-root']:
    hits = st.search(q, limit=5)
    print(q, '->', len(hits), 'results')
    assert hits, f'REGRESSION: {q!r} used to work and now returns nothing'
print('no regressions')
"
```

Expected: `no regressions` printed, no AssertionError.

- [ ] **Step 5: Commit the verification script**

```bash
cd "/mnt/nvme/PROMETHEUS/PROJECTS/more projects/atlas"
git add scripts/replay_search_queries.py
git commit -m "test: add historical-query replay script for search-quality verification"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers `embeddings.py`; Task 2 covers schema + storage; Task 3 covers the search tier; Tasks 4-5 cover both backfill paths named in the spec (folder reindex + chat/gpt summarize-pending); Task 6 covers the spec's full Testing section (replay, false-positive spot-check, regression check). Latency spot-check (spec's Testing item 4) is implicitly covered by Task 6 Step 2's baseline run — no separate task needed since it's the same script.
- **Placeholder scan:** none found — every step has complete, runnable code.
- **Type consistency:** `embed_fn(text, http=None) -> list[float] | None` signature is identical across Tasks 1, 3, 4, 5. `Store.search(query, limit=20, embed_fn=None)` signature is used consistently. `vec_to_blob`/`blob_to_vec` names match between Task 2's definition and Tasks 3/4/5's usage.
