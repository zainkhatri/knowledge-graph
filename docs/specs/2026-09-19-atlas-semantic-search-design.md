# ATLAS semantic search fallback — design

## Context

Same-day audit of ATLAS (`homelab-kg`) found real, measured problems:
- Only 4.1% of Claude Code sessions ever call it (60/1448).
- Of 186 real historical `kg_search` calls, 66% returned empty, 17% crashed, only 16% returned something useful.
- The crash was already fixed by existing FTS5 quoting. The empty-result problem was fixed today by adding an AND-first/quorum-OR fallback in `store.search()` — replayed against the same 188 real queries: success rate went 16%→59%, zero regressions.
- That fix has a hard ceiling: FTS5/bm25 lexical ranking cannot distinguish "topically relevant" from "coincidentally contains a rare word." Proven case: an off-topic poem-writing query outscored several genuine homelab hits by matching the word "ocean" in an unrelated chat title. No amount of lexical tuning fixes this — it needs semantic (embedding-based) similarity.

This spec covers adding a semantic search tier to close that gap.

## Goals

- When lexical search (AND, then quorum-OR) returns nothing, fall back to real semantic similarity search instead of nothing.
- Reduce false-positive noise on off-topic queries (the "ocean poem" class of failure) without needing to hand-tune more lexical heuristics.
- Zero new installs: use `nomic-embed-text` (already pulled in Ollama) for embeddings, numpy (already present) for similarity math.
- No regression to existing behavior — semantic tier only fires when both lexical tiers come back empty.

## Non-goals

- Not replacing lexical search as the primary path. Exact/near-exact matches should stay fast and free (no Ollama round-trip) — semantic is strictly a fallback tier.
- Not building a vector database or ANN index. At current scale (~25k nodes) brute-force cosine similarity over one numpy matrix is low-milliseconds; a real vector index would be premature optimization.
- Not changing the CLAUDE.md scoping instruction (homelab-relevant queries only) — that's a separate, already-resolved concern.

## Architecture

### New module: `atlas/embeddings.py`

Mirrors `understanding.py`'s existing pattern exactly (same `OLLAMA_HOST` env var, same `gpu_on_loan()` guard, same fail-soft contract):

```python
def embed(text: str) -> list[float] | None:
    """Returns a 768-dim embedding vector, or None on any failure
    (GPU on loan, Ollama unreachable, timeout, empty text)."""
```

Calls Ollama's `/api/embeddings` endpoint with `model=nomic-embed-text`. On any exception, returns `None` — callers must treat `None` as "skip this, don't crash."

### Schema change: `atlas/store.py`

Add one nullable column to the existing `nodes` table:

```sql
ALTER TABLE nodes ADD COLUMN embedding BLOB;
```

Stored as `numpy.array(vec, dtype='float32').tobytes()`. `NULL` for any node that hasn't been embedded yet (all existing nodes, until backfilled) — never a zero-vector placeholder, which would corrupt similarity math.

### Write path: backfill

Extend the existing budgeted backfill mechanisms — do NOT create a new timer:
- `collect.py`'s folder reindex (retry-budget pattern added earlier today) also computes and stores an embedding whenever it (re)generates understanding text for a node.
- `summarize-pending` (the chat/gpt-archive backfill) does the same for chat/skill/mcp/agent nodes when it fills in their summary.

Both already have per-run budget caps for Ollama calls; embedding is one more call per already-budgeted unit of work, not a new uncapped cost source. A node with understanding text but no embedding (e.g. generated before this feature shipped) gets its embedding filled in the next time that node is touched by a budgeted pass — same self-healing pattern as the understanding-text backlog fix.

### Read path: `store.search()`

Three tiers, in order, stopping at the first that returns results:

1. **AND** (unchanged) — exact, all query tokens must match.
2. **Quorum-OR** (unchanged, added today) — at least half the content tokens (stopwords excluded) must match, ranked by overlap count. Zero Ollama cost, stays as the free middle tier.
3. **Semantic** (new) — embed the query via `embed()`. If that returns `None` (Ollama down/GPU on loan), return whatever tier 2 gave (likely empty) — do not error. Otherwise, load all rows with non-null `embedding`, compute cosine similarity in one vectorized numpy operation, return the top N above a similarity floor.

The similarity floor needs empirical tuning (start conservative — see Testing) rather than a guessed constant.

## Error handling

- `embed()` returns `None` for any failure → search falls through to tier-2 result (possibly empty) — never raises up to the MCP layer.
- Backfill: embedding failure is caught independently and does not block understanding-text generation for that node — a node can have understanding text with a still-null embedding, picked up on a later pass.
- Existing nodes with `embedding IS NULL` are simply excluded from the similarity matrix — not a zero-vector false match.

## Testing plan

Before considering this done:
1. Replay the same 188 real historical `kg_search` queries (harness already built today) through the new 3-tier search. Compare success rate against today's 2-tier baseline (16%→59%) — expect a further increase on queries that still return empty today.
2. Specifically test the known false-positive case ("what is the capital of France..." / "write a poem about the ocean" style queries) — verify semantic similarity scores these below any real threshold, rather than assuming it will.
3. Verify zero regressions: every one of the 111 queries that already succeed under the 2-tier system must still succeed identically (tier 1/2 unchanged, tier 3 never overrides an earlier hit).
4. Spot-check latency: confirm the tier-3 Ollama embedding call adds acceptable latency (target: comparable to the ~3.5s/call already measured for understanding-text generation) only on the minority of searches that reach it.

## Rollout

- Ship `embeddings.py` + schema migration + search tier first, with the backlog of un-embedded nodes present (nothing breaks — they're just excluded from tier 3 until backfilled).
- Backfill rides the existing nightly/deep-backfill timers — no new cron job.
- No changes needed to `mcp_server.py`'s tool interface (`kg_search`'s signature is unchanged; the extra tier is internal to `store.search()`).
