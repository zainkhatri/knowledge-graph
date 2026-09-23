# ATLAS Semantic Search Tuning — Design

## Problem

`Store.search()` (`atlas/store.py`) has three tiers: exact AND match, quorum-OR,
and — only when both of those return nothing — a semantic tier that embeds the
query via Ollama and returns anything above `SIM_FLOOR = 0.5` cosine similarity.

That semantic tier was added specifically to reduce the false-positive problem
from calling `kg_search` on every Claude Code prompt (per
`docs/plans/2026-09-19-atlas-semantic-search.md`), but its own verification
step (Task 6: spot-check known false positives, tune `SIM_FLOOR` accordingly)
was never completed. `SIM_FLOOR = 0.5` is still the untuned placeholder value
from the original plan.

Confirmed live on the production DB this session: unrelated queries like
*"can you help me write a poem about the ocean"* and *"what is the capital of
France and how many people live there"* still return irrelevant hits via tier
3, because nothing stops tier 3 from embedding and matching against a query
that shares zero vocabulary with anything in the graph. This is the exact
class of false positive the semantic tier was supposed to fix.

Separately: replaying 220 historical `kg_search` queries
(`scripts/replay_search_queries.py`) shows 97% return *something* — a recall
metric only, not a precision metric, so it doesn't answer whether ATLAS is
useful on its own. The real question is whether what tier 3 returns is
*relevant*, which is what this design fixes.

## Scope

This design covers **tightening tier 3 only** — the semantic-search fallback
in `atlas/store.py`'s `Store.search()`. It does not touch:

- Tier 1 (AND) or tier 2 (quorum-OR) matching logic
- The "call `kg_search` on every prompt" policy (`proactive-skill-usage`
  memory) — that decision stands; this design makes what tier 3 returns on an
  irrelevant prompt more likely to be nothing, without requiring Claude to
  judge relevance before calling the tool
- Anything outside `atlas/store.py` and its test/verification scripts

## Design

### 1. Tier-0 gate (new)

Before tier 3 calls `embed_fn` (the expensive, network-round-trip step), do a
cheap FTS5 `OR` probe across the query's content tokens (same
`_STOPWORDS`-filtered token list tier 2 already computes). If that probe
returns **zero rows** — meaning nothing in the entire corpus shares even one
content word with the query — skip tier 3 entirely and return `[]`. This
reuses the existing FTS5 index (no new data structures, no Ollama call) and
kills the "poem about the ocean" class of query before the expensive
embedding step ever runs.

Note this is different from tier 2's quorum threshold: tier 2 requires
`ceil(n/2)` token overlap to accept a match; the tier-0 gate only requires
**at least one** shared token to even attempt semantic scoring. A query can
pass tier-0 (get to attempt semantic matching) and still fail tier 3's
`SIM_FLOOR` check.

### 2. Retune `SIM_FLOOR`

Current value (0.5) is unvalidated. Retune using three test sets, all
re-runnable via a script (not manual spot-checks):

- **Regression set** (must keep returning results): existing known-good
  queries — "EROS", "zeus backup", "dongle receiver", "disk usage
  pve-root" — plus a sample of the 213 currently-passing queries from
  `scripts/replay_search_queries.py`'s historical corpus.
- **Negative set** (must return empty after the fix): "can you help me write
  a poem about the ocean", "what is the capital of France and how many
  people live there", plus 3-5 additional adversarial phrasings — common
  vocabulary, zero homelab relevance.
- **Positive semantic-only set** (must still match — proves tier 3 isn't
  just disabled): 2-3 real homelab queries phrased with no keyword overlap
  against their target node's stored `understanding` text. Sourced by
  reading a few indexed nodes' `understanding` fields and phrasing a
  differently-worded query around the same fact.

Search for the lowest `SIM_FLOOR` value (starting around 0.6-0.65 per the
original plan's own suggestion) where: negative set = 100% empty, regression
set = 100% still pass, positive semantic-only set = 100% still pass.

## Verification

A new script (extends or sits alongside `scripts/replay_search_queries.py`)
runs all three test sets against `Store.search()` and reports pass/fail
**per query**, not just an aggregate percentage, so it's re-runnable if the
corpus or query sets change later. This script is what determines the final
`SIM_FLOOR` value and confirms the tier-0 gate doesn't regress any known-good
query.

## Out of scope / explicitly not changing

- Tiers 1-2 matching logic
- The every-prompt call policy
- Any instruction-level "should I call kg_search" judgment in Claude's memory
  — the fix lives entirely in ATLAS's backend, not in when/whether the tool
  is invoked
