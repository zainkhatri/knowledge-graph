# Session memory: save and summarize every Claude Code session

Date: 2026-09-24. Status: approved (design approved in chat).

## Goal
Every Claude Code session on every homelab box is saved forever, summarized
from its whole transcript, searchable in homelab-kg, and surfaced as context
when a new session starts.

## Problems this fixes
- Claude Code deletes transcripts after 30 days (`cleanupPeriodDays` default).
  ARES had nothing older than 2026-08-25.
- Only ARES host chats are indexed each night. ZEUS chats were a one-off import
  (stale since 2026-09-03; ZEUS root never indexed). EROS, LXC 101 and the Mac
  are never indexed.
- The indexer reads only the first 400 lines / 6 asks of each session.
- Summaries use a small local Ollama model at 250/night with a large backlog.

## Components

### 1. Archive — `ARES-DASHBOARD/system/kg-sync-sessions.sh`
Runs hourly (`kg-sessions.timer`) on the ARES host. For each source, rsync
(`-a`, never `--delete`) `.claude/projects/` into
`/mnt/nvme/PROMETHEUS/PERSONAL/CLAUDE-CODE-SESSIONS/<SRC>/`:

| SRC | Box id | Source |
|---|---|---|
| ARES | ARES | `/root/.claude/projects` |
| ARES-LXC101 | ARES | `/proc/<lxc101 pid>/root/root/.claude/projects` |
| ZEUS-zain | ZEUS | `zeus:/home/zain/.claude/projects` |
| ZEUS-root | ZEUS | `zeus:/root/.claude/projects` |
| EROS | EROS | `eros:/root/.claude/projects` |
| MAC | MAC | `mac:~/.claude/projects` |

Unreachable sources are skipped (bounded SSH timeouts); the next run catches up.
Each reachable box gets `cleanupPeriodDays: 36500` re-asserted in its
`~/.claude/settings.json`. Archive dir is mode 700.

### 2. Index — `atlas/chats.py:index_chats`
- Reads the archive root for each SRC with that source's box id.
- Whole-transcript parse via `atlas/transcript.py` (bounded line count, cheap
  prefilter). Keeps sessions with no human asks (named `session <sid8>`).
- Skips a file whose fingerprint (`mtime:size`) is unchanged.
- Node: `<BOX>:chat/<sid>`, kind `chat`, path = archive file. Meta stores cwd,
  title (`ai-title`/`custom-title`), last timestamp, up to 12 sampled asks.
- Fallback `understanding` (before summary) = title + sampled asks, ≤1500 chars.
- Never deletes nodes; a session whose source vanished keeps its node.

### 3. Summarize — `atlas/redact.py`, `atlas/understanding.py`
- `redact(text)` replaces secrets with `[REDACTED]`: `sk-…`/`sk-or-…`/`sk-ant-…`,
  `ghp_/gho_/github_pat_`, `xox[abpr]-`, `AKIA…`, `AIza…`, JWTs, `Bearer <tok>`,
  `NAME=value` / `"name": "value"` where NAME contains key/token/secret/passw/
  api/auth/cred, PEM private-key blocks, `scheme://user:pass@`, and long
  mixed-case+digit random strings (≥32 chars). Paths and hex hashes are kept.
- `summarize_session(digest, title, cwd, box)` → OpenRouter chat completion,
  model `KG_SUMMARY_MODEL` (default `google/gemini-2.5-flash-lite`), key from
  `OPENROUTER_API_KEY` env or `~/.claude/settings.json`. 3-6 sentence summary,
  ≤1500 chars, stored in `understanding` (FTS-indexed), status `live`.
- Every prompt sent anywhere goes through `redact` first (sessions, GPT and
  claude.ai asks alike). Ollama is the fallback only when no key exists.
- Summarize only sessions idle ≥15 min; re-summarize when the file changes.
- Parallel: 8 worker threads for HTTP; DB writes on the main thread.
- Guards: per-run `--summary-budget`; HTTP 402 raises `OutOfCredit` and ends
  the run cleanly.
- `summarize-pending` (GPT / claude.ai raw nodes) uses the same path.

### 4. Context — `/root/.claude/helpers/kg-session-context.py`
`SessionStart` hook on ARES. Reads the graph DB read-only, selects the 6 most
recent `chat` nodes whose cwd equals or is under/over the session cwd
(excluding the current session), prints `additionalContext` with date, title,
summary (≤400 chars each) and a pointer to `kg_search`. Fails silent, <1 s.

### 5. Nightly
`kg-nightly.sh` step 1b calls `kg-sync-sessions.sh` instead of
`index-chats`; step 1d `summarize-pending` uses OpenRouter.

## Testing
- `tests/test_redact.py`: each secret pattern redacted; paths, hashes, UUIDs,
  normal prose unchanged.
- `tests/test_transcript.py`: whole-file parse, title, noise filtering, digest
  head/sample/tail within budget.
- `tests/test_chats.py`: unchanged-file skip, empty-session kept, summaries use
  mocked HTTP, prompt is redacted, idle gate, OutOfCredit stops the run.
- Manual: inspect 3 real redacted digests + summaries before the backfill.

## Out of scope
Summarizing subagent transcripts (archived only). Mac-side hook.
