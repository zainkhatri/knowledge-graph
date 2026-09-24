# Session memory — implementation plan

Spec: `docs/superpowers/specs/2026-09-24-session-memory-design.md`.
Execution: inline, TDD per task, commit per task.

1. **redact.py** — write `tests/test_redact.py` (patterns + non-redaction cases),
   run red, implement `atlas/redact.py`, run green.
2. **transcript.py** — write `tests/test_transcript.py` for the existing module
   (title, noise, whole-file asks, digest budget); fix until green.
3. **understanding.py** — OpenRouter `llm()`, `summarize_session`,
   `OutOfCredit`, redaction inside `llm()`; keep `summarize_chat` signature.
   Tests with injected http; update the Ollama gpu-loan test expectations.
4. **chats.py** — `index_chats`: transcript read, fingerprint skip, keep empty
   sessions, idle gate, threaded summaries, budget, OutOfCredit.
   `summarize_pending`: drop the global gpu-loan early return when a key
   exists; use `summarize_session` from the archive file when it exists.
   Update `tests/test_chats.py`.
5. **cli.py** — `index-chats` gains `--min-idle`, `--workers`, `--no-summarize`.
6. **kg-sync-sessions.sh** + `kg-sessions.service/.timer` (hourly, lock with
   `flock`, timeout 3h). Wire `kg-nightly.sh` step 1b to it.
7. **SessionStart hook** `kg-session-context.py` + register in settings.json.
8. **Verify** — full pytest; dry run on 3 real sessions (print redacted digest
   + summary); run sync; backfill with budget monitoring; check coverage =
   archive top-level count vs chat nodes per box; check hook output.
