# Content indexing: OCR photos and extract text from documents

Date: 2026-09-25. Status: approved (design approved in chat).

## Goal
`kg_search` currently only matches folder-level summaries and chat text — it
has zero visibility into what's actually inside a file. Add a new indexing
pass that extracts text from photos (OCR) and documents (PDF/docx/txt/md) and
makes it searchable the same way everything else in the graph is.

## Problem this fixes
Asked to find a VIN, `kg_search` found chat sessions *about* the car but
nothing from the actual insurance-card photo, because no pass ever looks at
file content — only folder structure and chat transcripts are indexed today.
Finding the photo required a manual filename grep across `PERSONAL/` as a
fallback. Full-library OCR/doc-text indexing removes that fallback.

## Scope
- Every file under an indexed root, all boxes, all folders — **except** any
  path the existing `walker.default_vault_pred()` already excludes (My Eyes
  Only). Content indexing must never open a file under a vault-excluded path.
- Data never leaves the box: all extraction runs locally (Tesseract OCR,
  Ollama vision fallback, local PDF/docx parsers). No cloud OCR/vision APIs.
- File types: images (jpg/jpeg/png/heic/webp/…), PDF, docx, txt/md.

## Components

### 1. `atlas/content.py` — `index_content(st, root, box, budget=500, workers=4)`
New module, same shape as `atlas/chats.py`/`atlas/env.py`. Walks files (not
folders — `atlas/walker.py` already owns folder nodes and is untouched) under
`root`, reusing:
- `walker.IGNORE_DIRS` — skip the same dev/build noise dirs.
- `walker.default_vault_pred()` — any path matching a vault pattern is
  skipped entirely; the file is never opened, no node is created for it.

For each remaining file:
1. Compute a per-file fingerprint: `sha256(f"{mtime}:{size}")[:16]`.
2. If a node already exists at this path with the same fingerprint, skip —
   this is what makes repeated full-library runs cheap after the first pass.
3. Otherwise dispatch to an extractor by extension (below), capped at
   `budget` newly-processed files per invocation.

### 2. Extractors, by extension
- **Images** (jpg/jpeg/png/heic/webp/heif): shell out to
  `tesseract <path> stdout -l eng`. If the result has fewer than 8
  alphanumeric characters (the common case — most personal photos have no
  text), fall back to a local Ollama vision model over HTTP for a short
  description instead, tagging `meta.method="ollama-vision"` so search results
  can distinguish "exact OCR text" from "a description of the image."
- **PDF**: `pdftotext <path> -`.
- **docx**: `python-docx`, if installed.
- **txt/md**: read directly, capped at ~200KB with a `meta.truncated=true`
  note past the cap.

Binary/dependency availability (`tesseract`, `pdftotext`, an Ollama vision
model, `python-docx`) is checked once at the start of a run. A missing
dependency skips that whole file type for the run with a single warning —
never a crash, never per-file spam.

### 3. Storage — no schema change
Files currently get no node at all (only folders/clusters do), so this is a
new `kind`, not a new table:
- `id = f"{box}:{path}"` (unique — no collision with any existing folder node).
- `kind = "file-content"`.
- `understanding` = the extracted text (or truncated text / vision
  description), which is the existing FTS-indexed column — `kg_search` picks
  these up with zero changes to search or the FTS schema.
- `fingerprint` = the per-file hash above (reuses the existing column).
- `meta = {"method": "tesseract"|"ollama-vision"|"pdftotext"|"docx"|"raw",
  "ext": ..., "truncated": bool}`.
- An edge `(parent_folder_id, file_id, "contains")` — the same edge type
  `walker.py` already uses for folder→folder, so `kg_tree`/`children()` show
  file-content nodes under their folder for free.

### 4. CLI — `kg index-content <root> --box ARES --budget 500 --workers 4`
Registered in `atlas/cli.py` next to the other `index-*` subcommands.

### 5. Scheduling
A new `kg-content.timer`, separate from the existing nightly folder-walk
timer, so a slow OCR backlog across 52K+ photos never delays that run. Budget
default (500/run) means the first full pass over the photo library takes
many nights; every run after that only touches new/changed files.

## Error handling
Any per-file exception (corrupt image, unreadable PDF) is caught, the node is
written with `status="error"` and `meta.error=<str>`, and the walk continues
— same catch-and-continue discipline `walker.py`/`fingerprint.py` already
use for `OSError`. No new assertions beyond existing input boundaries; a
missing extractor binary is a startup-time check, not a per-file failure.

## Testing
- Fingerprint-based skip: unchanged file is not re-processed; changed file is.
- Vault-path exclusion: a file under a vault-excluded path is never opened
  and gets no node.
- Extension → extractor dispatch, with subprocess calls mocked.
- Tesseract → Ollama-vision fallback triggers when OCR output is too short.
- Graceful degrade when `tesseract`/`pdftotext`/`python-docx`/vision model is
  missing — file type is skipped for the run, no crash.
- txt/md truncation past the size cap.

## Out of scope
- Re-implementing or duplicating ARES-DASHBOARD's own vault-exclusion logic —
  ATLAS already has its own (`walker.default_vault_pred()`); this reuses it,
  it does not read `vault.json` or any dashboard-owned exclusion list.
- Re-ranking or otherwise changing how `kg_search` scores results — new
  `file-content` nodes are picked up by the existing FTS index unchanged.
- OCR quality tuning beyond the tesseract→vision fallback (e.g. preprocessing,
  rotation correction) — a future iteration if search quality still misses.
