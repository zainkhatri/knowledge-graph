# Photo descriptions (local vision), My Eyes Only excluded

Date: 2026-10-04. Status: shipped. Council-reviewed (5 advisors) the same day.

## Goal
Every photo under `PHOTOS/` gets a one-or-two sentence description from a local
vision model, merged with any OCR text, so agents can find photos by what is in
them ("bedside lamp", "colonnade laptop"). Owner requirement, verbatim: every
photo EXCEPT anything in My Eyes Only — "extremely important".

## Why it was needed
The 2026-09-25 content pass had a vision fallback (`llava:7b`) that never ran: no
vision model was ever installed, so 32.5k of 43.8k photo nodes were status
`empty` and none had a description.

## Shipped
- `atlas/photo_vision.py`: `VaultGuard`, `describe_pending`, `prune_photos`,
  `visible`, `prepare_image`, `clean`. 13 tests in `tests/test_photo_vision.py`
  plus one in `tests/test_mcp_server.py`.
- CLI: `kg describe-photos <root> --budget N --minutes M` (always prunes first),
  `kg prune-photos <root>`, `kg index-content ... --vault-guard`.
- Model: `gemma3:4b` on EROS (GTX 1070) via ARES's `:11434` socat proxy, ~15-20s
  per photo at 768px. Benchmarked against `qwen2.5vl:3b` (~22s, a bit more detail)
  and `moondream` (unusable output).
- `kg-photo-vision.timer`: 22:00 (until 01:55) and 02:30 (until 05:05), keeping
  EROS free for `kg-deep-backfill` (02:00) and `kg-nightly` (05:15). Roughly 1,200
  photos per night, about five weeks for the backlog. Newest photos first.
- The content pass no longer calls a vision model at all; photos get
  descriptions only through the guarded path above.

## The vault boundary (fails closed, layered)
1. Path rules on the path and its realpath: any dot-directory component (the
   real vault is `PHOTOS/.vault/`), `walker.default_vault_pred()`, no symlinks.
2. The dashboard's vault index (`ai_data/vault.json`): every vaulted item's
   original library path, and its file name anywhere in the library (catches a
   stray copy; costs ~375 photos that share a name with a vaulted item).
3. Known content hashes of vaulted items (`content_hashes.json`; only 4 of 559
   are known), checked against the bytes actually read.
4. Re-checked after the model returns; a photo vaulted mid-run is dropped and its
   node deleted. Nodes the guard rejects are deleted, not skipped.
5. Unreadable/empty vault index: `VaultIndexError`, the run aborts.
6. Query time (`mcp_server.py`): photo-derived nodes are shown only while the
   guard allows their file, reloaded every 60s; no guard means no photo nodes.
   Covers search, get, neighbors (both edge ends) and tree.
7. Nightly tripwire: any node under a PHOTOS dot-directory exits the job non-zero.

Vault files are never opened. The vault index is read only to build the sets.

## Known limits
- A copy of a vaulted photo saved under a different file name, with no known
  content hash, would be described. Closing this fully means hashing the vault
  originals (reading their bytes), which needs the owner's explicit OK.
- Descriptions are AI-generated and can be wrong; nodes carry `vision_model`,
  `vision_v`, `described_at`, and agents see an "untrusted data" note.
- First live prune (2026-10-04) removed 1,704 stale photo nodes (1,329 for files
  no longer at their path). The pre-change backup
  `data/homelab_kg.pre-photo-vision-2026-10-04.db` still holds those rows.

## Faces as context (v2, same day)
`FaceIndex` reads the dashboard's named face clusters (`ai_data/face_clusters.json`,
125 named of 135) and its photo index (`photo_index.db`, path -> thumb key), giving
names for 18,252 photos. Those names go into the prompt ("Face recognition
identified these people...; do not add any other names") and into the node as a
`People: A, B` line plus `meta.people`, so "hamza cafe laptop" finds the photo.
Excluded faces (`excluded_hashes`) are honored. Names are looked up only for
photos the VaultGuard already allowed. Missing face files just mean no names.
`VISION_VERSION` 2 re-describes the v1 pilot photos.

## High-confidence faces only (v4, same day)
The v2 names came straight from the dashboard's cluster tags. Calibration showed those
are loose: no human labels exist (0 seed / 0 excluded photos), the dashboard's expand
step accepts nearest-exemplar distance < 1.05 with a 0.02 lead, and of 28,182 name tags
only 22% survive a strict check (26% have no detected face, 18% ambiguous, 17% closest
to someone else, 12% too far, 5% weak detection).
Root cause found: the stored cluster `exemplars` are not representative (their centroid
sits ~1.0 from the cluster's own faces for Zain/Hamza/Bronny; bronny/zaeem and
mohsin/bholat share an identical exemplar). `FaceIndex` now builds each profile from
the cluster's own faces (`emb_indices`, trimmed centroid) and names a face only if:
det >= 0.7, distance <= 1.00 (0.01% impostor quantile is 1.067; genuine median 0.52),
lead >= 0.10 over the next person, and the cluster tags the photo. Omar / Omar Saleem
merge (close + shared word); Bronny / Zayd are close without a shared word and stay
separate (faces between them get no name). Result: 10,733 photos with confident names.
Dashboard data issues found (not changed here, the dashboard owns its face DB): stale
exemplars, two shared exemplars, possible duplicates bronny~zayd and tejas~jason.
