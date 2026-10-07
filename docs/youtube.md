# YouTube

## Today: dry-run publishing

`POST /projects/{id}/publish` with `{"dry_run": true, "privacy": "private",
"publish_at": "...", "playlist_id": "..."}` checks the publication gates (state
`READY_TO_PUBLISH`, rights approved, latest QC `PASS`, final render present), validates
metadata (title ≤ 100 chars, no `<`/`>`, description ≤ 5000 bytes, tags ≤ 500 chars,
scheduled publishing requires `private`), and records the exact `videos.insert` request
(`snippet` + `status` with `selfDeclaredMadeForKids` and `containsSyntheticMedia: true`)
in `publications`. `dry_run: false` returns 501.

Metadata is drafted by the editor stage: `#shorts` title suffix for Shorts, AI-assistance
note, attribution from the rights decision, and a warning if a Short exceeds 180 s.

## Phase 6 (not built)

OAuth installed-app flow (`make youtube-auth`), refresh tokens in the OS credential store or
`./secrets` (git-ignored, never in the repo or database), resumable upload, `thumbnails.set`,
playlist insertion, quota ledger in `cost_entries`, and an approval request before every
public upload unless the channel's autonomy level and policy allow it.

## Phase 5 (not built)

Discovery through `search.list` / `videos.list` with caching and quota accounting.
Discovered videos become `DISCOVERED` projects with rights `UNKNOWN` or `REFERENCE_ONLY`,
which never ingest without a human decision.
