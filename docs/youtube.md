# YouTube

Publishing is manual: nothing in the pipeline uploads on its own. A finished project sits in
`READY_TO_PUBLISH` until a person runs `publish` (or calls the API with `dry_run: false`).
Uploads are **private** unless you ask for `unlisted`; `public` needs both an explicit
`--privacy public` and `youtube.allow_public: true`.

## One-time setup (about 10 minutes)

The studio talks to YouTube as *you*, through a Google OAuth client you own. Google requires
each app to have its own client, so this is a per-studio step:

1. Open [Google Cloud console](https://console.cloud.google.com/) and create a project
   (any name, e.g. "Rokkur Studio").
2. **APIs & Services → Library**: enable **YouTube Data API v3**.
3. **APIs & Services → OAuth consent screen**: user type *External*, fill in the app name and
   your email, save. Under **Test users** add the Google account that owns your channel.
   (The app stays in "Testing"; that is fine for your own channel. Refresh tokens issued in
   testing mode expire after 7 days, so re-run `youtube-auth` when `publish` says the token
   is invalid, or publish the consent screen to stop that.)
4. **APIs & Services → Credentials → Create credentials → OAuth client ID**, application type
   **Desktop app**. Download the JSON.
5. Save it as `secrets/youtube_client_secret.json` in the studio folder. `secrets/` is
   git-ignored and mounted into the containers at `/app/secrets`; nothing in it is ever
   committed, logged or stored in the database.
6. Sign in once:

   ```powershell
   .\scripts\studio.ps1 youtube-auth
   ```

   It prints a Google link. Open it, allow access, and Google sends the browser back to
   `127.0.0.1:8401`, which the command is listening on. It then prints the channel it signed
   in as. The refresh token is written to `secrets/youtube_token.json` (owner-only permissions).
   If the browser cannot reach that port, run `youtube-auth --paste` and paste the address the
   browser landed on. `youtube-auth --status` checks the saved sign-in; `--sign-out` deletes it.
7. Turn uploads on in `.env`: `STUDIO_YOUTUBE__ENABLED=true`, then `.\scripts\studio.ps1 up`.

## Publishing a video

```powershell
.\scripts\studio.ps1 publish <project id>                     # private, asks for confirmation
.\scripts\studio.ps1 publish <project id> --dry-run           # show the exact request, send nothing
.\scripts\studio.ps1 publish <project id> --privacy unlisted
```

The project id is printed by `render`. `publish` shows the title, tags and any warnings first,
then asks `Upload to YouTube as private? [y/N]`. On success it prints the watch URL.

What it checks before sending anything (`services/publishing.py`): project is
`READY_TO_PUBLISH`, rights approved, latest QC `PASS`, metadata valid (title ≤ 100 chars, no
`<`/`>`, description ≤ 5000 bytes, tags ≤ 500 chars) and without warnings (a Short over 180 s
is refused rather than uploaded as a regular video by accident), final render on disk.

What it sends: resumable `videos.insert` (`snippet` + `status` with
`selfDeclaredMadeForKids: false`, `containsSyntheticMedia: true`, your privacy), then
`thumbnails.set` with the generated thumbnail. A thumbnail failure is recorded as a warning on
the publication; the video stays up.

Audit trail: `READY_TO_PUBLISH → PUBLISHING → PUBLISHED` with the video id and URL in the
`VIDEO_PUBLISHED` event; on failure the project returns to `READY_TO_PUBLISH` with the API
error on the `publications` row. Quota units (1600 per upload, 50 per thumbnail, 1 per channel
lookup) go to `cost_entries` as `youtube_quota`; the free daily quota is 10 000 units, so about
six uploads a day.

API: `POST /projects/{id}/publish` with `{"dry_run": false, "privacy": "unlisted"}`.
`publish_at` (scheduled publishing) is accepted only with `private`, as YouTube requires.

## Dry run

`POST /projects/{id}/publish` with `dry_run: true` (the default), or `publish --dry-run`,
runs the same gates and records the exact request in `publications` without an upload. The
pipeline does this automatically at the end of every render so you can inspect what would go out.

## Not built

Playlist insertion, scheduled publishing via the CLI, an approval request before public
uploads at higher autonomy levels, and Phase 5 discovery (`search.list` / `videos.list` with
caching and quota accounting; discovered videos start as `DISCOVERED` with rights `UNKNOWN`
or `REFERENCE_ONLY` and never ingest without a human decision).
