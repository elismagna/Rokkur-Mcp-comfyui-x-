# YouTube

Publishing is manual: nothing in the pipeline uploads on its own. A finished project sits in
`READY_TO_PUBLISH` until a person presses **Upload to YouTube** on its page, runs `publish`,
calls the API with `dry_run: false`, or approves an upload request (autonomy level 3, below).
Uploads are **private** unless you ask for `unlisted`; `public` needs both an explicit
`--privacy public` and `youtube.allow_public: true`. A scheduled release goes public by
itself later, so it needs `youtube.allow_public: true` too.

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

On the project page, under **YouTube**: edit the title, description and tags, pick **Now** or
**Upload as private now, make public at a set time**, the visibility and a playlist, then
**Upload to YouTube** (or **Dry run** to check everything without sending anything).

From the command line:

```powershell
.\scripts\studio.ps1 publish <project id>                     # private, asks for confirmation
.\scripts\studio.ps1 publish <project id> --dry-run           # show the exact request, send nothing
.\scripts\studio.ps1 publish <project id> --privacy unlisted
.\scripts\studio.ps1 publish <project id> --at "2026-10-09 18:00"   # private now, public then
.\scripts\studio.ps1 publish <project id> --at next --playlist "Shorts"
```

The project id is printed by `render`. `publish` shows the title, tags, any warnings and the
plan first, then asks `Upload to YouTube (private)? [y/N]`. On success it prints the watch URL.

## Scheduled release

YouTube can make a private video public at a set time (`status.publishAt`). The studio uploads
the video as private right away with that time; your PC does not have to be on when it goes
public. Rules (`services/publishing.py::resolve_schedule`):

- needs `youtube.allow_public: true`, because the video becomes public;
- the visibility is private until then (YouTube requires that);
- at least `youtube.min_lead_minutes` (30) ahead, so the upload and processing finish first.

Times typed on the command line (`--at "2026-10-09 18:00"`) are read in `youtube.timezone`;
the dashboard uses your browser's time zone. Set daily release slots to get a suggestion:

```yaml
youtube:
  allow_public: true
  timezone: Europe/Oslo
  release_times: ["18:00"]       # quote them; several are fine: ["12:00", "18:00"]
```

The next free slot (not already used by a scheduled upload or a waiting upload request) is
pre-filled on the project page, shown on the **YouTube** page with everything already
scheduled, and used by `--at next`.

## Playlists

Load your channel's playlists once (and again after you add one on YouTube): **Load
playlists** on the YouTube page, or `.\scripts\studio.ps1 youtube-playlists`, which also
prints their ids. The list is kept in `data/youtube/playlists.json` (ids and titles only).
Then pick one when publishing, or set `youtube.default_playlist_id` to preselect it. Adding
the video to the playlist happens right after the upload (`playlistItems.insert`, 50 quota
units); if it fails, the video stays up and the failure is shown as a warning.

## Upload requests (autonomy level 3)

At autonomy level 3 or 4 (`studio.autonomy_level` or the channel's level) with uploads on, a
finished video does not just wait: the studio checks it with a dry run and puts an upload
request on the **Approvals** page with the complete plan, for example *Upload "Neon alley
#shorts" as private; it goes public Fri 9 Oct 18:00 CEST; add it to the playlist "Shorts"*.
The plan uses the channel's default visibility (never public directly), the next free release
time when release times are set and public uploads are allowed, and the default playlist.

- **Approve and upload** sends it exactly as planned. If the planned time has passed, the
  request is refused; reject it and schedule the video from its page.
- **Reject** leaves the video in `READY_TO_PUBLISH`; you can still publish it by hand.
- Publishing the video by hand closes its open request (`superseded`).
- If the video can't be uploaded as planned (for example its metadata has a warning), no
  request is made and the project's activity log says why (`PUBLISH_PROPOSAL_SKIPPED`).

API: `GET /approvals`, then `POST /approvals/{id}` with `{"approve": true}` uploads.

What it checks before sending anything (`services/publishing.py`): project is
`READY_TO_PUBLISH`, rights approved, latest QC `PASS`, metadata valid (title ≤ 100 chars, no
`<`/`>`, description ≤ 5000 bytes, tags ≤ 500 chars) and without warnings (a Short over 180 s
is refused rather than uploaded as a regular video by accident), final render on disk.

What it sends: resumable `videos.insert` (`snippet` + `status` with
`selfDeclaredMadeForKids: false`, `containsSyntheticMedia: true`, your privacy and, when
scheduled, `publishAt`), then `thumbnails.set` with the generated thumbnail, then
`playlistItems.insert` when a playlist was picked. A thumbnail or playlist failure is recorded
as a warning on the publication; the video stays up.

Audit trail: `READY_TO_PUBLISH → PUBLISHING → PUBLISHED` with the video id and URL in the
`VIDEO_PUBLISHED` event; on failure the project returns to `READY_TO_PUBLISH` with the API
error on the `publications` row. Quota units (1600 per upload, 50 per thumbnail, 50 per
playlist insert, 1 per channel or playlist lookup) go to `cost_entries` as `youtube_quota`; the
free daily quota is 10 000 units, so about six uploads a day.

API: `POST /projects/{id}/publish` with `{"dry_run": false, "privacy": "unlisted"}`; add
`"publish_at": "2026-10-09T16:00:00Z"` for a scheduled release and `"playlist_id"` for a
playlist. `GET /youtube/playlists`, `POST /youtube/playlists/refresh` and
`GET /youtube/releases` back the YouTube page.

## Dry run

`POST /projects/{id}/publish` with `dry_run: true` (the default), or `publish --dry-run`,
runs the same gates and records the exact request in `publications` without an upload. The
pipeline does this automatically at the end of every render so you can inspect what would go out.

## Not built

Uploading without a person's click (no autonomy level does this), editing an upload request's
plan on the Approvals page (reject it and publish from the project page instead), and Phase 5
discovery (`search.list` / `videos.list` with caching and quota accounting; discovered videos
start as `DISCOVERED` with rights `UNKNOWN` or `REFERENCE_ONLY` and never ingest without a
human decision).
