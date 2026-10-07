# Docker setup

`docker-compose.yml` runs `db` (Postgres 16), `migrate` (one-shot Alembic upgrade), `api`
and `worker`. All published ports bind to `127.0.0.1` (API `8400`, Postgres `55432`); the
services share the internal `studio` network. The Docker socket is never mounted.

| Volume | Purpose |
|---|---|
| `./data` → `/app/data` | asset store (`data/projects/<id>/…`), fixtures, audit |
| `${STUDIO_MEDIA_DIR:-./media}` → `/media` (read-only) | your source videos and references |
| `./workflows`, `./config` (read-only) | templates and configuration |
| `pgdata` | database |

Scale workers with `docker compose up -d --scale worker=2`; the GPU lease still allows one
heavy render at a time.

## Tests

Tests need FFmpeg and a Postgres database they may wipe:

```bash
createdb rokkur_test
TEST_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/rokkur_test pytest
```

Against the compose database: `TEST_DATABASE_URL=postgresql+psycopg://rokkur:<pw>@127.0.0.1:55432/rokkur_test`
after `docker compose exec db createdb -U rokkur rokkur_test`.
