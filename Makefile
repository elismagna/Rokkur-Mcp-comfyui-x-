# Docker-first commands. On Windows without make, use scripts/studio.ps1 <command>.
COMPOSE ?= docker compose
RUN = $(COMPOSE) run --rm api

.PHONY: up down logs test migrate studio worker comfy-check agent-check youtube-auth publish youtube-playlists prompt-schedule taste timings smoke-test comfy-render render audit lint build

build:        ; $(COMPOSE) build migrate
up:           ; $(COMPOSE) build migrate && $(COMPOSE) up -d
down:         ; $(COMPOSE) down
logs:         ; $(COMPOSE) logs -f --tail=200
migrate:      ; $(RUN) rokkur-studio migrate
studio:       ; $(COMPOSE) up -d api
worker:       ; $(COMPOSE) up -d worker
comfy-check:  ; $(RUN) rokkur-studio comfy-check
youtube-auth: ; $(COMPOSE) run --rm -p 127.0.0.1:8401:8401 api rokkur-studio youtube-auth
publish: ; $(RUN) rokkur-studio publish $(ARGS)
prompt-schedule: ; $(RUN) rokkur-studio prompt-schedule $(ARGS)
taste: ; $(RUN) rokkur-studio taste
timings: ; $(RUN) rokkur-studio timings $(ARGS)
youtube-playlists: ; $(RUN) rokkur-studio youtube-playlists
smoke-test:   ; $(RUN) rokkur-studio smoke-test --inject-fault
agent-check: ; $(RUN) rokkur-studio agent-check
comfy-render: ; $(RUN) rokkur-studio smoke-test --renderer comfyui --profile PREVIEW --timeout 3600
# make render ARGS='/media/clip.mp4 --theme "1970s claymation" --rights USER_OWNED --evidence "I filmed it"'
render: ; $(RUN) rokkur-studio render $(ARGS)
audit:        ; $(RUN) rokkur-studio audit
# Tests run on the host against a local Postgres (TEST_DATABASE_URL), see docs/setup-docker.md.
test:         ; pytest
lint:         ; ruff check src tests && mypy
