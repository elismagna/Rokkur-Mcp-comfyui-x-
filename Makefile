# Docker-first commands. On Windows without make, use scripts/studio.ps1 <command>.
COMPOSE ?= docker compose
RUN = $(COMPOSE) run --rm api

.PHONY: up down logs test migrate studio worker comfy-check youtube-auth smoke-test comfy-render audit lint build

build:        ; $(COMPOSE) build migrate
up:           ; $(COMPOSE) build migrate && $(COMPOSE) up -d
down:         ; $(COMPOSE) down
logs:         ; $(COMPOSE) logs -f --tail=200
migrate:      ; $(RUN) rokkur-studio migrate
studio:       ; $(COMPOSE) up -d api
worker:       ; $(COMPOSE) up -d worker
comfy-check:  ; $(RUN) rokkur-studio comfy-check
youtube-auth: ; $(RUN) rokkur-studio youtube-auth
smoke-test:   ; $(RUN) rokkur-studio smoke-test --inject-fault
comfy-render: ; $(RUN) rokkur-studio smoke-test --renderer comfyui --profile PREVIEW --timeout 3600
audit:        ; $(RUN) rokkur-studio audit
# Tests run on the host against a local Postgres (TEST_DATABASE_URL), see docs/setup-docker.md.
test:         ; pytest
lint:         ; ruff check src tests && mypy
