# AGENTS.md

Instructions for AI coding assistants (Codex, Claude) working in this repo.

**Read `docs/AI_HANDOFF.md` first.** It holds the shared project state: what is built and
verified, key decisions, known issues, what each assistant is working on, and how we share
the repo. Inspect the current files before suggesting changes, and update the handoff's
"Current work" and "Log" sections after meaningful work.

- Pull before you start, and commit and push when a piece of work is done. Claude only sees
  what is on GitHub (`main`).
- Never commit `.env`, `secrets/`, `data/`, tokens or credentials, and never write them into
  the handoff.
- Follow the project rules listed in the handoff (rights gate, private-by-default publishing,
  no YouTube downloading, localhost-only ports, no faked integrations).
- Before pushing, run `ruff check src tests`, `mypy src` and `pytest`. `TEST_DATABASE_URL` must
  point at a separate test database, because the tests drop every table in it.
