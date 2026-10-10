# REA: Reverse Engineer Anything, as a studio tool

Dashboard: **REA** (`/ui/rea`). API: `/rea` (`GET /rea/status`, `POST /rea`, `GET /rea/{id}`,
`GET /rea/{id}/output`, `DELETE`). CLI: `rokkur-studio rea-run PRESET [target] [--query …]
[--provider …] [--wait]`, `rea-list`. Every run is one real `rea … --json` command executed by
the worker through the job queue (kind `rea`), with its output kept at
`data/rea/<run id>/output.json` (or `.txt` when REA printed no JSON), the exact command in
`command.txt` and REA's progress lines in `stderr.txt`. Rows live in `rea_runs` (migration
`0005`). Nothing is faked: when `rea` is not on the worker's PATH the page says so and how to
install it, and earlier runs stay readable.

REA is Elis's fork `elismagna/rea` of `morluto/rea` (MIT): a CLI and MCP server that inspects
native binaries (through Ghidra, Hopper or IDA), JavaScript and Electron apps and `.asar`
files (statically, no engine needed), .NET assemblies and websites, and returns Evidence with
observations, inferences, limitations and what stays unresolved.

## Presets

| Preset | REA command | Needs |
|---|---|---|
| Analyse a program or app | `analyze PATH` | a file, a macOS `.app`, an app folder or an `.asar`; native files need a provider |
| Inspect an artifact | `inspect-artifact PATH` | a file (REA refuses folders and says to use analyze) |
| Search inside / Function dossier / Decompile / Cross references / Trace / Instructions | `search`, `function`, `decompile`, `xrefs`, `trace`, `instructions` | a file plus the text, name or `0x` address; a provider for native targets |
| Doctor / providers / capabilities | `doctor`, `providers`, `capabilities` | nothing |

`services/rea.py:build_args` checks the request before anything runs (absolute, readable
target; the query a preset needs; address form; a plain provider id), adds `--provider` from
the request or `rea.provider`, and for file targets adds `--snapshot data/rea/snapshots/<name>_<hash>.json`
so REA reuses its cache on the same bytes. `summarize` digests the JSON for the run list
from what REA states (subject, provider, operation, confidence, limitations, statistics,
graph sizes, doctor checks, capability counts); REA's error contract (`error`, `code`,
`message`, `remediation.action`) becomes the run's error text. `doctor` exits 1 when a
required engine is missing: that run shows as failed with its checks, which is the answer.

## Set-up

Where the worker runs: Node.js 22+ and `npm install --global rea-agents`, or set
`rea.command: "npx -y rea-agents@latest"` in `config/studio.yaml` (`STUDIO_REA__COMMAND` in
`.env`). The Docker image has no Node.js, so in Docker the page reports REA unavailable; run
the studio without Docker for REA, or run `rea` on the host and set `rea.command` to a wrapper
the container can reach. Native targets need `rea setup` for Hopper, `GHIDRA_INSTALL_DIR`
for Ghidra, or the IDA adapter (REA's `docs/installation.md`). Only analyse software you may
inspect; launched targets run with the worker's permissions.

## Verified

Checked here against the real `rea-agents` 6.3.0 from npm (Node 22): `--help`, `doctor`,
`capabilities`, `providers`, `analyze` on a small JavaScript app folder (full Evidence
record), and REA's error contract when `inspect-artifact` or `search` get a folder. The tests
(`tests/test_rea.py`) use a fake `rea` that answers in those shapes. Not run on the PC yet.
