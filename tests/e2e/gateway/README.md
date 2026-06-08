# Gateway end-to-end probes

A local, opt-in pytest suite that **builds the `hermes-agent` Docker image,
boots a real gateway container per LLM provider, and runs OpenAI-compatible
probes against it.** Intended to be run by hand after pulling upstream changes —
not in CI.

It grew out of the standalone `probe_structured_output.py` script: every probe
is a plain stdlib HTTP call, just wrapped in pytest so the Docker lifecycle and
the provider matrix are handled for you.

## TL;DR

```bash
# 1. put the provider keys you have into .env.test (repo root, gitignored)
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env.test

# 2. run the probes (builds the image on first run; reuses it after)
HERMES_E2E=1 pytest tests/e2e/gateway -v -s
```

> Always pass **`-s`**. The first run builds the Docker image (minutes) and
> every run polls `/health` after `docker run`; without `-s` pytest captures
> that output and the terminal looks frozen. `-s` streams it live.

## Prerequisites

- **Docker** running and on `PATH` (`docker info` must succeed). On macOS that
  means Docker Desktop is started.
- **At least one provider API key** (see `.env.test`).
- The dev test deps installed (`pytest`), i.e. the same environment
  `scripts/run_tests.sh` uses.

First run builds the `hermes-agent` image, which is slow (Node, Playwright,
Python deps — can take many minutes). Later runs reuse the cached image unless
you pass `HERMES_E2E_REBUILD=1`.

## What it checks

| Module | Probes |
|--------|--------|
| `test_smoke.py` | `/health`, bearer auth is enforced, `/v1/models` advertises the model, a basic non-streaming chat completion |
| `test_structured_output.py` | `response_format: json_schema` (chat **and** `/v1/responses`) is enforced; `response_format: json_object` matches the backend's expected behavior |
| `test_streaming.py` | SSE yields content deltas; `delta.reasoning_content` is well-formed when the backend emits reasoning |
| `test_subagent_progress.py` | a `delegate_task` batch streams its child activity as `event: hermes.tool.progress` (`subagent.*` status + identity fields), well-formed when the model delegates |

## How it works

- **Provider matrix** — `providers.py`. The suite discovers which backends to
  test from the API-key env vars present **when pytest starts** (e.g.
  `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GLM_API_KEY`,
  `KIMI_API_KEY`, `MINIMAX_API_KEY`). One container is booted per available
  provider and every probe runs against each.

  > Keys are read at import time on purpose: the repo's root `conftest.py`
  > blanks all `*_API_KEY` vars during each test, so that's the only window.

- **Ephemeral config** — each container gets a throwaway `HERMES_HOME` tmp dir
  with a generated `config.yaml`. Your real `~/.hermes` is never touched.

- **Lifecycle** — `docker_gateway.py` ensures the image, runs the container with
  the API server enabled (`API_SERVER_*`) on a free localhost port, waits for
  `/health`, and removes the container at the end of the session.

## Configuring keys: `.env.test`

The matrix reads provider keys from the environment. Rather than passing them
on the command line every run, put them in **`.env.test` at the repo root** — it
is gitignored and loaded automatically (by `env_files.py`) before the matrix is
built. Copy the template lines and fill in the keys you have:

```dotenv
# .env.test
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

Variables already set in your shell take precedence over the file (the shell
always wins). A starter `.env.test` with every supported var commented out is
checked out for you — see that file for the full list and optional knobs.

## Running

```bash
# With keys in .env.test, just flip the master switch:
HERMES_E2E=1 pytest tests/e2e/gateway -v -s
```

`-s` is recommended for every run: it disables pytest output capture so the
image build, `docker run`, and `/health` polling stream live instead of looking
hung. On failure the harness also prints the container's `docker logs` tail.

Without `HERMES_E2E=1`, every probe is skipped — so the suite is inert during a
normal `scripts/run_tests.sh` run and in CI. (You can also set `HERMES_E2E=1`
inside `.env.test`; see the note in that file.)

Run a single backend by giving only that key, or select with `-k`:

```bash
HERMES_E2E=1 pytest tests/e2e/gateway -k anthropic -v -s
```

## Typical workflow: after pulling upstream

This suite exists to catch regressions when you sync the fork with upstream
hermes. The drill:

```bash
git pull                      # or merge upstream
HERMES_E2E=1 HERMES_E2E_REBUILD=1 pytest tests/e2e/gateway -v -s
```

`HERMES_E2E_REBUILD=1` forces a fresh `docker build` so you're probing the new
code, not a stale image. A green run means the gateway still speaks the
OpenAI-compatible contract (auth, models, json_schema, json_object, streaming +
`reasoning_content`, and `hermes.tool.progress` subagent events) across every
backend you have a key for.

## Knobs (env vars)

| Var | Default | Purpose |
|-----|---------|---------|
| `HERMES_E2E` | _unset_ | Master switch — must be `1`/`true`/`yes` to run anything |
| `HERMES_E2E_REBUILD` | `0` | `1` forces `docker build` even if the image exists — **set this after pulling upstream** |
| `HERMES_E2E_IMAGE` | `hermes-agent` | Image tag to build/run |
| `HERMES_E2E_MODEL_<PROVIDER>` | per-provider default | Override the model for one backend, e.g. `HERMES_E2E_MODEL_ANTHROPIC=claude-sonnet-4-6` |
| `HERMES_E2E_MODEL_NAME` | `hermes-agent` | Model name the API server advertises / probes request |
| `HERMES_E2E_BUILD_TIMEOUT` | `1800` | Seconds for the image build |
| `HERMES_E2E_READY_TIMEOUT` | `240` | Seconds to wait for `/health` after `docker run` |

## Adding a probe

Drop a `test_*.py` next to the others. Take the `gateway` fixture (a
`GatewayClient` already pointed at a running container) and assert against it:

```python
import pytest
from .constants import MODEL

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(0)]

def test_my_probe(gateway):
    resp = gateway.post("/v1/chat/completions", {"model": MODEL, "messages": [...]})
    assert resp.status == 200
    # backend-specific expectations: gateway.provider.spec.<field>
```

To branch on backend behavior, read `gateway.provider` (a `ResolvedProvider`).
Add new backends or per-backend expectations in `providers.py`.

## Troubleshooting

- **Everything skips with "opt-in" reason** — `HERMES_E2E` isn't set. Export it
  or uncomment it in `.env.test`.
- **`no-provider` skip** — no provider key was found at startup. Check `.env.test`
  (uncommented, correct var name) and remember the shell overrides the file.
- **`docker CLI/daemon not available`** — start Docker; verify `docker info`.
- **Container "exited before becoming ready" / `/health` timeout** — the harness
  prints the last 80 lines of `docker logs` for the failed container. Common
  causes: a bad/missing key for that provider, or an unknown model id — override
  it with `HERMES_E2E_MODEL_<PROVIDER>` (model defaults drift as providers
  rename models). Bump `HERMES_E2E_READY_TIMEOUT` if first boot is just slow.
- **`json_object` test fails on a new backend** — the expected behavior lives in
  `providers.py` (`json_object`: `accept` / `reject` / `any`). Adjust the spec.
- **Stale image after upstream pull** — add `HERMES_E2E_REBUILD=1`.
- **Leftover containers** — they're auto-removed at session end; if a run was
  killed, clean up with `docker ps -a --filter name=hermes-e2e- -q | xargs docker rm -f`.
