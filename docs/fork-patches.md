# Fork Patch Register

`useomnia/hermes-agent` is a fork of `NousResearch/hermes-agent`. This file is the
register of **patches we carry that upstream does not have**, and the known
interactions between them and upstream work in flight.

Read it before you merge upstream or write a new patch. Update it in the same
change as the patch — an entry added later is an entry nobody wrote.

## When to update

- **New patch that adds or changes a surface upstream also owns** — add an entry.
- **Upstream merge** — walk every entry, update `Upstream status`, delete entries
  whose divergence is gone (upstream adopted it, or we dropped it).
- **A patch you carried gets upstreamed** — don't delete the entry; mark it
  `converged` and say what to check on the next merge. Convergence is where
  silent behavioural drift hides.

Purely local bug fixes with no upstream counterpart do not need an entry. The bar
is: *would someone merging upstream, or writing a patch in this area, be
surprised?*

## Check divergence with the right ref

⚠️ **`upstream/main` is ambiguous in this checkout.** There is a stale *local
branch* named `upstream/main` that shadows the remote-tracking ref. `git log
upstream/main..main` silently reads the local branch and reports nonsense.

Always spell the remote ref out:

```bash
git fetch upstream
U=refs/remotes/upstream/main

git log -1 --format='%h %cs %s' $U      # what upstream actually is
git rev-list --count main..$U           # how far behind we are
git show $U:path/to/file.py | grep …    # does upstream have this yet?
git cat-file -e $U:path/to/file.py      # does the file exist upstream at all?
```

Last verified against `a0a63a1bc2` (2026-08-31); `main` was 8182 commits behind.

## Entry schema

```
### <surface> — <what we added>
- Fork PR:        #<n> (and the files that carry it)
- Upstream status: absent | in flight (#<n>) | converged
- Interaction:    what breaks or drifts on a merge, and what to verify
- Downstream:     who outside this repo depends on the shape
```

---

## Structural hazard — the runs API has been refactored upstream

Not a patch of ours, but it conditions every entry below.

- **Ours:** every `/v1/runs` handler lives inline in
  `gateway/platforms/api_server.py` (~10.9k lines).
- **Upstream:** extracted into `gateway/platforms/api_server_runs.py`, plus
  `api_server_run_idempotency.py`, `api_server_room_dispatch.py`,
  `api_server_room_grants.py`. `api_server.py` now forwards
  (`_handle_steer_run` → `_api_runs._handle_steer_run`).

Every runs-API patch we carry will conflict **structurally**, not just textually,
on the next upstream merge — the hunks have nowhere to land. Budget the merge
accordingly, and prefer writing new runs-API code so it can be relocated in one
move rather than threaded through unrelated handlers.

---

## Entries

### `/v1/runs` — unattended interaction policy

- **Fork PR:** #96 (`fix(api-server): enforce unattended run policy`) —
  `agent/unattended.py` (fork-only file), `interaction_policy` handling in
  `_handle_runs`.
- **Upstream status:** absent. No `interaction_policy` anywhere in upstream's
  `api_server.py`.
- **Interaction:** the request body validation sits in the same block a per-turn
  `budget` field would extend. Two patches editing the same validation ladder.
- **Downstream:** Omnia sends `interaction_policy` on every headless turn; a
  headless run has nobody to answer an approval, so a regression here parks runs
  until their deadline.

### `/v1/runs` — resumable turn event log

- **Fork PR:** #58, #70, #82 — `gateway/turn_event_log.py` (fork-only file),
  `_turn_event_logs` in `api_server.py`, `DEFAULT_RUN_LOG_CAP_BYTES`,
  `failure_reason = "log_cap_exceeded"`.
- **Upstream status:** absent.
- **Interaction:** this is the model to copy for any new per-run resource cap —
  it already does cooperative stop + terminal reason + stream close with a code.
  Also the largest single divergence in the runs API, so it is the most likely
  merge casualty.
- **Downstream:** the Omnia sprite proxy's projector task consumes
  `/v1/runs/{run_id}/events` server-side and depends on the frame contract and
  cursor replay. Breaking the frame shape breaks turn persistence, not just a
  UI stream.

### `/v1/runs/{run_id}/steer` — mid-turn steering

- **Fork PR:** #72 (`feat(api): add mid-turn steering for active runs`).
- **Upstream status:** **converged** — upstream now ships its own
  `POST /v1/runs/{run_id}/steer` and its own `pending_steer` drain in the turn
  finalizer. Independent implementation, same endpoint.
- **Interaction:** ⚠️ **verify the request shape survives the merge.** Ours
  accepts `mode: "redirect" | "steer"` and returns `{"status": "redirected" |
  "queued"}`. Whether upstream's accepts `mode` is **unverified** — check before
  merging, do not assume.
- **Downstream:** the Omnia proxy calls `service.steer(..., mode="steer")`
  (`turn_runtime.py:1578`) and `subagent_wake_routes.py` depends on it for
  delegation wakes. If upstream's variant has no `mode`, a naive merge silently
  turns every append into a redirect — it rewrites the turn's intent instead of
  adding to it, and nothing errors.

### `/v1/runs` — per-request budget field

- **Fork PR:** `feat/runs-turn-budget` — `budget: {max_cost_usd?,
  max_iterations?}` validated in `_handle_runs`, `agent/cost_budget.py` (new),
  the ceiling check in `agent/conversation_loop.py` immediately before an API
  call is counted, `cost_budget_exhausted` explained in
  `_format_turn_completion_explanation`, terminal `failure_reason:
  "budget_exceeded"` modelled on `_close_log_cap_exceeded`, and
  `usage.estimated_cost_usd` on the terminal run.
- **Upstream status:** **three overlapping proposals in flight, none merged.**
  - **#92587** per-model execution budgets — caps non-delegation tool executions
    per turn, keyed by model glob, checked in `tool_executor.py` before dispatch.
    Config-driven, delegation-exempt.
  - **#91892** per-session cumulative *token* budget — adds
    `agent/session_budget.py` and `agent.session_budget_tokens`, guard at the top
    of the conversation loop.
  - **#88514** session token hard stop — `agent.session_token_hard_stop`, warns at
    80%, ends the turn before the next call, and deliberately skips the
    budget-exhausted summary fallback.
- **Interaction:**
  - Name the module `agent/cost_budget.py`, **not** `agent/session_budget.py` —
    that path is #91892's and would collide head-on.
  - All three upstream proposals guard at the top of the conversation loop, the
    same seam as ours. Expect a genuine three-way merge in
    `conversation_loop.py`, not a textual one.
  - Ours is denominated in **estimated USD per turn**; all three upstream ones are
    tokens or tool-call counts, and per *session* or per *config*. The concerns
    compose — do not let a merge collapse them into one knob.
  - #88514's argument for skipping the summary fallback applies to ours verbatim:
    `_handle_max_iterations` spends one more toolless full-context call, on
    exactly the turn the fuse exists to stop. Keep the suppression.
- **Downstream:** Omnia's proxy-side cost fuse is the primary enforcement and
  ships independently of this patch; this is the between-calls backstop. See the
  Omnia-side plan for the split.

### Adjacent: `tools/budget_config.py` is not ours and not about money

Upstream's own module, char budgets for tool-result persistence scaled to the
context window. It shares the word "budget" and nothing else. Don't patch it when
you mean spend.
