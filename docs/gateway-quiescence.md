# Omnio gateway quiescence contract

Hermes exposes an authenticated handover probe at `/v1/omnio/quiescence`.
The caller sends the gateway API key as `Authorization: Bearer ...`.

`POST /v1/omnio/quiescence` accepts:

```json
{"operation":"prepare","mode":"graceful","request_id":"handover-42"}
```

`mode` defaults to `graceful`; `GET /v1/omnio/quiescence` is the equivalent
status operation. Explicit `/prepare`, `/status`, and `/release` paths are
available as additive aliases. A successful response has
`object="hermes.gateway.quiescence"`, `state="quiescent"`, `known=true`, and
`total=0`. Responses also include the process `boot_id`, a monotonic
`generation`, the canonical aggregate `total`, and per-class `counts`; Omnia
should retain these proof-identity fields with its handover barrier. A busy
response is HTTP 409; an unavailable/unknown count is HTTP 503. `request_id`
is an opaque caller/barrier identity. For a force prepare, a caller-supplied
ID is persisted and the matching release must carry that exact ID in addition
to the `generation` and `boot_id` proof; a release with a stale ID is rejected.
If force prepare omits it, Hermes returns a generated ID and retains
generation+boot compatibility for older callers. Repeating a prepare observes
current state without creating another gate or writer.
Graceful status/prepare polling does not advance `generation`; it changes only
when a force epoch is created or successfully released.

The response shape is:

```json
{
  "object": "hermes.gateway.quiescence",
  "operation": "prepare",
  "state": "quiescent",
  "mode": "graceful",
  "latched": false,
  "boot_id": "<opaque-process-boot-id>",
  "generation": 3,
  "known": true,
  "counts": {
    "api_runs": 0,
    "gateway_agents": 0,
    "background_agent_tasks": 0,
    "cron_jobs": 0,
    "processes": 0,
    "process_watchers": 0,
    "completion_queue": 0
  },
  "total": 0,
  "observed_at": 0
}
```

`errors` is present only when a subsystem cannot be proved. Omnia must treat
`known=false`, any non-zero count, or any non-200 response as non-quiescent.

The aggregate covers API reservations and runs, gateway session agents,
background agent tasks and durable async delegations, in-flight cron jobs,
running terminal processes, active or queued process watchers, and the
completion queue. Async delegations use one SQLite transaction and one
predicate: rows remain counted while `state` is `running`, `stalling`, or
`finalizing`, or while delivery is `pending`/`claimed` (including a non-null
legacy delivery claim). Dispatch registration and its durable row share the
same lock, so finalization cannot pass through a false zero between the
running and pending-delivery states.

Graceful prepare is a snapshot, not a Hermes admission latch. Omnia must first
persist its own `admission_state=quiescing` transaction, which rejects new
external user/app/cron turns while allowing continuation and wake delivery;
then it asks Hermes for the zero snapshot and takes its SQL turn fence. This
keeps a pending durable wake deliverable instead of stranding it behind the
handover gate. The proxy must retain that admission fence through Git flush,
promotion, and teardown: a successful Hermes snapshot is not itself a lasting
gateway-wide admission barrier. STARTING/READY profiles and admitted request
reservations must be included in the proxy's atomic fence, and cold or
unreachable profiles must be unknown rather than zero.

`mode="force"` first blocks new external API, cron, and messaging turns,
interrupts active agents/cron jobs/async delegations, and kills registered
terminal processes. Internal completion wakes are allowed to drain while the
proof is in progress. Hermes returns `quiescent` only after a fresh known zero
snapshot; timeout or interruption failure returns busy/error and leaves the
force gate latched. `POST ...` with `operation="release"` (or the release
alias) must include the exact `generation` and `boot_id` returned by force
prepare. When prepare included `request_id`, release must also include that
exact ID; stale or missing proof identity is rejected with 409 and leaves the
gate latched. A matching release is idempotent and reopens the gate after the
caller has completed its handover. Hermes writes and reads back the force
marker before reporting force success; marker write/readback failure returns
busy/error and leaves the in-memory gate latched. Clearing the marker is also
verified before release reports success. The force-retired identity is
persisted in the profile marker across a clean restart, and the new adapter
starts latched; the old proof identity is still required for release.

Cold-state rule: an unreachable or starting gateway is **unknown**, never
quiescent. Gateway startup invalidates the previous ordinary marker, while
preserving a force-retired identity until its matching release. Clean teardown
persists `gateway_quiescence.json` only as the last observed state; a
`quiescent` marker is valid only when that final snapshot was known zero. Each
marker carries a random boot identity and startup/shutdown writes are
serialized; publication fsyncs the containing directory, and a late shutdown
from an older process cannot overwrite a newer process's `starting` marker.
Missing markers are allowed on first boot, but malformed or unreadable
existing markers (including malformed force identities) are never repaired as
if fresh. If startup cannot durably write and read back the `starting` marker,
Hermes aborts startup rather than admitting work.
Durable async rows are independently recoverable after a crash (running rows
become unknown/pending on the next boot). Supervisors that cannot reach the
listener may use `gateway.quiescence.collect_offline_durable_snapshot()`, but
must wait for a live probe when it reports `unknown` because process watcher
and completion-queue state is intentionally not durable.
