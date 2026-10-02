# Omnia continuation context

`POST /v1/runs` accepts `ag_ui_state` on both ordinary runs and continuations.
The validated state is ephemeral user context. It is never saved in SessionDB.
On a typed run it precedes the current user input. On a continuation it follows
the closed history, preserving the earlier cached prefix. It stays at that
boundary throughout the run.

Continuation API version 3 adds optional `continuation.notes`: up to 20 nonempty
strings, each at most 10,000 characters. Clients must inspect
`turn_continuation_api_version` before sending this field to older gateways.
The `/v1/capabilities` response also advertises `run_continuation_notes` and
`run_ag_ui_state`.

Each note is a separate durable user message bounded by the exact marker in
`agent/context_notes.py`. The marker identifies an application fact, grants no
user authority, and explains that replaying history is not a fresh delivery.
Omnia's fixed sandbox doctrine describes the same marker. Notes are never
concatenated into per-run instructions.

Closing results precede notes. A late approval that permits execution stores
the notes with its durable grant; the conversation loop appends them only after
every call in the original block has a result. A retry recovers that grant and
the pending notes. The continuation's Turn ID binds a receipt to the original
close and note list: repeating it is idempotent, while changing it conflicts.

A nested `execute_code` approval has no independent tool result. Its generated
decision fact follows the interrupted parent result and is also returned in
`response.omnio.continuation.notes`, allowing Omnia to project the same fact
into its durable conversation history.

Existing clients may omit notes. Existing sessions need no migration; receipts
use message display metadata. Omnia must release Hermes before reprovisioning
its updated proxy and sandbox doctrine.
