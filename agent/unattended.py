"""Shared policy for agent runs that have no interactive user surface."""

INTERACTION_POLICY_ALLOW = "allow"
INTERACTION_POLICY_FORBID = "forbid"
INTERACTION_POLICIES = frozenset({INTERACTION_POLICY_ALLOW, INTERACTION_POLICY_FORBID})

UNATTENDED_RUN_GUIDANCE = (
    "There is no user present for this run. Do not ask questions, request "
    "clarification, seek approval, or wait for follow-up. Make reasonable safe "
    "decisions and complete the task end to end before returning your final "
    "response. If an action cannot proceed safely without user input or "
    "approval, skip that action, continue everything else that can be completed, "
    "and report the limitation in the final response."
)

# These native toolsets are inherently interactive. Product plugins can add
# their own interaction-only toolsets at an integration boundary.
UNATTENDED_DISABLED_TOOLSETS = ("clarify",)


def with_unattended_guidance(instructions: object) -> str:
    """Append the unattended contract without discarding caller context."""
    caller_instructions = instructions if isinstance(instructions, str) else ""
    if not caller_instructions.strip():
        return UNATTENDED_RUN_GUIDANCE
    return f"{caller_instructions.rstrip()}\n\n{UNATTENDED_RUN_GUIDANCE}"
