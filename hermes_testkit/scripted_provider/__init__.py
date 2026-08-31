"""Public deterministic OpenAI-compatible provider for conformance runs.

The stdlib-only server is safe to run in an Omnio Sprite: it never contacts a
real model provider, defaults to loopback, redacts credentials from captures,
and requires an inference API key before a non-loopback bind.  Arm scripts
through the control API or pass ``--script`` to
``python -m hermes_testkit.scripted_provider --help``.  Version 1 supports
text, tool-call, HTTP-error, connection-close, and held responses; text may
optionally provide ordered ``chunks`` whose concatenation is the final text.
Top-level unordered groups support deterministic concurrent request races.
"""

from .schema import (
    DEFAULT_MODEL,
    FIXED_CREATED,
    SCRIPT_SCHEMA_VERSION,
    ResponseStep,
    Script,
    ScriptValidationError,
    ToolCall,
    UnorderedStepGroup,
    matches_request,
    parse_script,
)
from .server import CapturedRequest, ScriptedInferenceServer, ScriptedProviderServer

__all__ = [
    "CapturedRequest",
    "DEFAULT_MODEL",
    "FIXED_CREATED",
    "ResponseStep",
    "SCRIPT_SCHEMA_VERSION",
    "Script",
    "ScriptValidationError",
    "ScriptedInferenceServer",
    "ScriptedProviderServer",
    "ToolCall",
    "UnorderedStepGroup",
    "matches_request",
    "parse_script",
]
