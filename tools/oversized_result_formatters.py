"""Optional per-tool replacement formatters for results that got persisted.

`tools/tool_result_storage.py` protects the context window by writing an
oversized tool result to the sandbox and putting a generic
``<persisted-output>`` preview in its place. That preview is the right answer
for a search or a file read -- the model is told where the rest lives and can
go and get it.

It is the wrong answer for a result whose whole point is that the model read
all of it. A `skill_view` preview still opens with ``"success": true`` and a
fragment of the skill body, so the model believes a mandatory skill is loaded
when most of it was dropped.

This registry lets exactly those tools say what their receipt should look like
when their content did NOT reach the model:

    register_formatter("skill_view", _skill_view_incomplete_result)

The contract is deliberately narrow:

* A formatter is consulted ONLY after the storage layer has already decided,
  on its own, to persist a result. It cannot change a threshold, a persistence
  decision, the aggregate candidate order, or what is written to disk -- the
  full result is on disk unchanged either way. It only changes the rendered
  receipt.
* ``formatter(content, *, tool_name) -> str | None``. Returning ``None`` (or
  anything that is not a non-empty string) means "no opinion": the caller
  emits its normal generic block.
* A formatter that raises is treated as ``None`` and logged. A broken
  formatter degrades to today's behaviour; it never fails a tool call.
* The registry is EMPTY by default. With nothing registered, every lookup is
  one ``dict.get`` returning ``None`` and the storage layer is byte-identical
  to what it was before this module existed.
"""

import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

# fn(content, *, tool_name) -> replacement receipt, or None to fall through.
OversizedResultFormatter = Callable[..., Optional[str]]

_FORMATTERS: Dict[str, OversizedResultFormatter] = {}


def register_formatter(tool_name: str, formatter: OversizedResultFormatter) -> None:
    """Register *formatter* as the persisted-result receipt for *tool_name*."""
    if not tool_name or not callable(formatter):
        raise ValueError("register_formatter needs a tool name and a callable")
    _FORMATTERS[str(tool_name)] = formatter


def unregister_formatter(tool_name: str) -> None:
    """Remove any formatter for *tool_name* (no-op when none is registered)."""
    _FORMATTERS.pop(str(tool_name), None)


def has_formatter(tool_name: str) -> bool:
    """True when *tool_name* has a replacement formatter registered."""
    return bool(tool_name) and str(tool_name) in _FORMATTERS


def format_oversized_result(content: str, tool_name: str) -> Optional[str]:
    """Return a tool-specific receipt for persisted *content*, or None.

    None means the caller keeps its generic ``<persisted-output>`` block. That
    is the answer for every tool without a formatter, for a formatter that
    declines, and for a formatter that raises.
    """
    if not tool_name:
        return None
    formatter = _FORMATTERS.get(str(tool_name))
    if formatter is None:
        return None
    try:
        replacement = formatter(content, tool_name=str(tool_name))
    except Exception as exc:
        logger.warning(
            "Oversized-result formatter for %s failed, using generic receipt: %s",
            tool_name, exc,
        )
        return None
    if isinstance(replacement, str) and replacement:
        return replacement
    return None
