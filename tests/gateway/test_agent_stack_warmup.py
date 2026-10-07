"""The post-READY agent-stack warm-up pays first-build costs and never raises."""

from unittest.mock import patch

import gateway.run as gateway_run


def test_warm_up_assembles_the_api_server_tool_definitions():
    calls = []

    with patch.object(gateway_run, "_resolve_runtime_agent_kwargs", return_value={}), patch.object(
        gateway_run, "_load_gateway_config", return_value={}
    ), patch("model_tools.get_tool_definitions", side_effect=lambda **kw: calls.append(kw)):
        gateway_run._warm_agent_stack()

    assert len(calls) == 1
    assert calls[0]["quiet_mode"] is True


def test_warm_up_survives_every_step_failing():
    def boom(*_args, **_kwargs):
        raise RuntimeError("unavailable")

    with patch.object(gateway_run, "_resolve_runtime_agent_kwargs", side_effect=boom), patch(
        "model_tools.get_tool_definitions", side_effect=boom
    ):
        gateway_run._warm_agent_stack()


def test_warm_up_runs_on_a_daemon_thread():
    started = []

    class _Thread:
        def __init__(self, *, target, name, daemon):
            started.append((target, name, daemon))

        def start(self):
            pass

    with patch.object(gateway_run.threading, "Thread", _Thread):
        gateway_run._start_agent_stack_warmup()

    assert started == [(gateway_run._warm_agent_stack, "agent-warmup", True)]
