"""The long-running gateway avoids constructing unrelated CLI commands."""

from unittest.mock import Mock

from hermes_cli import main
from hermes_cli import config


def test_gateway_run_uses_canonical_parser_and_dispatch(monkeypatch):
    dispatch = Mock(return_value=None)
    prepare = Mock()
    monkeypatch.setattr(main, "cmd_gateway", dispatch)
    monkeypatch.setattr(main, "_prepare_agent_startup", prepare)
    monkeypatch.setattr(config, "get_container_exec_info", lambda: None)

    used = main._try_fast_gateway_run([
        "gateway",
        "run",
        "--replace",
        "--external-supervisor",
        "-vv",
    ])

    assert used is True
    args = dispatch.call_args.args[0]
    assert args.command == "gateway"
    assert args.gateway_command == "run"
    assert args.replace is True
    assert args.external_supervisor is True
    assert args.verbose == 2
    prepare.assert_called_once_with(args)


def test_non_run_gateway_command_keeps_full_parser_path(monkeypatch):
    monkeypatch.setattr(config, "get_container_exec_info", lambda: None)

    assert main._try_fast_gateway_run(["gateway", "status"]) is False


def test_managed_container_keeps_container_routing(monkeypatch):
    monkeypatch.setattr(
        config, "get_container_exec_info", lambda: {"runtime": "docker"}
    )

    assert main._try_fast_gateway_run(["gateway", "run", "--replace"]) is False
