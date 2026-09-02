"""POST /v1/runs — the per-turn `budget` field, end to end at the HTTP seam.

Three contracts:

- a malformed policy is rejected BEFORE a run is allocated (fail-closed);
- a valid policy reaches ``_create_agent`` as a cost ceiling and an iteration
  override, and an absent one changes nothing;
- a turn that ends on its ceiling terminalizes as ``failed`` /
  ``budget_exceeded`` rather than reporting an empty success.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.cost_budget import COST_BUDGET_EXIT_REASON, CostBudget
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _cost_budget_exit,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


def _mock_agent(*, exit_reason: str = "text_response(done)", cost_usd: float = 0.0):
    agent = MagicMock()
    agent.run_conversation.return_value = {
        "final_response": None if exit_reason == COST_BUDGET_EXIT_REASON else "done",
        "turn_exit_reason": exit_reason,
    }
    agent.session_prompt_tokens = 11
    agent.session_completion_tokens = 22
    agent.session_total_tokens = 33
    agent.session_estimated_cost_usd = cost_usd
    agent._cost_budget_turn_start_usd = 0.0
    return agent


async def _await_terminal(cli, run_id, *, tries: int = 60):
    for _ in range(tries):
        status = await (await cli.get(f"/v1/runs/{run_id}")).json()
        if status["status"] in {"completed", "failed", "cancelled"}:
            return status
        await asyncio.sleep(0.05)
    raise AssertionError(f"run {run_id} never reached a terminal state")


@pytest.fixture
def adapter():
    return _make_adapter()


class TestBudgetValidation:
    """Fail-closed: a policy we cannot read is a 400, never an uncapped run."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "budget",
        [
            "2.50",                              # not an object
            {"maxCostUsd": 2.5},                 # camelCase typo
            {"max_cost_usd": 0},                 # non-positive
            {"max_cost_usd": -1},
            {"max_cost_usd": True},              # bool is not a number
            {"max_iterations": 0},
            {"max_iterations": 1.5},             # not an integer
            {"max_cost_usd": 2.5, "extra": 1},   # one good field, one unknown
        ],
    )
    async def test_rejects_malformed_budget_before_allocating_a_run(
        self, adapter, budget
    ):
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                response = await cli.post(
                    "/v1/runs", json={"input": "hello", "budget": budget}
                )
                data = await response.json()

        assert response.status == 400
        assert data["error"]["code"] == "invalid_budget"
        assert data["error"]["param"] == "budget"
        mock_create.assert_not_called()
        assert adapter._run_statuses == {}
        assert adapter._run_streams == {}

    @pytest.mark.asyncio
    async def test_null_budget_is_valid_and_uncapped(self, adapter):
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _mock_agent()
                response = await cli.post(
                    "/v1/runs", json={"input": "hello", "budget": None}
                )
                assert response.status == 202
                await _await_terminal(cli, (await response.json())["run_id"])

        assert mock_create.call_args.kwargs["cost_budget"] is None


class TestBudgetReachesTheAgent:
    @pytest.mark.asyncio
    async def test_both_fields_are_threaded_through(self, adapter):
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _mock_agent()
                response = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "budget": {"max_cost_usd": 2.5, "max_iterations": 40},
                    },
                )
                assert response.status == 202
                await _await_terminal(cli, (await response.json())["run_id"])

        kwargs = mock_create.call_args.kwargs
        assert kwargs["cost_budget"] == CostBudget(max_cost_usd=2.5)
        assert kwargs["max_iterations_override"] == 40

    @pytest.mark.asyncio
    async def test_absent_budget_changes_nothing(self, adapter):
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _mock_agent()
                response = await cli.post("/v1/runs", json={"input": "hello"})
                assert response.status == 202
                await _await_terminal(cli, (await response.json())["run_id"])

        kwargs = mock_create.call_args.kwargs
        assert kwargs["cost_budget"] is None
        assert kwargs["max_iterations_override"] is None

    @pytest.mark.asyncio
    async def test_cost_only_budget_leaves_iterations_untouched(self, adapter):
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _mock_agent()
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "budget": {"max_cost_usd": 0.25}},
                )
                assert response.status == 202
                await _await_terminal(cli, (await response.json())["run_id"])

        kwargs = mock_create.call_args.kwargs
        assert kwargs["cost_budget"] == CostBudget(max_cost_usd=0.25)
        assert kwargs["max_iterations_override"] is None


class TestTerminalState:
    def test_exit_classifier_reads_the_turn_reason(self):
        assert _cost_budget_exit({"turn_exit_reason": COST_BUDGET_EXIT_REASON})
        assert not _cost_budget_exit({"turn_exit_reason": "text_response(done)"})
        assert not _cost_budget_exit({"turn_exit_reason": "budget_exhausted"})
        assert not _cost_budget_exit({})
        assert not _cost_budget_exit(None)
        assert not _cost_budget_exit("cost_budget_exhausted")

    @pytest.mark.asyncio
    async def test_ceiling_exit_fails_the_run_with_budget_exceeded(self, adapter):
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _mock_agent(
                    exit_reason=COST_BUDGET_EXIT_REASON, cost_usd=2.75
                )
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "budget": {"max_cost_usd": 2.5}},
                )
                assert response.status == 202
                status = await _await_terminal(
                    cli, (await response.json())["run_id"]
                )

        assert status["status"] == "failed"
        assert status["failure_reason"] == "budget_exceeded"
        # The caller must be able to tell a spend stop from a crash.
        assert "cost budget" in status["error"]

    @pytest.mark.asyncio
    async def test_normal_completion_reports_estimated_cost(self, adapter):
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.return_value = _mock_agent(cost_usd=0.1234)
                response = await cli.post("/v1/runs", json={"input": "hello"})
                assert response.status == 202
                status = await _await_terminal(
                    cli, (await response.json())["run_id"]
                )

        assert status["status"] == "completed"
        assert status["usage"]["estimated_cost_usd"] == pytest.approx(0.1234)
        assert status["usage"]["total_tokens"] == 33
