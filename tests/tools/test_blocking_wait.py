"""Direct contracts for the shared blocking-wait registry."""

import threading

from tools.blocking_wait import BlockingWaitRegistry
from tools.interrupt import set_interrupt

SESSION = "sess-1"


def _start_wait(
    registry: BlockingWaitRegistry[str, str, str],
    call_id: str,
    *,
    timeout_s: float = 5,
) -> tuple[threading.Thread, dict[str, tuple[str | None, str | None]]]:
    parked = threading.Event()
    result: dict[str, tuple[str | None, str | None]] = {}

    def worker() -> None:
        result[call_id] = registry.wait(
            SESSION,
            call_id,
            timeout_s,
            "waiting in registry test",
            payload=f"payload-{call_id}",
            on_parked=lambda _surface: parked.set(),
        )

    thread = threading.Thread(target=worker)
    thread.start()
    assert parked.wait(timeout=3), "the waiter should be parked"
    return thread, result


def test_resolve_uses_call_id_refinement_and_fifo_without_an_id():
    registry: BlockingWaitRegistry[str, str, str] = BlockingWaitRegistry()
    registry.register_surface(SESSION, "surface")
    thread_a, result_a = _start_wait(registry, "call-A")
    thread_b, result_b = _start_wait(registry, "call-B")

    assert registry.resolve(SESSION, "call-missing", "wrong") is False
    assert thread_a.is_alive() and thread_b.is_alive()
    assert registry.resolve(SESSION, "call-B", "second") is True
    thread_b.join(timeout=3)
    assert result_b["call-B"] == ("second", None)
    assert thread_a.is_alive(), "call-id resolution must not release the FIFO head"

    assert registry.resolve(SESSION, "", "first") is True
    thread_a.join(timeout=3)
    assert result_a["call-A"] == ("first", None)


def test_wait_timeout_records_expired_reason():
    registry: BlockingWaitRegistry[str, str, str] = BlockingWaitRegistry()
    registry.register_surface(SESSION, "surface")

    assert registry.wait(SESSION, "call-1", 0, "waiting") == (None, "expired")
    assert registry.consume_completion_reason(SESSION, "call-1") == "expired"
    assert registry.consume_completion_reason(SESSION, "call-1") is None


def test_interrupt_records_cancelled_reason():
    registry: BlockingWaitRegistry[str, str, str] = BlockingWaitRegistry()
    registry.register_surface(SESSION, "surface")
    thread, result = _start_wait(registry, "call-1")

    set_interrupt(True, thread.ident)
    try:
        thread.join(timeout=5)
        assert not thread.is_alive(), "interrupt must release the blocking wait"
        assert result["call-1"] == (None, "cancelled")
        assert registry.consume_completion_reason(SESSION, "call-1") == "cancelled"
    finally:
        set_interrupt(False, thread.ident)


def test_on_release_finishes_before_the_waiter_event_is_signalled():
    registry: BlockingWaitRegistry[str, str, str] = BlockingWaitRegistry()
    registry.register_surface(SESSION, "surface")
    waiter, result = _start_wait(registry, "call-1")
    release_started = threading.Event()
    allow_release = threading.Event()

    def on_release(_entry) -> None:
        release_started.set()
        assert allow_release.wait(timeout=3)

    resolver = threading.Thread(
        target=lambda: registry.resolve(
            SESSION,
            "call-1",
            "approved",
            on_release=on_release,
        )
    )
    resolver.start()
    assert release_started.wait(timeout=3)
    waiter.join(timeout=0.1)
    assert waiter.is_alive(), "the waiter must stay parked until on_release returns"

    allow_release.set()
    resolver.join(timeout=3)
    waiter.join(timeout=3)
    assert not resolver.is_alive()
    assert not waiter.is_alive()
    assert result["call-1"] == ("approved", None)


def test_stale_unregister_is_a_complete_noop():
    registry: BlockingWaitRegistry[str, str, str] = BlockingWaitRegistry()
    stale_token = registry.register_surface(SESSION, "old")
    owner_token = registry.register_surface(SESSION, "current")
    assert registry.wait(SESSION, "call-old", 0, "waiting") == (None, "expired")
    waiter, result = _start_wait(registry, "call-1")

    assert registry.unregister_surface(SESSION, stale_token) is False
    assert registry.has_surface(SESSION) is True
    assert registry.surface_value(SESSION) == "current"
    assert registry.pending_count(SESSION) == 1
    assert waiter.is_alive(), "stale unregister must not release waiters"
    assert registry.consume_completion_reason(SESSION, "call-old") == "expired"

    assert registry.wait(SESSION, "call-old", 0, "waiting") == (None, "expired")
    assert registry.unregister_surface(SESSION, owner_token) is True
    waiter.join(timeout=3)
    assert not waiter.is_alive()
    assert result["call-1"] == (None, "cancelled")
    assert registry.consume_completion_reason(SESSION, "call-old") is None


def test_registered_none_surface_is_distinct_from_no_surface():
    registry: BlockingWaitRegistry[str, None, str] = BlockingWaitRegistry()
    registry.register_surface(SESSION)

    assert registry.has_surface(SESSION) is True
    assert registry.surface_value(SESSION) is None
    assert registry.wait(SESSION, "call-1", 0, "waiting") == (None, "expired")


def test_clear_blindly_drops_the_surface_and_releases_all_waiters():
    registry: BlockingWaitRegistry[str, str, str] = BlockingWaitRegistry()
    registry.register_surface(SESSION, "surface")
    assert registry.wait(SESSION, "call-old", 0, "waiting") == (None, "expired")
    thread_a, result_a = _start_wait(registry, "call-A")
    thread_b, result_b = _start_wait(registry, "call-B")

    registry.clear(SESSION)

    thread_a.join(timeout=3)
    thread_b.join(timeout=3)
    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert result_a["call-A"] == (None, "cancelled")
    assert result_b["call-B"] == (None, "cancelled")
    assert registry.has_surface(SESSION) is False
    assert registry.pending_count(SESSION) == 0
    assert registry.consume_completion_reason(SESSION, "call-old") is None
