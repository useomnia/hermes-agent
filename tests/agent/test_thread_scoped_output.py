"""Tests for agent.thread_scoped_output.thread_scoped_silence.

Behaviour contract: a thread inside ``thread_scoped_silence()`` has its
stdout/stderr routed to devnull, while every OTHER thread keeps writing to the
real stream — even concurrently, while the first thread is still inside the
context.  This is the property the old process-global
``contextlib.redirect_stdout(devnull)`` violated (issue #55769 / #55925).
"""

import contextlib
import io

import pytest
import sys
import threading
import time

import agent.thread_scoped_output as thread_output
from agent.thread_scoped_output import thread_scoped_silence


def _run_with_real_stream(fn):
    """Bind a StringIO as the real stdout, run fn, return what reached it."""
    real_out = io.StringIO()
    orig = sys.stdout
    sys.stdout = real_out
    try:
        fn()
    finally:
        sys.stdout = orig
    return real_out.getvalue()


def test_current_thread_is_silenced():
    def body():
        with thread_scoped_silence():
            print("dropped")
        print("kept")

    captured = _run_with_real_stream(body)
    assert "dropped" not in captured
    assert "kept" in captured


def test_concurrent_thread_keeps_output_during_silence_window():
    """A loud thread writing WHILE another thread is silenced must survive."""
    inside_silence = threading.Event()
    loud_done = threading.Event()

    def silenced_worker():
        with thread_scoped_silence():
            print("SILENCED")
            inside_silence.set()
            # Hold the silence window until the loud thread has written.
            loud_done.wait(timeout=2.0)

    def loud_worker():
        inside_silence.wait(timeout=2.0)
        print("LOUD")
        loud_done.set()

    def body():
        t1 = threading.Thread(target=silenced_worker)
        t2 = threading.Thread(target=loud_worker)
        t1.start()
        t2.start()
        t1.join(timeout=15.0)
        t2.join(timeout=15.0)
        assert not t1.is_alive() and not t2.is_alive(), "worker threads didn't finish"

    captured = _run_with_real_stream(body)
    assert "SILENCED" not in captured
    assert "LOUD" in captured


def test_stderr_is_also_routed_per_thread():
    real_err = io.StringIO()
    orig = sys.stderr
    sys.stderr = real_err
    try:
        with thread_scoped_silence():
            sys.stderr.write("err-dropped\n")
        sys.stderr.write("err-kept\n")
    finally:
        sys.stderr = orig
    out = real_err.getvalue()
    assert "err-dropped" not in out
    assert "err-kept" in out


def test_nested_silence_same_thread_composes():
    def body():
        with thread_scoped_silence():
            with thread_scoped_silence():
                print("inner")
            # Still inside the OUTER context — depth-counted, so this thread
            # remains silenced after the inner context exits.
            print("after-inner")
        print("after-outer")

    captured = _run_with_real_stream(body)
    assert "inner" not in captured
    assert "after-inner" not in captured
    assert "after-outer" in captured


def test_unsilence_cleans_up_after_exit():
    """After the context exits, the calling thread writes to the real stream."""
    seen = []

    def body():
        with thread_scoped_silence():
            pass
        print("post")
        seen.append("post")

    captured = _run_with_real_stream(body)
    assert "post" in captured
    assert seen == ["post"]


def test_many_concurrent_silenced_and_loud_threads():
    """Stress: interleaved silenced/loud threads keep their respective fates."""
    start = threading.Event()
    results_lock = threading.Lock()

    def silenced(i):
        start.wait(timeout=2.0)
        with thread_scoped_silence():
            print(f"S{i}")
            time.sleep(0.05)

    def loud(i):
        start.wait(timeout=2.0)
        time.sleep(0.02)
        print(f"L{i}")

    def body():
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=silenced, args=(i,)))
            threads.append(threading.Thread(target=loud, args=(i,)))
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(timeout=15.0)
        assert not any(t.is_alive() for t in threads), "straggler thread would truncate captured output"

    captured = _run_with_real_stream(body)
    for i in range(5):
        assert f"S{i}" not in captured, f"silenced S{i} leaked"
        assert f"L{i}" in captured, f"loud L{i} swallowed"


def test_temporary_global_redirects_do_not_allocate_new_sinks(monkeypatch):
    """A displaced proxy is temporary, not a reason to leak another FD pair."""
    opened_sinks = []

    def fake_open(*_args, **_kwargs):
        sink = io.StringIO()
        opened_sinks.append(sink)
        return sink

    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(thread_output, "_sinks", {}, raising=False)
    monkeypatch.setattr(thread_output, "_routing_states", {}, raising=False)
    monkeypatch.setattr(thread_output, "open", fake_open, raising=False)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        with thread_scoped_silence():
            pass
        assert len(opened_sinks) == 2
        original_proxies = dict(thread_output._installed)

        for _ in range(20):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with thread_scoped_silence():
                    print("hidden")

        with thread_scoped_silence():
            pass
        assert len(opened_sinks) == 2
        assert thread_output._installed == original_proxies
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr


def test_silence_survives_redirect_restoring_an_older_proxy(monkeypatch):
    """Silencing is stream-wide, even when a redirect swaps proxy generations."""
    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(thread_output, "_sinks", {}, raising=False)
    monkeypatch.setattr(thread_output, "_routing_states", {}, raising=False)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    passthrough = io.StringIO()
    sys.stdout = passthrough
    entered = threading.Event()
    release = threading.Event()

    try:
        with thread_scoped_silence():
            pass

        def worker():
            with thread_scoped_silence():
                entered.set()
                assert release.wait(timeout=10)
                print("must-stay-silenced")

        redirected = io.StringIO()
        with contextlib.redirect_stdout(redirected):
            thread = threading.Thread(target=worker)
            thread.start()
            assert entered.wait(timeout=10)

        release.set()
        thread.join(timeout=10)

        assert not thread.is_alive()
        assert "must-stay-silenced" not in passthrough.getvalue()
        assert "must-stay-silenced" not in redirected.getvalue()
    finally:
        release.set()
        sys.stdout, sys.stderr = original_stdout, original_stderr


def test_repeated_contexts_keep_the_silenced_descriptor_open():
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        for _ in range(3):
            with thread_scoped_silence():
                assert sys.stdout.fileno() >= 0
                assert sys.stderr.fileno() >= 0
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr


def test_exception_restores_output_for_the_calling_thread(monkeypatch):
    stdout, stderr = io.StringIO(), io.StringIO()
    with monkeypatch.context() as capture:
        capture.setattr(sys, "stdout", stdout)
        capture.setattr(sys, "stderr", stderr)
        with pytest.raises(RuntimeError, match="handler failed"):
            with thread_scoped_silence():
                print("hidden")
                raise RuntimeError("handler failed")
        print("after error")
        print("after error", file=sys.stderr)
    assert stdout.getvalue() == "after error\n"
    assert stderr.getvalue() == "after error\n"
