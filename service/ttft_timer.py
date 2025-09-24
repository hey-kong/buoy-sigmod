import time
from contextlib import contextmanager


_elapsed = 0.0
_start_at = None
_running = False


def start():
    global _start_at, _running
    if not _running:
        _start_at = time.perf_counter()
        _running = True


def stop():
    global _elapsed, _start_at, _running
    if _running and _start_at is not None:
        _elapsed += time.perf_counter() - _start_at
        _start_at = None
        _running = False


def resume():
    start()


def reset():
    global _elapsed, _start_at
    _elapsed = 0.0
    _start_at = time.perf_counter() if _running else None


def elapsed() -> float:
    if _running and _start_at is not None:
        return _elapsed + (time.perf_counter() - _start_at)
    return _elapsed


@contextmanager
def timer(reset_before_start: bool = True):
    if reset_before_start:
        reset()
    start()
    try:
        yield
    finally:
        stop()


@contextmanager
def without_timing():
    was_running = _running
    if was_running:
        stop()
    try:
        yield
    finally:
        if was_running:
            start()
