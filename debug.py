"""Global debug-print toggle, controlled by cli.py's --debug flag."""

_enabled = False


def set_debug(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def is_debug() -> bool:
    return _enabled


def debug_print(*args, **kwargs) -> None:
    if _enabled:
        print(*args, **kwargs)
