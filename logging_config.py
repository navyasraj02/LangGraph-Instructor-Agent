"""Simple logging setup."""
import logging


def _quiet_third_party_loggers(debug: bool) -> None:
    """Hide framework noise (HTTP traces, AutoGen runtime event JSON, etc.)."""
    if debug:
        return
    for name in (
        "autogen",
        "autogen_agentchat",
        "openai",
        "httpx",
        "httpcore",
        "urllib3",
        "grpc",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    # ERROR spam on CancelledError / queue teardown during Ctrl+C (not useful in the terminal).
    logging.getLogger("autogen_core").setLevel(logging.CRITICAL)
    logging.getLogger("autogen_core.events").setLevel(logging.CRITICAL)
    # "Task exception was never retrieved" during interrupt teardown.
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)


def setup_logging(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    fmt = "[%(asctime)s] %(message)s"
    try:
        logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", force=True)
    except TypeError:
        logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    _quiet_third_party_loggers(debug)
    return logging.getLogger("moodle_agent")


def failure_tag(exc: Exception) -> str:
    """Compact failure tag for run summary logs."""
    msg = str(exc)
    if "rate_limit_exceeded" in msg or "RateLimitError" in msg:
        return "rate_limit_exceeded"
    return type(exc).__name__


logger = setup_logging()
