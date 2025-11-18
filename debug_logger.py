from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Iterator

try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init(autoreset=True)
except Exception:  # pragma: no cover - color optional
    class Dummy:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
        RESET_ALL = ""

    Fore = Style = Dummy()  # type: ignore


LOG_DIR = Path(os.getenv("LOG_DIR", "logs")).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRACE_FILE = LOG_DIR / "execution_trace.log"

BOX_TOP = "┌" + "─" * 78 + "┐"
BOX_BOTTOM = "└" + "─" * 78 + "┘"
BOX_MID = "├" + "─" * 78 + "┤"
LINE_PREFIX = "│ "


@dataclass(slots=True)
class DebugSettings:
    debug_enabled: bool
    console_level: int
    file_level: int


class BoxFormatter(logging.Formatter):
    def __init__(self, use_color: bool = True) -> None:
        super().__init__("%(message)s")
        self.use_color = use_color

    def colorize(self, level: int, text: str) -> str:
        if not self.use_color:
            return text
        color = {
            logging.DEBUG: Fore.CYAN,
            logging.INFO: Fore.GREEN,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR: Fore.RED,
            logging.CRITICAL: Fore.MAGENTA,
        }.get(level, "")
        return f"{color}{text}{Style.RESET_ALL}"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        lines = message.strip().splitlines() or [""]
        boxed_lines = [BOX_TOP]
        for idx, line in enumerate(lines):
            if idx == 0 and getattr(record, "box_title", None):
                title = f" {record.box_title} "  # type: ignore[attr-defined]
                boxed_lines.append(self.colorize(record.levelno, LINE_PREFIX + title.ljust(77) + "│"))
                boxed_lines.append(BOX_MID)
                continue
            boxed_lines.append(LINE_PREFIX + line.ljust(77) + "│")
        boxed_lines.append(BOX_BOTTOM)
        return "\n".join(boxed_lines)


def _attach_handler(logger: Logger, handler: logging.Handler) -> None:
    logger.addHandler(handler)
    logger.propagate = False


def configure_debug_logging(settings: DebugSettings) -> Logger:
    logger = logging.getLogger("rag.debug")
    if getattr(logger, "_configured", False):  # type: ignore[attr-defined]
        logger.setLevel(settings.console_level)
        return logger

    logger.setLevel(settings.console_level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.console_level)
    console_handler.setFormatter(BoxFormatter(use_color=True))
    _attach_handler(logger, console_handler)

    file_handler = logging.FileHandler(TRACE_FILE, encoding="utf-8")
    file_handler.setLevel(settings.file_level)
    file_handler.setFormatter(BoxFormatter(use_color=False))
    _attach_handler(logger, file_handler)

    logger._configured = True  # type: ignore[attr-defined]
    return logger


@contextmanager
def log_step(logger: Logger, title: str, level: int = logging.INFO) -> Iterator[None]:
    start = time.perf_counter()
    record = logging.LogRecord(
        name=logger.name,
        level=level,
        pathname="",
        lineno=0,
        msg=f"{Fore.YELLOW if level == logging.INFO else ''}▶ {title}",
        args=(),
        exc_info=None,
    )
    record.box_title = title  # type: ignore[attr-defined]
    logger.handle(record)
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Step '{title}' completed in {elapsed_ms:.2f} ms")


def log_key_value(logger: Logger, heading: str, data: dict[str, Any], level: int = logging.INFO) -> None:
    lines = [f"{heading}:"]
    for key, value in data.items():
        lines.append(f"  • {key}: {value}")
    logger.log(level, "\n".join(lines))


def log_list(logger: Logger, heading: str, items: list[Any], level: int = logging.INFO) -> None:
    lines = [f"{heading}:"]
    for item in items:
        lines.append(f"  • {item}")
    logger.log(level, "\n".join(lines))


def log_table(logger: Logger, heading: str, rows: list[tuple[str, str]], level: int = logging.INFO) -> None:
    max_key = max((len(row[0]) for row in rows), default=0)
    lines = [heading]
    for key, value in rows:
        lines.append(f"  {key.ljust(max_key)} : {value}")
    logger.log(level, "\n".join(lines))


def format_similarity_scores(scores: list[tuple[str, float]]) -> list[str]:
    formatted = []
    for chunk_id, score in scores:
        if score >= 0.7:
            color = Fore.GREEN
        elif score >= 0.4:
            color = Fore.YELLOW
        else:
            color = Fore.RED
        formatted.append(f"{color}{chunk_id}: {score:.4f}{Style.RESET_ALL}")
    return formatted


def log_error(logger: Logger, message: str, exc: Exception | None = None) -> None:
    if exc:
        logger.error(f"{message}\nException: {exc}", exc_info=True)
    else:
        logger.error(message)
