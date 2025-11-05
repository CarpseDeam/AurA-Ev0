"""Lightweight publish-subscribe event bus for Aura orchestration."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List

LOGGER = logging.getLogger(__name__)
EventHandler = Callable[["Event"], None]


class EventType(Enum):
    """Supported orchestration events."""

    PLANNING_STARTED = auto()
    PLAN_READY = auto()
    SESSION_STARTED = auto()
    SESSION_OUTPUT = auto()
    SESSION_COMPLETE = auto()
    ALL_COMPLETE = auto()
    ERROR = auto()


@dataclass(frozen=True)
class Event:
    """Container passed to subscribers."""

    type: EventType
    data: Dict[str, Any]


class EventBus:
    """Simple in-process publish-subscribe dispatcher."""

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._lock = RLock()

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for the given event type."""
        if handler is None:
            raise ValueError("Handler must be provided.")
        with self._lock:
            self._handlers[event_type].append(handler)

    def publish(self, event_type: EventType, **data: Any) -> None:
        """Publish an event to all subscribers."""
        event = Event(type=event_type, data=dict(data))
        handlers = self._snapshot_handlers(event_type)
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Event handler failed for %s: %s", event_type, exc)

    def _snapshot_handlers(self, event_type: EventType) -> Iterable[EventHandler]:
        with self._lock:
            return list(self._handlers.get(event_type, []))


_EVENT_BUS: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the singleton event bus instance."""
    global _EVENT_BUS
    if _EVENT_BUS is None:
        _EVENT_BUS = EventBus()
    return _EVENT_BUS
