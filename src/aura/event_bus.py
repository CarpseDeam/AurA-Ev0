"""Thread-safe in-process event bus for Aura."""

from __future__ import annotations

import logging
from collections import defaultdict
from threading import RLock
from typing import Callable, DefaultDict, Dict, Iterable, List, Type, TypeVar

LOGGER = logging.getLogger(__name__)
EventT = TypeVar("EventT")
EventHandler = Callable[[EventT], None]


class EventBus:
    """Simple pub-sub dispatcher keyed by event class."""

    def __init__(self) -> None:
        self._handlers: DefaultDict[type, List[EventHandler]] = defaultdict(list)
        self._lock = RLock()

    def subscribe(self, event_type: Type[EventT], handler: EventHandler) -> None:
        """Register a handler for the given event class."""
        if handler is None or event_type is None:
            raise ValueError("Both event_type and handler must be provided.")
        with self._lock:
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: Type[EventT], handler: EventHandler) -> None:
        """Remove a previously registered handler."""
        if handler is None or event_type is None:
            return
        with self._lock:
            listeners = self._handlers.get(event_type)
            if not listeners:
                return
            try:
                listeners.remove(handler)
            except ValueError:
                return
            if not listeners:
                self._handlers.pop(event_type, None)

    def emit(self, event: object) -> None:
        """Publish an event instance to interested subscribers."""
        if event is None:
            return
        handlers = self._snapshot_handlers(type(event))
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Event handler failed for %s", type(event).__name__)

    def _snapshot_handlers(self, event_type: type) -> Iterable[EventHandler]:
        with self._lock:
            seen: set[EventHandler] = set()
            snapshot: list[EventHandler] = []
            for cls in event_type.mro():
                if cls is object:
                    break
                for handler in self._handlers.get(cls, []):
                    if handler not in seen:
                        seen.add(handler)
                        snapshot.append(handler)
            for handler in self._handlers.get(object, []):
                if handler not in seen:
                    snapshot.append(handler)
            return snapshot


_EVENT_BUS: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the process-wide singleton event bus."""
    global _EVENT_BUS
    if _EVENT_BUS is None:
        _EVENT_BUS = EventBus()
    return _EVENT_BUS


__all__ = ["EventBus", "get_event_bus"]
