"""Unit tests for the in-process EventBus."""

from __future__ import annotations

from dataclasses import dataclass

from aura.event_bus import EventBus


@dataclass
class BaseEvent:
    message: str


@dataclass
class DerivedEvent(BaseEvent):
    details: str | None = None


def test_event_bus_emits_to_registered_handlers() -> None:
    bus = EventBus()
    received: list[str] = []

    def handler(event: DerivedEvent) -> None:
        received.append(event.message)

    bus.subscribe(DerivedEvent, handler)
    bus.emit(DerivedEvent(message="update", details="ok"))

    assert received == ["update"]


def test_event_bus_supports_inheritance_chain() -> None:
    bus = EventBus()
    base_messages: list[str] = []
    derived_messages: list[str] = []

    bus.subscribe(BaseEvent, lambda e: base_messages.append(e.message))
    bus.subscribe(DerivedEvent, lambda e: derived_messages.append(e.message))

    bus.emit(DerivedEvent(message="child", details=None))

    assert base_messages == ["child"]
    assert derived_messages == ["child"]


def test_unsubscribe_removes_handler() -> None:
    bus = EventBus()
    called = False

    def handler(event: BaseEvent) -> None:
        nonlocal called
        called = True

    bus.subscribe(BaseEvent, handler)
    bus.unsubscribe(BaseEvent, handler)
    bus.emit(BaseEvent(message="hi"))

    assert called is False
