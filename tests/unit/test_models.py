"""Unit tests for database-backed models."""

from __future__ import annotations

from aura.models.conversation import Conversation
from aura.models.message import Message, MessageRole
from aura.models.project import Project


def test_project_crud_roundtrip(isolated_db) -> None:  # noqa: PT019 - fixture enforces isolation
    project = Project.create(
        name="Test Project",
        description="Initial description",
        working_directory="/tmp/project",
        custom_instructions="Keep responses short.",
        settings={"verbosity": "high"},
    )
    assert project.id is not None
    assert project.name == "Test Project"
    assert project.settings == {"verbosity": "high"}

    project.update(name="Renamed Project", description="Updated")
    refreshed = Project.get_by_id(project.id)
    assert refreshed is not None
    assert refreshed.name == "Renamed Project"
    assert refreshed.description == "Updated"

    project.delete()
    assert Project.get_by_id(project.id) is None


def test_conversation_tracks_messages_and_history(isolated_db) -> None:  # noqa: PT019
    project = Project.create(name="Workspace")
    conversation = Conversation.create(project_id=project.id)
    assert conversation.id is not None

    Message.create(conversation.id, MessageRole.USER, "First message from user.")
    Message.create(conversation.id, MessageRole.ASSISTANT, "Assistant response.")

    messages = conversation.get_messages()
    assert [msg.role for msg in messages] == [MessageRole.USER, MessageRole.ASSISTANT]

    history = conversation.get_history()
    assert history[0] == (MessageRole.USER, "First message from user.")

    conversation.generate_title_from_first_message()
    titled = Conversation.get_by_id(conversation.id)
    assert titled is not None
    assert titled.title.startswith("First message")


def test_message_delete_removes_record(isolated_db) -> None:  # noqa: PT019
    conversation = Conversation.create()
    message = Message.create(conversation.id, MessageRole.USER, "Hello")
    assert message.id is not None

    message.delete()
    assert Message.get_by_id(message.id) is None
