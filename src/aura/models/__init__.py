"""
Models package for Aura's project organization system.

This package provides data models and CRUD operations for:
- Projects: Project metadata, settings, and custom instructions
- Conversations: Conversation threads within projects
- Messages: Individual messages in conversations
"""

from .project import Project
from .conversation import Conversation
from .message import Message, MessageRole

__all__ = ['Project', 'Conversation', 'Message', 'MessageRole']
