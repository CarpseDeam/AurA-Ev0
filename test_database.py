"""Test script for Aura project organization database."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aura.database import initialize_database, check_database_health
from aura.models import Project, Conversation, Message


def test_database():
    """Test database initialization and basic CRUD operations."""
    print("=" * 70)
    print("Testing Aura Project Organization Database")
    print("=" * 70)

    # Initialize database
    print("\n1. Initializing database...")
    try:
        initialize_database()
        print("   [OK] Database initialized successfully")
    except Exception as e:
        print(f"   [FAIL] Failed to initialize database: {e}")
        return False

    # Check database health
    print("\n2. Checking database health...")
    if check_database_health():
        print("   [OK] Database health check passed")
    else:
        print("   [FAIL] Database health check failed")
        return False

    # Test Project CRUD
    print("\n3. Testing Project CRUD operations...")
    try:
        # Create a project
        project = Project.create(
            name="Test Project",
            description="A test project for Aura",
            working_directory="C:/Projects/AurA-Ev0",
            custom_instructions="Always use type hints and docstrings."
        )
        print(f"   [OK] Created project: {project.name} (ID: {project.id})")

        # Read project
        retrieved = Project.get_by_id(project.id)
        assert retrieved.name == "Test Project"
        print(f"   [OK] Retrieved project: {retrieved.name}")

        # Update project
        project.update(description="Updated description")
        updated = Project.get_by_id(project.id)
        assert updated.description == "Updated description"
        print(f"   [OK] Updated project description")

        # List all projects
        all_projects = Project.get_all()
        print(f"   [OK] Found {len(all_projects)} project(s)")

    except Exception as e:
        print(f"   [FAIL] Project CRUD failed: {e}")
        return False

    # Test Conversation CRUD
    print("\n4. Testing Conversation CRUD operations...")
    try:
        # Create a conversation
        conv = Conversation.create(project_id=project.id)
        print(f"   [OK] Created conversation (ID: {conv.id})")

        # Add messages
        Message.create(conv.id, "user", "Hello, how can I create a new feature?")
        Message.create(conv.id, "assistant", "I can help you create a new feature. What would you like to build?")
        print(f"   [OK] Added 2 messages to conversation")

        # Auto-generate title
        conv.generate_title_from_first_message()
        updated_conv = Conversation.get_by_id(conv.id)
        print(f"   [OK] Generated title: {updated_conv.title}")

        # Get messages
        messages = conv.get_messages()
        print(f"   [OK] Retrieved {len(messages)} messages")

        # Get conversation history
        history = conv.get_history()
        assert len(history) == 2
        assert history[0][0] == "user"
        assert history[1][0] == "assistant"
        print(f"   [OK] Got conversation history: {len(history)} turns")

    except Exception as e:
        print(f"   [FAIL] Conversation CRUD failed: {e}")
        return False

    # Test Recent conversations
    print("\n5. Testing recent conversations...")
    try:
        # Create another conversation
        conv2 = Conversation.create(project_id=project.id, title="Another conversation")
        Message.create(conv2.id, "user", "Test message")
        Message.create(conv2.id, "assistant", "Test response")

        recent = Conversation.get_recent(limit=10)
        print(f"   [OK] Found {len(recent)} recent conversation(s)")

        most_recent = Conversation.get_most_recent()
        print(f"   [OK] Most recent conversation: {most_recent.title}")

    except Exception as e:
        print(f"   [FAIL] Recent conversations test failed: {e}")
        return False

    # Test Project deletion (CASCADE)
    print("\n6. Testing CASCADE deletion...")
    try:
        project_id = project.id
        project.delete()
        print(f"   [OK] Deleted project (ID: {project_id})")

        # Verify conversations were also deleted
        deleted_conv = Conversation.get_by_id(conv.id)
        assert deleted_conv is None
        print(f"   [OK] Conversations were CASCADE deleted")

    except Exception as e:
        print(f"   [FAIL] CASCADE deletion failed: {e}")
        return False

    print("\n" + "=" * 70)
    print("All tests passed! [OK]")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_database()
    sys.exit(0 if success else 1)
