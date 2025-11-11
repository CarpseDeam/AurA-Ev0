"""Unit tests for ToolManager's filesystem and code-analysis helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.exceptions import FileVerificationError
from aura.tools.tool_manager import ToolManager


def test_create_and_read_file(tool_manager: ToolManager, workspace_dir: Path) -> None:
    result = tool_manager.create_file("notes.txt", "hello world")
    assert "Successfully created" in result
    assert (workspace_dir / "notes.txt").read_text(encoding="utf-8") == "hello world"

    contents = tool_manager.read_project_file("notes.txt")
    assert contents == "hello world"


def test_modify_file_replaces_content(tool_manager: ToolManager, workspace_dir: Path) -> None:
    target = workspace_dir / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")

    message = tool_manager.modify_file("app.py", "value = 1", "value = 99")
    assert "Successfully modified" in message
    assert target.read_text(encoding="utf-8") == "value = 99\n"


def test_modify_file_raises_when_old_content_missing(
    tool_manager: ToolManager,
    workspace_dir: Path,
) -> None:
    target = workspace_dir / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(FileVerificationError):
        tool_manager.modify_file("app.py", "missing", "value = 2")


def test_delete_file_removes_target(tool_manager: ToolManager, workspace_dir: Path) -> None:
    target = workspace_dir / "temp.txt"
    target.write_text("temp", encoding="utf-8")
    assert target.exists()

    response = tool_manager.delete_file("temp.txt")
    assert "Successfully deleted" in response
    assert not target.exists()


def test_list_project_files_includes_gitignored_assets_by_default(workspace_dir: Path) -> None:
    (workspace_dir / ".gitignore").write_text("/Assets/**\n", encoding="utf-8")
    assets_dir = workspace_dir / "Assets"
    assets_dir.mkdir()
    asset_file = assets_dir / "tree.png"
    asset_file.write_text("png", encoding="utf-8")

    manager = ToolManager(str(workspace_dir))
    result = manager.list_project_files(".", extension=".png")

    assert result["count"] == 1
    assert result["files"] == ["Assets/tree.png"]


def test_list_project_files_can_respect_gitignore(workspace_dir: Path) -> None:
    (workspace_dir / ".gitignore").write_text("/Assets/**\n", encoding="utf-8")
    assets_dir = workspace_dir / "Assets"
    assets_dir.mkdir()
    (assets_dir / "tree.png").write_text("png", encoding="utf-8")

    manager = ToolManager(str(workspace_dir))
    result = manager.list_project_files(".", extension=".png", respect_gitignore=True)

    assert result["count"] == 0
    assert result["files"] == []
    assert "error" not in result


def test_list_project_files_filters_special_directories(
    tool_manager: ToolManager,
    workspace_dir: Path,
) -> None:
    git_dir = workspace_dir / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("version = 0\n", encoding="utf-8")
    keep_file = workspace_dir / "main.py"
    keep_file.write_text("print('hello')\n", encoding="utf-8")

    result = tool_manager.list_project_files(".", extension="")

    assert "main.py" in result["files"]
    assert not any(path.startswith(".git/") for path in result["files"])


def test_get_imports_categorizes_modules(tool_manager: ToolManager, workspace_dir: Path) -> None:
    script = workspace_dir / "imports_example.py"
    script.write_text(
        "import os\n"
        "import requests\n"
        "from pathlib import Path\n"
        "from aura import config\n"
        "from .local_module import helper\n",
        encoding="utf-8",
    )

    data = tool_manager.get_imports("imports_example.py")
    assert sorted(data["stdlib"]) == ["os", "pathlib"]
    assert "requests" in data["third_party"]


def test_find_definition_returns_symbol_location(tool_manager: ToolManager, workspace_dir: Path) -> None:
    module = workspace_dir / "sample.py"
    module.write_text(
        "def helper(value: int) -> int:\n"
        "    \"\"\"Return value doubled.\"\"\"\n"
        "    return value * 2\n",
        encoding="utf-8",
    )

    result = tool_manager.find_definition("helper")
    assert result["found"] is True
    assert result["type"] == "function"
    assert str(module) in result["file"]
    assert "Return value doubled" in result["docstring"]


def test_verify_asset_paths_reports_existing_and_missing(tool_manager: ToolManager, workspace_dir: Path) -> None:
    assets_dir = workspace_dir / "Assets"
    assets_dir.mkdir()
    mesh = assets_dir / "Tree.fbx"
    mesh.write_bytes(b"fbx-data")

    outcome = tool_manager.verify_asset_paths(["Assets/Tree.fbx", "Assets/Rock.obj"])
    assert outcome["requested"] == 2
    assert outcome["paths"]["Assets/Tree.fbx"] is True
    assert outcome["paths"]["Assets/Rock.obj"] is False
    assert "Assets/Tree.fbx" in outcome["existing"]
    assert "Assets/Rock.obj" in outcome["missing"]


def test_list_project_assets_groups_entries_by_type(tool_manager: ToolManager, workspace_dir: Path) -> None:
    project_root = workspace_dir / "GameProject" / "Assets"
    (project_root / "Meshes").mkdir(parents=True)
    (project_root / "Textures").mkdir()
    (project_root / "Audio").mkdir()

    (project_root / "Meshes" / "Tree.FBX").write_bytes(b"mesh")
    (project_root / "Textures" / "Tree.png").write_bytes(b"png")
    (project_root / "Audio" / "wind.wav").write_bytes(b"wav")

    data = tool_manager.list_project_assets("GameProject", "Assets")

    assert data["total_assets"] == 3
    assert data["counts"]["meshes"] == 1
    assert data["counts"]["textures"] == 1
    assert data["counts"]["sounds"] == 1

    mesh_entry = data["assets"]["meshes"][0]
    assert mesh_entry["path"].endswith("Tree.FBX")
    assert mesh_entry["extension"] == ".fbx"


def test_search_assets_by_pattern_supports_type_filter(tool_manager: ToolManager, workspace_dir: Path) -> None:
    assets_dir = workspace_dir / "Content"
    (assets_dir / "Trees").mkdir(parents=True)
    (assets_dir / "Audio").mkdir()

    (assets_dir / "Trees" / "PineTree.fbx").write_bytes(b"mesh")
    (assets_dir / "Audio" / "forest.mp3").write_bytes(b"sound")

    matches = tool_manager.search_assets_by_pattern("*Tree*.fbx", file_type="mesh", directory="Content")

    assert matches["count"] == 1
    assert matches["resolved_type"] == "meshes"
    assert matches["matches"][0]["path"].endswith("PineTree.fbx")


def test_get_asset_metadata_returns_details(tool_manager: ToolManager, workspace_dir: Path) -> None:
    asset = workspace_dir / "Assets" / "Rock.obj"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"stone")

    metadata = tool_manager.get_asset_metadata("Assets/Rock.obj")

    assert metadata["exists"] is True
    assert metadata["file_size"] == 5
    assert metadata["extension"] == ".obj"
    assert metadata["relative_path"] == "Assets/Rock.obj"
    assert Path(metadata["full_path"]).name == "Rock.obj"
    assert metadata["type"] == "meshes"
    assert metadata["last_modified"].endswith("Z")

    missing = tool_manager.get_asset_metadata("Assets/Missing.fbx")
    assert missing["exists"] is False
    assert "error" in missing
