from __future__ import annotations

from aura.tools.tool_manager import ToolManager


def test_read_project_file_handles_existing_and_missing_files(temp_workspace):
    manager = ToolManager(str(temp_workspace))
    target = temp_workspace / "example.txt"
    target.write_text("sample", encoding="utf-8")

    assert manager.read_project_file("example.txt") == "sample"
    assert "does not exist" in manager.read_project_file("missing.txt")
    assert "outside the workspace" in manager.read_project_file("../example.txt")


def test_list_project_files_filters_extension_and_missing_dir(temp_workspace):
    manager = ToolManager(str(temp_workspace))
    pkg = temp_workspace / "pkg"
    pkg.mkdir()
    (pkg / "sample.py").write_text("# sample", encoding="utf-8")
    (pkg / "notes.md").write_text("docs", encoding="utf-8")

    files = manager.list_project_files(".", ".py")
    assert "pkg/sample.py" in files
    assert manager.list_project_files("missing-dir", ".py") == []
    assert manager.list_project_files("../", ".py")[0].startswith("Error:")


def test_search_in_files_finds_matches_and_respects_limit(temp_workspace):
    manager = ToolManager(str(temp_workspace))
    data = "\n".join(f"target line {i}" for i in range(60))
    path = temp_workspace / "big.py"
    path.write_text(data, encoding="utf-8")

    results = manager.search_in_files("target", directory=".", file_extension=".py")

    assert len(results["matches"]) == 50  # hard limit inside implementation
    assert results["matches"][0]["file"].endswith("big.py")
    denied = manager.search_in_files("target", directory="../", file_extension=".py")
    assert denied.get("error")


def test_read_multiple_files_reads_and_collects_errors(temp_workspace):
    manager = ToolManager(str(temp_workspace))
    first = temp_workspace / "first.py"
    first.write_text("print('one')", encoding="utf-8")

    results = manager.read_multiple_files(["first.py", "missing.py", "../outside.py"])

    assert results["first.py"].startswith("print")
    assert "does not exist" in results["missing.py"]
    assert "outside the workspace" in results["../outside.py"]
