from __future__ import annotations

from pathlib import Path

from aura.tools import file_system_tools as fs_tools


def test_read_project_file_handles_existing_and_missing_files(temp_workspace, monkeypatch):
    monkeypatch.chdir(temp_workspace)
    target = Path("example.txt")
    target.write_text("sample", encoding="utf-8")

    assert fs_tools.read_project_file("example.txt") == "sample"
    assert "does not exist" in fs_tools.read_project_file("missing.txt")


def test_list_project_files_filters_extension_and_missing_dir(temp_workspace, monkeypatch):
    monkeypatch.chdir(temp_workspace)
    pkg = Path("pkg")
    pkg.mkdir()
    (pkg / "sample.py").write_text("# sample", encoding="utf-8")
    (pkg / "notes.md").write_text("docs", encoding="utf-8")

    files = fs_tools.list_project_files(".", ".py")
    assert str(pkg / "sample.py") in files
    assert fs_tools.list_project_files("missing-dir", ".py") == []


def test_search_in_files_finds_matches_and_respects_limit(temp_workspace, monkeypatch):
    monkeypatch.chdir(temp_workspace)
    data = "\n".join(f"target line {i}" for i in range(60))
    path = Path("big.py")
    path.write_text(data, encoding="utf-8")

    results = fs_tools.search_in_files("target", directory=".", file_extension=".py")

    assert len(results["matches"]) == 50  # hard limit inside implementation
    assert results["matches"][0]["file"].endswith("big.py")


def test_read_multiple_files_reads_and_collects_errors(temp_workspace, monkeypatch):
    monkeypatch.chdir(temp_workspace)
    first = Path("first.py")
    first.write_text("print('one')", encoding="utf-8")

    results = fs_tools.read_multiple_files(["first.py", "missing.py"])

    assert results["first.py"].startswith("print")
    assert "does not exist" in results["missing.py"]
