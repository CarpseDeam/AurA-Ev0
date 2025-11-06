from __future__ import annotations

from pathlib import Path

from aura.tools import symbol_tools


def test_find_definition_locates_symbol(temp_workspace):
    module = temp_workspace / "module_a.py"
    module.write_text(
        "class Target:\n    pass\n\ndef not_target():\n    return True\n",
        encoding="utf-8",
    )

    result = symbol_tools.find_definition("Target", str(temp_workspace))

    assert result["found"] is True
    assert result["file"].endswith("module_a.py")
    assert result["type"] == "class"


def test_find_usages_limits_results(temp_workspace):
    ref_file = temp_workspace / "references.py"
    ref_file.write_text("\n".join("helper()" for _ in range(120)), encoding="utf-8")

    result = symbol_tools.find_usages("helper", str(temp_workspace))

    assert result["total_usages"] == 100
    assert result["files_count"] >= 1


def test_get_imports_categorizes_modules(temp_workspace):
    file_path = temp_workspace / "imports.py"
    file_path.write_text(
        (
            "import os\n"
            "import requests\n"
            "from aura.tools import python_tools\n"
            "from .local_module import thing\n"
        ),
        encoding="utf-8",
    )

    (temp_workspace / "local_module.py").write_text("thing = 1\n", encoding="utf-8")

    result = symbol_tools.get_imports(str(file_path))

    assert "os" in result["stdlib"]
    assert "requests" in result["third_party"]
    assert "aura.tools" in result["local"]
