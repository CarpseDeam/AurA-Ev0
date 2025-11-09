"""Project context tools that understand dependencies, docs, and databases."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

try:
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover
    Requirement = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


def get_project_dependencies(
    requirements_path: str = "requirements.txt",
) -> dict:
    """Return declared dependencies from requirements.txt and/or pyproject.toml."""
    LOGGER.info("?? TOOL CALLED: get_project_dependencies")
    project_root = Path.cwd()
    summary: dict[str, Any] = {"requirements": [], "pyproject": {}, "errors": []}

    requirements_file = _resolve_path(requirements_path)
    if requirements_file.exists():
        summary["requirements"] = _parse_requirements_file(requirements_file)
    else:
        summary["errors"].append(f"requirements.txt not found at {requirements_file}")

    pyproject_file = _resolve_path(pyproject_path)
    if pyproject_file.exists() and tomllib:
        summary["pyproject"] = _parse_pyproject(pyproject_file)
    elif not pyproject_file.exists():
        summary["errors"].append(f"pyproject.toml not found at {pyproject_file}")
    elif not tomllib:
        summary["errors"].append("tomllib is not available to parse pyproject.toml")

    if summary["requirements"] or summary["pyproject"]:
        summary["project_root"] = str(project_root)
    else:
        summary["message"] = "No dependency manifests found."

    return summary


def update_documentation(
    symbol_name: str,
    old_reference: str,
    new_reference: str,
) -> dict:
    """Update Markdown docs that reference a symbol or old signature."""
    LOGGER.info("?? TOOL CALLED: update_documentation(%s)", symbol_name)
    docs_path = _resolve_path(docs_directory)
    if not docs_path.exists():
        return {"updated": 0, "error": f"Docs directory does not exist: {docs_directory}"}

    pattern = old_reference or symbol_name
    if not pattern:
        return {"updated": 0, "error": "A symbol name or old reference must be provided."}

    updated_files: list[str] = []
    replacements = 0

    for md_file in _iter_markdown_files(docs_path):
        content = md_file.read_text(encoding="utf-8")
        if pattern not in content:
            continue
        new_content = content.replace(pattern, new_reference)
        if new_content != content:
            md_file.write_text(new_content, encoding="utf-8")
            updated_files.append(str(md_file))
            replacements += content.count(pattern)

    if not updated_files:
        return {
            "updated": 0,
            "message": f"No references to '{pattern}' found within {docs_directory}.",
        }

    return {
        "updated": len(updated_files),
        "replacements": replacements,
        "files": updated_files,
    }


def get_database_schema(connection_url: str = "") -> dict:
    """Inspect a SQL database using SQLAlchemy and return a schema summary."""
    LOGGER.info("?? TOOL CALLED: get_database_schema")
    try:
        from sqlalchemy import create_engine, inspect
    except ImportError:
        return {
            "success": False,
            "error": "SQLAlchemy is not installed. Install it with: pip install SQLAlchemy",
        }

    connection = connection_url or os.getenv("DATABASE_URL")
    if not connection:
        return {
            "success": False,
            "error": "Provide a connection URL or set the DATABASE_URL environment variable.",
        }

    engine = None
    try:
        engine = create_engine(connection)
        inspector = inspect(engine)

        schema: dict[str, Any] = {}
        for table in inspector.get_table_names():
            columns = [
                {
                    "name": column.get("name"),
                    "type": str(column.get("type")),
                    "nullable": column.get("nullable"),
                    "default": column.get("default"),
                }
                for column in inspector.get_columns(table)
            ]
            schema[table] = {
                "columns": columns,
                "primary_key": inspector.get_pk_constraint(table).get("constrained_columns", []),
                "foreign_keys": inspector.get_foreign_keys(table),
                "indexes": inspector.get_indexes(table),
            }

        return {
            "success": True,
            "engine": engine.name,
            "database": _redact_connection_url(connection),
            "tables": schema,
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Database schema inspection failed: %s", exc)
        return {"success": False, "error": f"Failed to inspect database: {exc}"}
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:  # noqa: BLE001
                pass


def _resolve_path(path_like: str) -> Path:
    candidate = Path(path_like)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _parse_requirements_file(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in stripped:
            stripped = stripped.split("#", 1)[0].strip()
        if not stripped:
            continue
        entries.append(_parse_requirement(stripped))
    return entries


def _parse_requirement(spec: str) -> dict[str, Any]:
    if Requirement:
        try:
            req = Requirement(spec)
            return {
                "name": req.name,
                "specifier": str(req.specifier) or None,
                "extras": sorted(req.extras),
                "marker": str(req.marker) if req.marker else None,
            }
        except Exception:
            pass
    return {"name": spec, "specifier": None, "extras": [], "marker": None}


def _parse_pyproject(path: Path) -> dict[str, Any]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project_section = data.get("project", {})
    optional = project_section.get("optional-dependencies", {})
    tool_section = data.get("tool", {})
    poetry = tool_section.get("poetry", {})

    return {
        "project": project_section.get("dependencies", []),
        "optional": optional,
        "tool_poetry": {
            "dependencies": poetry.get("dependencies"),
            "dev-dependencies": poetry.get("dev-dependencies"),
        },
    }


def _iter_markdown_files(base: Path) -> list[Path]:
    if base.is_file() and base.suffix.lower() == ".md":
        return [base]
    files = [path for path in base.rglob("*.md") if path.is_file()]
    return files


def _redact_connection_url(url: str) -> str:
    if "://" not in url:
        return url
    scheme, remainder = url.split("://", 1)
    if "@" not in remainder:
        return f"{scheme}://***"
    _, tail = remainder.split("@", 1)
    return f"{scheme}://***@{tail}"
