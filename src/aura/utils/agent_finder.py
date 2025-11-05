"""Discover and validate CLI agents on the system."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class AgentInfo:
    """Metadata for a discovered CLI agent."""

    name: str
    display_name: str
    executable_path: str
    version: str
    is_available: bool


def find_cli_agents() -> list[AgentInfo]:
    """Discover all available CLI agents on the system."""
    agents = []
    agent_configs = [
        ("gemini", "Gemini CLI", ["gemini", "gemini.cmd", "gemini.exe"]),
        ("claude", "Claude Code", ["claude", "claude.cmd", "claude.exe"]),
        ("codex", "Codex", ["codex", "codex.cmd", "codex.exe"]),
    ]

    for agent_name, display_name, executables in agent_configs:
        executable_path = _find_executable(executables)
        if executable_path:
            is_valid, version = validate_agent(executable_path, agent_name)
            agents.append(
                AgentInfo(
                    name=agent_name,
                    display_name=display_name,
                    executable_path=executable_path,
                    version=version,
                    is_available=is_valid,
                )
            )
        else:
            agents.append(
                AgentInfo(
                    name=agent_name,
                    display_name=display_name,
                    executable_path="",
                    version="unknown",
                    is_available=False,
                )
            )

    return agents


def validate_agent(executable_path: str, agent_name: str) -> tuple[bool, str]:
    """Check if agent executable works and get version."""
    try:
        result = subprocess.run(
            [executable_path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip() or "unknown"
            version = version.split("\n")[0][:50]
            return True, version
        return False, "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False, "unknown"


def _find_executable(names: list[str]) -> str:
    """Search for executable in PATH and common locations."""
    for name in names:
        path = shutil.which(name)
        if path:
            return path

    for name in names:
        for search_path in _get_search_paths():
            candidate = os.path.join(search_path, name)
            if os.path.isfile(candidate):
                return candidate

    return ""


def _get_search_paths() -> list[str]:
    """Get platform-specific search paths for CLI tools."""
    paths = []
    if os.name == "nt":
        appdata = os.getenv("APPDATA", "")
        localappdata = os.getenv("LOCALAPPDATA", "")
        userprofile = os.getenv("USERPROFILE", "")
        if appdata:
            paths.append(os.path.join(appdata, "npm"))
        if localappdata:
            paths.append(os.path.join(localappdata, "Programs"))
        if userprofile:
            paths.append(os.path.join(userprofile, "AppData", "Roaming", "npm"))
    else:
        paths.extend([
            "/usr/local/bin",
            os.path.expanduser("~/.local/bin"),
            os.path.expanduser("~/node_modules/.bin"),
        ])

    return [p for p in paths if os.path.isdir(p)]


__all__ = ["AgentInfo", "find_cli_agents", "validate_agent"]
