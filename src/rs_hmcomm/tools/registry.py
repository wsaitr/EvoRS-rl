"""Explicit allow-list registry for agent tools."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rs_hmcomm.agents.action_protocol import CallTool

from .base import Tool, ToolContext, ToolError, ToolResult


class UnknownToolError(ToolError, KeyError):
    pass


class DuplicateToolError(ToolError, ValueError):
    pass


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("tool must satisfy the Tool protocol")
        name = tool.name
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool name must be a non-empty string")
        if name in self._tools:
            raise DuplicateToolError(f"tool is already registered: {name}")
        self._tools[name] = tool

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(f"tool is not registered: {name}") from exc

    def execute(self, call: CallTool, context: ToolContext) -> ToolResult:
        return self.get(call.tool_name).execute(call, context)

    def descriptions(self) -> tuple[dict[str, Any], ...]:
        """Return JSON-safe tool descriptions suitable for an agent prompt."""

        return tuple(
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        )


__all__ = ["DuplicateToolError", "ToolRegistry", "UnknownToolError"]
