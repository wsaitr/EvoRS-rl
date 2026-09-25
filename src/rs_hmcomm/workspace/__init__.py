"""Controlled shared workspace for EvoRS-Comm episodes."""

from .reducer import WorkspaceConflictError, WorkspaceReducer, build_scoped_view
from .state import (
    ScopedWorkspaceView,
    WorkspaceDelta,
    WorkspaceIntegrityError,
    WorkspaceState,
)

__all__ = [
    "ScopedWorkspaceView",
    "WorkspaceConflictError",
    "WorkspaceDelta",
    "WorkspaceIntegrityError",
    "WorkspaceReducer",
    "WorkspaceState",
    "build_scoped_view",
]
