"""
Master Agent 适配器模块
"""

from .tool_adapter import (
    ToolStatus,
    ToolArtifact,
    ToolFollowup,
    ToolResult,
    ToolResultAdapter,
    adapt_tool_result
)

__all__ = [
    "ToolStatus",
    "ToolArtifact", 
    "ToolFollowup",
    "ToolResult",
    "ToolResultAdapter",
    "adapt_tool_result"
]
