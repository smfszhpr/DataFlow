"""
Common State Definitions - 避免循环导入的通用状态定义
"""

from typing import Any, Dict, List, Optional, TypedDict


class BaseAgentState(TypedDict, total=False):
    """基础Agent状态定义 - 兼容 myscalekb_agent_base"""
    # myscalekb_agent_base 标准字段
    input: Any
    query: str
    chat_history: List[Any]
    agent_metadata: Any  # AgentMetadata
    agent_outcome: Any
    intermediate_steps: List[Any]
    trace_id: Optional[str]
    
    # 通用扩展字段
    session_id: Optional[str]
    current_step: str
    form_data: Optional[Dict[str, Any]]
    xml_content: Optional[str]
    execution_result: Optional[str]
    conversation_history: List[Dict[str, str]]
    last_tool_results: Optional[Dict[str, Any]]
    
    # 多轮编排支持
    pending_actions: List[Any]
    tool_results: List[Dict[str, Any]]
    loop_guard: int
    max_steps: int
    context_vars: Dict[str, Any]
    next_action: Optional[str]
