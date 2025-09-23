"""
Former Agent 兼容性模块
为了保持与旧版本 Master Agent 的兼容性
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from .agent import FormerAgentV2


class FormRequest(BaseModel):
    """兼容旧版本的FormRequest模型"""
    user_query: str
    conversation_history: List[Dict[str, str]] = []
    session_id: Optional[str] = None


class FormResponse(BaseModel):
    """兼容旧版本的FormResponse模型"""
    need_more_info: bool
    agent_response: str
    xml_form: Optional[str] = None
    form_type: Optional[str] = None
    conversation_history: List[Dict[str, str]] = []
