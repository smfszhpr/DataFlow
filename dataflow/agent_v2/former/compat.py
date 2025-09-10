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


class FormerAgentCompat:
    """Former Agent 兼容层
    
    提供与旧版本FormerAgent相同的接口，内部使用新的FormerAgentV2实现
    """
    
    def __init__(self):
        self.agent_v2 = FormerAgentV2()
        self.session_states = {}  # 存储会话状态
    
    async def process_conversation(self, form_request: FormRequest) -> FormResponse:
        """处理对话请求，兼容旧版本接口"""
        try:
            # 使用新版本agent处理请求（同步调用）
            result = self.agent_v2.process_request(
                user_requirement=form_request.user_query,
                user_input=form_request.user_query
            )
            
            # 分析结果，判断是否需要更多信息
            need_more_info = not result.get("is_complete", False)
            
            # 构建响应消息
            if result.get("success", False):
                if result.get("xml_content"):
                    # 已生成XML，完成任务
                    agent_response = f"✅ 已为您生成{result.get('form_type', '表单')}的XML配置。"
                    xml_form = result.get("xml_content", "")
                else:
                    # 需要更多信息
                    missing_fields = result.get("missing_fields", [])
                    if missing_fields:
                        agent_response = f"为了完成{result.get('form_type', '表单')}的生成，还需要以下信息：\n" + \
                                       "\n".join([f"• {field}" for field in missing_fields])
                    else:
                        agent_response = f"正在处理您的{result.get('form_type', '表单')}需求，请稍后..."
                    xml_form = None
            else:
                # 处理失败
                agent_response = f"处理您的请求时遇到问题：{result.get('error_message', '未知错误')}"
                xml_form = None
            
            # 更新会话历史
            session_id = form_request.session_id or "default"
            if session_id not in self.session_states:
                self.session_states[session_id] = []
            
            self.session_states[session_id].extend([
                {"role": "user", "content": form_request.user_query},
                {"role": "assistant", "content": agent_response}
            ])
            
            # 构建响应
            response = FormResponse(
                need_more_info=need_more_info,
                agent_response=agent_response,
                xml_form=xml_form,
                form_type=result.get("form_type"),
                conversation_history=self.session_states[session_id]
            )
            
            return response
            
        except Exception as e:
            # 错误处理
            error_response = FormResponse(
                need_more_info=False,
                agent_response=f"抱歉，处理您的请求时发生错误：{str(e)}",
                xml_form=None,
                form_type=None,
                conversation_history=form_request.conversation_history
            )
            return error_response
    
    def process_request(self, user_requirement: str, user_input: str = "") -> Dict[str, Any]:
        """直接访问新版本的process_request方法"""
        return self.agent_v2.process_request(user_requirement, user_input)
