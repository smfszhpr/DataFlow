#!/usr/bin/env python3
"""
Continue Chat Tool - 前端WebSocket用户输入工具
注意：此工具不应该被 Master Agent 直接调用，只应该被 Former Tool 调用
"""

import logging
import asyncio
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

from dataflow.agent_v2.base.core import BaseTool

logger = logging.getLogger(__name__)


class ContinueChatToolParams(BaseModel):
    """Continue Chat Tool 参数"""
    prompt: str = Field(description="向用户展示的提示信息")
    context: Optional[str] = Field(default="", description="当前上下文信息")
    timeout_seconds: Optional[int] = Field(default=60, description="等待用户响应的超时时间")


class ContinueChatTool(BaseTool):
    """Continue Chat Tool - 前端WebSocket用户输入工具"""
    
    def __init__(self):
        # 存储等待响应的会话
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._setup_websocket_integration()
    
    def _setup_websocket_integration(self):
        """设置 WebSocket 集成"""
        try:
            # 延迟导入以避免循环依赖
            from dataflow.agent_v2.websocket.events import event_router
            
            # 注册用户响应处理器
            event_router.register_handler("continue_chat_response", self._handle_websocket_response)
            logger.info("✅ Continue Chat Tool WebSocket 集成已设置")
            
        except Exception as e:
            logger.warning(f"⚠️ WebSocket 集成设置失败: {e}")
    
    @classmethod
    def name(cls) -> str:
        return "continue_chat"
    
    @classmethod
    def description(cls) -> str:
        return """【内部工具 - 不直接调用】通过前端WebSocket获取用户输入补充信息。

【前置工具】：former_agent - 只有Former工具可以调用此工具
【后置工具】：返回Former工具或其他后续工具

此工具专门用于：
- Former工具检测到需求信息不完整时
- 需要用户补充具体信息时
- 多轮交互收集详细需求时

⚠️ 注意：此工具不应被Master Agent直接调用，只应作为其他工具的后置工具使用。"""
    
    @classmethod
    def params(cls) -> type:
        return ContinueChatToolParams
    
    @classmethod
    def prerequisite_tools(cls) -> List[str]:
        """前置工具列表"""
        return []  # 不再自动推荐
    @classmethod
    def suggested_followup_tools(cls) -> List[str]:
        """建议的后置工具列表"""
        return []  # 不再自动推荐
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行 Continue Chat 工具"""
        try:
            # 从kwargs创建参数对象
            params = ContinueChatToolParams(**kwargs)
            prompt = params.prompt
            context = params.context
            timeout_seconds = params.timeout_seconds or 60
            
            logger.info(f"Continue Chat Tool 启动: {prompt}")
            
            # 从当前执行上下文中获取 session_id
            session_id = self._get_current_session_id()
            if not session_id:
                logger.warning("⚠️ 无法获取session_id，使用后备模式")
                return await self._fallback_user_input(prompt, context)
            
            logger.info(f"📱 获取到session_id: {session_id}")
            
            # 检查WebSocket连接是否存在
            try:
                from dataflow.agent_v2.websocket.events import connection_manager
                ws_sink = await connection_manager.get_connection(session_id)
                if not ws_sink:
                    logger.warning(f"⚠️ WebSocket连接不存在 (session: {session_id})，使用后备模式")
                    return await self._fallback_user_input(prompt, context)
                logger.info(f"✅ WebSocket连接存在: {session_id}")
            except Exception as e:
                logger.warning(f"⚠️ 检查WebSocket连接失败: {e}，使用后备模式")
                return await self._fallback_user_input(prompt, context)
            
            # 构建发送给前端的消息
            message_to_frontend = {
                "type": "user_input_request",
                "prompt": prompt,
                "context": context,
                "timestamp": asyncio.get_event_loop().time(),
                "request_id": f"continue_chat_{session_id}_{int(asyncio.get_event_loop().time())}"
            }
            
            logger.info(f"📤 准备发送消息到前端: {message_to_frontend}")
            
            # 发送到前端WebSocket
            success = await self._send_to_frontend(session_id, message_to_frontend)
            if not success:
                logger.warning("❌ WebSocket 发送失败，使用后备模式")
                return await self._fallback_user_input(prompt, context)
            
            logger.info(f"📤 已发送用户输入请求到前端，等待响应...")
            
            # 等待用户响应
            try:
                # 创建等待响应的 Future
                future = asyncio.Future()
                self._pending_requests[session_id] = future
                
                logger.info(f"⏰ 开始等待用户响应 (超时: {timeout_seconds}秒)...")
                user_response = await asyncio.wait_for(future, timeout=timeout_seconds)
                
                logger.info(f"✅ 收到用户响应: {user_response}")
                
                # 检查是否有Former Tool的会话上下文
                former_session_context = self._get_former_session_context(context)
                if former_session_context:
                    logger.info(f"🔗 检测到Former Tool会话，调用Former Tool继续对话")
                    return await self._handle_former_tool_continue(user_response, former_session_context)
                
                # 默认返回用户响应
                return {
                    "success": True,
                    "user_input": user_response,
                    "output": f"用户响应: {user_response}",
                    "context_updated": True
                }
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ 等待用户响应超时 ({timeout_seconds}秒)")
                return {
                    "success": False,
                    "error": "用户响应超时",
                    "output": f"等待用户响应超时 ({timeout_seconds}秒)，请稍后重试",
                    "timeout": True
                }
            finally:
                # 清理等待中的请求
                self._pending_requests.pop(session_id, None)
                logger.info(f"🧹 清理等待请求: {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Continue Chat Tool 执行失败: {e}")
            import traceback
            logger.error(f"❌ 详细错误信息:\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "output": f"用户输入获取失败: {str(e)}"
            }
    
    def _get_former_session_context(self, context: str) -> Optional[Dict[str, Any]]:
        """从上下文中提取Former Tool的会话信息"""
        try:
            # 尝试从上下文中解析Former Tool相关信息
            # 检查是否包含Former Tool的关键词
            if "former_tool_active" in context or "session_id" in context:
                import json
                # 尝试解析JSON格式的上下文
                try:
                    context_data = json.loads(context)
                    if isinstance(context_data, dict) and context_data.get("former_tool_active"):
                        return context_data
                except json.JSONDecodeError:
                    pass
                
                # 从字符串中提取session_id
                import re
                session_match = re.search(r'session_id["\']?\s*:\s*["\']?([a-f0-9-]+)', context)
                if session_match:
                    return {
                        "session_id": session_match.group(1),
                        "former_tool_active": True
                    }
            
            return None
        except Exception as e:
            logger.warning(f"解析Former Tool会话上下文失败: {e}")
            return None
    
    async def _handle_former_tool_continue(self, user_response: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """处理Former Tool的继续对话"""
        try:
            logger.info(f"📝 调用Former Tool继续对话: {session_context}")
            
            # 获取Former Tool实例
            from dataflow.agent_v2.former.former_tool import FormerTool, FormerToolParams
            former_tool = FormerTool()
            
            # 构建Former Tool参数
            params = FormerToolParams(
                user_query=user_response,
                session_id=session_context.get("session_id"),
                action="continue_chat",
                user_response=user_response
            )
            
            # 调用Former Tool的continue_chat功能
            result = former_tool.execute(params)
            
            if result.get("success"):
                # 检查是否还需要继续对话
                if result.get("requires_user_input"):
                    return {
                        "success": True,
                        "output": result.get("message", ""),
                        "requires_user_input": True,
                        "session_id": result.get("session_id"),
                        "form_stage": result.get("form_stage"),
                        "followup_recommendation": {
                            "needs_followup": True,
                            "tool_name": "continue_chat",
                            "reason": "Former Tool需要继续用户交互",
                            "session_context": {
                                "session_id": result.get("session_id"),
                                "form_stage": result.get("form_stage"),
                                "former_tool_active": True
                            }
                        }
                    }
                else:
                    # 表单已完成，可能需要调用后置工具
                    return {
                        "success": True,
                        "output": result.get("message", ""),
                        "form_complete": True,
                        "form_data": result.get("form_data", {}),
                        "session_id": result.get("session_id"),
                        "followup_recommendation": {
                            "needs_followup": True,
                            "tool_name": "code_workflow_agent",
                            "reason": "表单已完成，建议执行代码生成工作流",
                            "form_data": result.get("form_data", {})
                        }
                    }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Former Tool执行失败"),
                    "output": result.get("message", "Former Tool处理失败")
                }
                
        except Exception as e:
            logger.error(f"Former Tool继续对话失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": f"Former Tool继续对话失败: {str(e)}"
            }
    
    def _get_current_session_id(self) -> Optional[str]:
        """获取当前会话ID"""
        # 尝试从多个可能的地方获取 session_id
        import contextvars
        import inspect
        
        # 方法1：从 contextvars 获取（如果设置了的话）
        try:
            session_var = contextvars.ContextVar('session_id', default=None)
            session_id = session_var.get()
            if session_id:
                logger.info(f"📱 从contextvars获取session_id: {session_id}")
                return session_id
        except:
            pass
        
        # 方法2：从调用栈中查找Master Agent的状态
        try:
            frame = inspect.currentframe()
            while frame:
                # 查找frame中的局部变量
                local_vars = frame.f_locals
                
                # 检查是否包含AgentState类型的数据
                for var_name, var_value in local_vars.items():
                    if hasattr(var_value, 'get') and isinstance(var_value, dict):
                        # 检查字典是否包含session_id
                        if 'session_id' in var_value:
                            session_id = var_value['session_id']
                            logger.info(f"📱 从调用栈获取session_id: {session_id} (来源: {var_name})")
                            return session_id
                    
                    # 检查是否是AgentState对象
                    if hasattr(var_value, 'session_id'):
                        session_id = getattr(var_value, 'session_id', None)
                        if session_id:
                            logger.info(f"📱 从AgentState对象获取session_id: {session_id}")
                            return session_id
                
                frame = frame.f_back
        except Exception as e:
            logger.debug(f"从调用栈获取session_id失败: {e}")
        
        # 方法3：从 WebSocket 连接管理器获取活跃会话
        try:
            from dataflow.agent_v2.websocket.events import connection_manager
            active_sessions = connection_manager.get_active_sessions()
            if active_sessions:
                # 返回第一个活跃会话（通常只有一个）
                session_id = active_sessions[0]
                logger.info(f"📱 从connection_manager获取session_id: {session_id}")
                return session_id
        except Exception as e:
            logger.debug(f"从connection_manager获取session_id失败: {e}")
        
        logger.warning("⚠️ 无法获取session_id")
        return None
    
    async def _send_to_frontend(self, session_id: str, message: Dict[str, Any]) -> bool:
        """发送消息到前端WebSocket"""
        try:
            from dataflow.agent_v2.websocket.events import connection_manager
            from dataflow.agent_v2.events import EventBuilder
            
            # 获取WebSocket连接
            ws_sink = await connection_manager.get_connection(session_id)
            if not ws_sink:
                logger.warning(f"未找到会话 {session_id} 的WebSocket连接")
                return False
            
            # 构建事件并发送
            event_builder = EventBuilder(session_id)
            await ws_sink.emit(event_builder.state_update({
                "type": "user_input_request",
                "data": message
            }))
            
            logger.info(f"📤 已发送用户输入请求到前端: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"发送消息到前端失败: {e}")
            return False
    
    async def _handle_websocket_response(self, response_data: Dict[str, Any], session_id: str):
        """处理来自WebSocket的用户响应"""
        try:
            user_input = response_data.get("user_input", "")
            
            # 检查是否有等待中的请求
            if session_id in self._pending_requests:
                future = self._pending_requests[session_id]
                if not future.done():
                    future.set_result(user_input)
                    logger.info(f"✅ 用户响应已传递 [session: {session_id}]: {user_input}")
                else:
                    logger.warning(f"⚠️ Future已完成，忽略响应 [session: {session_id}]")
            else:
                logger.warning(f"⚠️ 未找到等待中的请求 [session: {session_id}]")
                
        except Exception as e:
            logger.error(f"处理WebSocket响应失败: {e}")
    
    async def _fallback_user_input(self, prompt: str, context: str) -> Dict[str, Any]:
        """后备用户输入方式（控制台输入）"""
        try:
            logger.info(f"使用控制台后备模式获取用户输入")
            
            # 显示提示信息
            print(f"\n🤖 {prompt}")
            if context:
                print(f"📋 上下文: {context}")
            
            # 获取用户输入（在实际部署中这应该通过WebSocket）
            user_input = input("👤 请输入您的回复: ").strip()
            
            return {
                "success": True,
                "user_input": user_input,
                "output": f"用户输入: {user_input}",
                "fallback_mode": True
            }
            
        except Exception as e:
            logger.error(f"后备用户输入失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": f"用户输入获取失败: {str(e)}"
            }


# 全局实例（用于WebSocket回调）
_continue_chat_tool_instance = None

def get_continue_chat_tool():
    """获取 Continue Chat Tool 全局实例"""
    global _continue_chat_tool_instance
    if _continue_chat_tool_instance is None:
        _continue_chat_tool_instance = ContinueChatTool()
    return _continue_chat_tool_instance
