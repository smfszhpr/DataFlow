#!/usr/bin/env python3
"""
Event-Driven Master Agent - 从底层重新实现支持实时事件推送的Master Agent
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .core import EventSink, Event, EventBuilder, EventType
from ..master.agent import MasterAgent, AgentState, create_master_agent
from langchain.schema import AgentAction, AgentFinish

logger = logging.getLogger(__name__)


class EventDrivenMasterAgent:
    """事件驱动的Master Agent - 原生支持实时事件推送"""
    
    def __init__(self, base_master_agent: MasterAgent):
        """基于现有Master Agent创建事件驱动版本"""
        self.base_agent = base_master_agent
        self.current_sink = None
        self.current_event_builder = None
        
        # 复制基础属性
        self.llm = base_master_agent.llm
        self.tools = base_master_agent.tools
        self.conversation_sessions = base_master_agent.conversation_sessions
        
    async def execute_with_events(
        self,
        user_input: str,
        session_id: str,
        sink: EventSink,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """执行用户请求并推送实时事件"""
        
        # 设置当前事件上下文
        self.current_sink = sink
        self.current_event_builder = EventBuilder(session_id)
        
        try:
            # 发送执行开始事件
            await self._emit_event(self.current_event_builder.run_started(
                user_input=user_input,
                execution_mode="event_driven_native"
            ))
            
            # 获取会话历史
            if session_id not in self.conversation_sessions:
                self.conversation_sessions[session_id] = []
            
            conversation_history = self.conversation_sessions[session_id]
            
            # 初始化状态
            state = AgentState(
                input=user_input,
                intermediate_steps=[],
                conversation_history=conversation_history.copy(),
                session_id=session_id
            )
            
            # 发送初始化完成事件
            await self._emit_event(self.current_event_builder.state_update({
                "phase": "initialization_completed",
                "user_input": user_input,
                "session_id": session_id,
                "conversation_history_length": len(conversation_history)
            }))
            
            # 执行工作流 - 使用可靠的手工路由，确保事件流正常
            final_state = await self._execute_workflow_with_events(state)
            
            # 获取最终输出
            output = "执行完成，但未获取到输出"
            if hasattr(final_state, 'agent_outcome') and hasattr(final_state.agent_outcome, 'return_values'):
                output = final_state.agent_outcome.return_values.get("output", "执行完成")
            
            # 保存对话历史
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": output})
            
            if len(conversation_history) > 40:
                conversation_history = conversation_history[-40:]
            
            self.conversation_sessions[session_id] = conversation_history
            
            # 发送执行完成事件
            await self._emit_event(self.current_event_builder.run_finished(
                result=output,
                total_steps=len(final_state.intermediate_steps or [])
            ))
            
            return {
                "success": True,
                "output": output,
                "session_id": session_id,
                "steps_count": len(final_state.intermediate_steps or [])
            }
            
        except Exception as e:
            error_msg = f"执行失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            await self._emit_event(self.current_event_builder.run_error(
                error=error_msg,
                session_id=session_id
            ))
            
            return {
                "success": False,
                "output": error_msg,
                "session_id": session_id,
                "error": str(e)
            }
        finally:
            self.current_sink = None
            self.current_event_builder = None
    
    async def _execute_workflow_with_events(self, state: AgentState) -> AgentState:
        """手工执行工作流逻辑，确保实时事件正常发送"""
        
        max_iterations = 10
        current_step = "bootstrap"
        
        for iteration in range(max_iterations):
            logger.info(f"🔄 工作流第 {iteration + 1} 轮，当前步骤: {current_step}")
            
            # 防止无限循环
            if iteration >= max_iterations - 1:
                logger.warning(f"⚠️ 达到最大迭代次数 ({max_iterations})，强制进入总结阶段")
                current_step = "summarize"
            
            if current_step == "bootstrap":
                await self._emit_event(self.current_event_builder.state_update({
                    "phase": "bootstrap_started",
                    "iteration": iteration + 1
                }))
                
                state = await self.base_agent.bootstrap_node(state)
                
                await self._emit_event(self.current_event_builder.state_update({
                    "phase": "bootstrap_completed",
                    "has_agent_outcome": hasattr(state, 'agent_outcome') and state.agent_outcome is not None
                }))
                
                current_step = self.base_agent.action_forward(state)
                logger.info(f"📍 Bootstrap 路由结果: {current_step}")
                
            elif current_step == "execute_tools":
                # 🔧 修复：正确处理agent_outcome的list形态
                if hasattr(state, 'agent_outcome'):
                    ao = state.agent_outcome
                    if isinstance(ao, list) and ao:
                        action = ao[0]  # 取第一个action
                        tool_name = getattr(action, "tool", "unknown")
                        tool_input = getattr(action, "tool_input", {})
                        
                        await self._emit_event(self.current_event_builder.tool_started(
                            tool_name=tool_name,
                            tool_input=tool_input
                        ))
                        logger.info(f"🔧 开始执行工具: {tool_name}")
                
                state = await self.base_agent.execute_tools_node(state)
                
                if hasattr(state, 'intermediate_steps') and state.intermediate_steps:
                    latest_action, latest_result = state.intermediate_steps[-1]
                    await self._emit_event(self.current_event_builder.tool_finished(
                        tool_name=latest_action.tool,
                        tool_output=latest_result
                    ))
                    logger.info(f"✅ 工具执行完成: {latest_action.tool}")
                
                current_step = "planner"
                logger.info(f"📍 Execute Tools 完成，回到planner继续决策")
                
            elif current_step == "planner":
                await self._emit_event(self.current_event_builder.plan_started())
                logger.info("🧠 开始规划阶段")
                
                state = await self.base_agent.planner_node(state)
                
                decision_info = {
                    "planning_completed": True,
                    "has_next_action": hasattr(state, 'agent_outcome') and state.agent_outcome is not None,
                    "iteration": iteration + 1
                }
                await self._emit_event(self.current_event_builder.plan_decision(decision_info))
                logger.info("📋 规划阶段完成")
                
                current_step = self.base_agent.action_forward(state)
                logger.info(f"📍 Planner 路由结果: {current_step}")
                
            elif current_step == "general_conversation":
                await self._emit_event(self.current_event_builder.state_update({
                    "phase": "general_conversation_started"
                }))
                
                state = await self.base_agent.general_conversation_node(state)
                
                await self._emit_event(self.current_event_builder.state_update({
                    "phase": "general_conversation_completed"
                }))
                
                current_step = self.base_agent.action_forward(state)
                logger.info(f"� General Conversation 路由结果: {current_step}")
                
            elif current_step == "summarize" or current_step == "end":
                await self._emit_event(self.current_event_builder.summarize_started())
                logger.info("📝 开始总结阶段")
                
                state = await self.base_agent.summarize_node(state)
                
                summary = "总结完成"
                if hasattr(state, 'agent_outcome') and hasattr(state.agent_outcome, 'return_values'):
                    summary = state.agent_outcome.return_values.get('output', '总结完成')
                
                await self._emit_event(self.current_event_builder.summarize_finished(summary=summary))
                logger.info("✅ 总结阶段完成")
                
                break
                
            else:
                logger.warning(f"⚠️ 未知的路由结果: {current_step}，强制进入总结阶段")
                current_step = "summarize"
                continue
        
        # 确保有最终输出
        if not (hasattr(state, 'agent_outcome') and hasattr(state.agent_outcome, 'return_values')):
            logger.warning("⚠️ 工作流结束但没有最终输出，强制执行总结")
            state = await self.base_agent.summarize_node(state)
        
        return state
    
    async def _emit_event(self, event: Event):
        """发送事件并记录日志"""
        if self.current_sink:
            await self.current_sink.emit(event)
            logger.info(f"📤 实时事件: {event.type.value}")


class EventDrivenMasterAgentExecutor:
    """事件驱动Master Agent执行器"""
    
    def __init__(self, base_master_agent: MasterAgent):
        self.event_agent = EventDrivenMasterAgent(base_master_agent)
    
    async def execute(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """兼容原有接口的执行方法"""
        from .core import PrintSink
        
        sink = PrintSink("🎭")
        return await self.event_agent.execute_with_events(
            user_input=user_input,
            session_id=session_id,
            sink=sink
        )
    
    async def run_with_events(
        self,
        user_input: str,
        session_id: str,
        sink: EventSink,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """带事件推送的执行方法"""
        return await self.event_agent.execute_with_events(
            user_input=user_input,
            session_id=session_id,
            sink=sink,
            conversation_history=conversation_history
        )


def create_event_driven_master_agent() -> tuple[EventDrivenMasterAgent, EventDrivenMasterAgentExecutor]:
    """创建事件驱动的Master Agent"""
    base_agent, _ = create_master_agent()
    event_agent = EventDrivenMasterAgent(base_agent)
    executor = EventDrivenMasterAgentExecutor(base_agent)
    return event_agent, executor


# 测试函数
async def test_event_driven_agent():
    """测试事件驱动Master Agent"""
    print("🧪 测试事件驱动Master Agent...")
    
    # 创建事件驱动Agent
    event_agent, executor = create_event_driven_master_agent()
    
    # 创建详细的事件接收器
    class DetailedPrintSink:
        async def emit(self, event):
            now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"📡 [{now}] {event.type.value}")
            
            # 显示详细数据
            if event.data:
                for key, value in event.data.items():
                    if key == 'tool_output' and isinstance(value, dict):
                        if 'apikey' in value:
                            print(f"    🔑 API密钥: {value['apikey']}")
                        if 'message' in value:
                            print(f"    💬 消息: {value['message']}")
                    elif key == 'user_input':
                        print(f"    👤 用户输入: {value}")
                    elif key == 'phase':
                        print(f"    📍 阶段: {value}")
        
        async def close(self):
            pass
    
    sink = DetailedPrintSink()
    
    # 测试执行
    result = await executor.run_with_events(
        user_input="获取今天的API密钥",
        session_id="test_event_driven_001",
        sink=sink
    )
    
    print(f"\n✅ 测试结果: {result}")


if __name__ == "__main__":
    asyncio.run(test_event_driven_agent())


# 导出
__all__ = [
    'EventDrivenMasterAgent',
    'EventDrivenMasterAgentExecutor', 
    'create_event_driven_master_agent'
]
