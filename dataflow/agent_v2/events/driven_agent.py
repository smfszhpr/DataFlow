#!/usr/bin/env python3
"""
Event-Driven Master Agent - 基于LangGraph astream的实时事件推送
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
    """事件驱动的Master Agent - 使用LangGraph原生astream消除代码冗余"""
    
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
                execution_mode="event_driven_langgraph"
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
            
            # 🔧 使用LangGraph原生astream_events，消除代码冗余
            final_state, steps_count = await self._execute_with_langgraph_events(state)
            
            # 🚀 优化：简化最终输出获取逻辑
            output = "执行完成"
            
            # 优先从final_state.output获取
            if hasattr(final_state, 'output') and final_state.output:
                output = final_state.output
            # 其次从agent_outcome获取
            elif hasattr(final_state, 'agent_outcome') and final_state.agent_outcome:
                if hasattr(final_state.agent_outcome, 'return_values'):
                    output = final_state.agent_outcome.return_values.get("output", output)
                elif hasattr(final_state.agent_outcome, 'output'):
                    output = final_state.agent_outcome.output
            # 最后从intermediate_steps获取
            elif hasattr(final_state, 'intermediate_steps') and final_state.intermediate_steps:
                steps_count = max(steps_count, len(final_state.intermediate_steps))
                last_step = final_state.intermediate_steps[-1]
                if len(last_step) >= 2:
                    output = f"工具执行结果: {last_step[1]}"
            
            # 保存对话历史
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": output})
            
            if len(conversation_history) > 40:
                conversation_history = conversation_history[-40:]
            
            self.conversation_sessions[session_id] = conversation_history
            
            # 发送执行完成事件
            await self._emit_event(self.current_event_builder.run_finished(
                result=output,
                total_steps=steps_count
            ))
            
            return {
                "success": True,
                "output": output,
                "session_id": session_id,
                "steps_count": steps_count
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

    async def _execute_with_langgraph_events(self, state: AgentState) -> tuple[AgentState, int]:
        """使用LangGraph原生astream_events执行，参考myscalekb-agent的实现"""
        
        graph = self.base_agent.compiled_graph
        step_count = 0  # 🚀 优化：基于工具完成次数统计
        final_state = state
        root_finished = False
        
        # 使用astream_events获取标准事件流 - 借鉴myscalekb-agent的方法
        async for event in graph.astream_events(state, config={"recursion_limit": 15}, version="v2"):
            # 过滤隐藏事件
            if "langsmith:hidden" in event.get("tags", []):
                continue
                
            kind = event["event"]
            run_id = event.get("run_id")
            event_name = event.get("name", "")
            tags = event.get("tags", [])
            
            logger.debug(f"🔄 LangGraph事件: {kind} | 名称: {event_name}")
            
            # 🎯 优化：识别根图结束事件获取最终状态
            if kind == "on_chain_end" and ("langgraph:root" in tags or event_name in ["graph", "compiled_graph"]):
                output_data = event.get("data", {}).get("output", {})
                logger.debug(f"🏁 根图结束，获取最终状态: {type(output_data)}")
                
                if isinstance(output_data, dict):
                    try:
                        # 尝试从字典构造AgentState
                        for key, value in output_data.items():
                            if hasattr(final_state, key):
                                setattr(final_state, key, value)
                        logger.debug(f"✅ 最终状态更新成功")
                    except Exception as e:
                        logger.warning(f"⚠️ 最终状态更新失败: {e}")
                root_finished = True
                
            # 处理不同类型的事件
            elif kind == "on_chain_start":
                # 🚀 优化：不再用chain_start统计步数
                await self._handle_chain_start_event(event)
                
            elif kind == "on_tool_start":
                await self._handle_tool_start_event(event)
                
            elif kind == "on_tool_end":
                # 🚀 优化：基于工具完成统计实际执行步数
                step_count += 1
                await self._handle_tool_end_event(event)
                
            elif kind == "on_chat_model_start":
                await self._handle_model_start_event(event)
                
            elif kind == "on_chat_model_stream":
                await self._handle_model_stream_event(event)
                
            elif kind == "on_chat_model_end":
                await self._handle_model_end_event(event)
                
            elif kind == "on_chain_end":
                final_state = await self._handle_chain_end_event(event, final_state)
        
        # 🚀 优化：最终步数以intermediate_steps为权威
        try:
            authoritative_step_count = len(getattr(final_state, 'intermediate_steps', []))
            if authoritative_step_count > 0:
                step_count = authoritative_step_count
                logger.debug(f"📊 权威步数统计: {step_count} (基于intermediate_steps)")
        except Exception as e:
            logger.warning(f"⚠️ 权威步数统计失败: {e}")
        
        logger.info(f"🎯 执行完成，根图状态: {root_finished}, 最终步数: {step_count}")
        return final_state, step_count
    
    # ===== 新的事件处理方法 - 参考myscalekb-agent =====
    
    async def _handle_chain_start_event(self, event: dict):
        """处理链式调用开始事件 - 优化：减少冗余事件"""
        event_name = event.get("name", "")
        logger.debug(f"🔗 链式调用开始: {event_name}")
        
        # 🚀 优化：只发送关键节点的开始事件，减少冗余
        if event_name == "planner":
            await self._emit_event(self.current_event_builder.plan_started())
        elif event_name == "summarize":
            await self._emit_event(self.current_event_builder.summarize_started())
        # bootstrap节点不再发送开始事件，减少冗余
    
    async def _handle_tool_start_event(self, event: dict):
        """处理工具开始事件"""
        tool_data = event.get("data", {})
        tool_name = tool_data.get("name", "unknown_tool")
        tool_input = tool_data.get("input", {})
        
        logger.debug(f"🔧 工具开始: {tool_name}")
        await self._emit_event(self.current_event_builder.tool_started(
            tool_name=tool_name,
            tool_input=tool_input
        ))
    
    async def _handle_tool_end_event(self, event: dict):
        """处理工具结束事件"""
        tool_data = event.get("data", {})
        tool_name = tool_data.get("name", "unknown_tool")
        tool_output = tool_data.get("output", "")
        
        logger.debug(f"✅ 工具完成: {tool_name}")
        await self._emit_event(self.current_event_builder.tool_finished(
            tool_name=tool_name,
            tool_output=tool_output
        ))
    
    async def _handle_model_start_event(self, event: dict):
        """处理模型开始事件"""
        logger.debug("🤖 模型开始生成")
        await self._emit_event(self.current_event_builder.model_started())
    
    async def _handle_model_stream_event(self, event: dict):
        """处理模型流事件"""
        chunk_data = event.get("data", {})
        chunk = chunk_data.get("chunk", {})
        content = chunk.get("content", "")
        
        if content:
            await self._emit_event(self.current_event_builder.model_streaming(content))
    
    async def _handle_model_end_event(self, event: dict):
        """处理模型结束事件"""
        logger.debug("🤖 模型生成完成")
        await self._emit_event(self.current_event_builder.model_finished())
    
    async def _handle_chain_end_event(self, event: dict, current_state: AgentState) -> AgentState:
        """处理链式调用结束事件 - 优化：减少冗余处理"""
        event_name = event.get("name", "")
        output_data = event.get("data", {}).get("output", {})
        
        logger.debug(f"🏁 链式调用结束: {event_name}")
        
        # 🚀 优化：只处理关键节点的结束事件
        if event_name == "planner":
            # 检查是否有下一步动作
            has_next_action = bool(getattr(current_state, 'agent_outcome', None))
            if hasattr(current_state, 'agent_outcome') and isinstance(current_state.agent_outcome, list):
                has_next_action = len(current_state.agent_outcome) > 0
                
            await self._emit_event(self.current_event_builder.plan_decision({
                "planning_completed": True,
                "has_next_action": has_next_action,
            }))
            
        elif event_name == "summarize":
            # 从输出中获取总结
            summary = "总结完成"
            if isinstance(output_data, dict):
                summary = output_data.get("output", summary)
                if hasattr(output_data, "agent_outcome") and hasattr(output_data.agent_outcome, "return_values"):
                    summary = output_data.agent_outcome.return_values.get("output", summary)
            
            await self._emit_event(self.current_event_builder.summarize_finished(summary=summary))
        
        # 更新状态
        if isinstance(output_data, dict) and hasattr(current_state, '__dict__'):
            for key, value in output_data.items():
                if hasattr(current_state, key):
                    setattr(current_state, key, value)
        
        return current_state
    
    # ===== 旧方法已移除 - 现在使用标准LangGraph事件 =====
    
    async def _emit_event(self, event: Event):
        """发送事件并记录日志"""
        if self.current_sink:
            await self.current_sink.emit(event)
            logger.debug(f"📤 实时事件: {event.type.value}")


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


# 导出
__all__ = [
    'EventDrivenMasterAgent',
    'EventDrivenMasterAgentExecutor', 
    'create_event_driven_master_agent'
]
