"""
DataFlow Master Agent
基于 MyScaleKB-Agent 架构的主控智能体 - 使用真正的LangGraph工作流
"""
import logging
import asyncio
import time
import uuid
import json
from typing import Dict, List, Any, Union, Optional, Tuple, TypedDict, Annotated
from pydantic import BaseModel
from enum import Enum
import operator

# LangGraph核心组件
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import StructuredTool
from langchain_core.agents import AgentFinish as LCAgentFinish, AgentAction as LCAgentAction

# from dataflow.agent_v2.base.core import SubAgent, GraphBuilder, BaseTool, node, edge, conditional_entry

# 使用 myscalekb_agent_base 库的组件
from myscalekb_agent_base.sub_agent import SubAgent
from myscalekb_agent_base.graph_builder import GraphBuilder, node, edge, conditional_entry
from myscalekb_agent_base.schemas.agent_metadata import AgentMetadata

# 保留自己的组件
from dataflow.agent_v2.base.core import BaseTool

# 导入事件系统
from ..events.core import EventSink, Event, EventType

from dataflow.agent_v2.llm_client import get_llm_client
from dataflow.agent_v2.subagents.apikey_agent import APIKeyTool
from dataflow.agent_v2.subagents.mock_tools import SleepTool
from dataflow.agent_v2.subagents.csvtools import CSVProfileTool, CSVDetectTimeColumnsTool, CSVVegaSpecTool, ASTStaticCheckTool, UnitTestStubTool, LocalIndexBuildTool, LocalIndexQueryTool
from dataflow.agent_v2.former.former_tool import FormerTool
from dataflow.agent_v2.subagents.code_workflow_tool import CodeWorkflowTool
from dataflow.agent_v2.subagents.pipeline_workflow_tool import PipelineWorkflowTool
from dataflow.agent_v2.subagents.continue_chat_tool import ContinueChatTool

from concurrent.futures import ThreadPoolExecutor

def to_langchain_tool(tool: BaseTool) -> StructuredTool:
    ArgsSchema = tool.params()  # 你的工具已经提供了 Pydantic 参数类

    async def _arun(**kwargs):
        # 对于Former工具，特殊处理参数转换
        if tool.name() == "former":
            from dataflow.agent_v2.former.former_tool import FormerToolParams
            params = FormerToolParams(**kwargs)
            return tool.execute(params)  # FormerTool是同步的
        else:
            # 其他工具正常处理
            return await tool.execute(**kwargs)

    return StructuredTool.from_function(
        coroutine=_arun,                      # 异步函数
        name=tool.name(),
        description=tool.description(),
        args_schema=ArgsSchema,               # 参数校验
        return_direct=False,                  # 常规情况 False；需要时可 True
    )

logger = logging.getLogger(__name__)


# 使用 myscalekb_agent_base 兼容的 AgentState 结构
class AgentState(TypedDict, total=False):
    """Master Agent 状态定义 - 兼容 myscalekb_agent_base 结构"""
    # myscalekb_agent_base 标准字段
    input: Any  # 输入消息 (可以是字符串或 UserMessage)
    query: str  # 转换后的查询字符串
    chat_history: List[Any]  # 聊天历史
    agent_metadata: AgentMetadata  # 代理元数据
    agent_outcome: Union[Any, None]  # 代理输出
    intermediate_steps: Annotated[List[Tuple[Any, Any]], operator.add]  # 中间步骤
    trace_id: Union[str, None]  # 追踪ID
    
    # DataFlow 扩展字段 
    session_id: Optional[str]
    current_step: str
    form_data: Optional[Dict[str, Any]]
    form_session: Optional[Dict[str, Any]]  # FormerTool表单会话状态，统一存储到Master Agent
    xml_content: Optional[str]
    
    execution_result: Optional[str]
    conversation_history: List[Dict[str, str]]  # 对话历史
    last_tool_results: Optional[Dict[str, Any]]  # 最近的工具结果
    
    # 多轮编排支持
    pending_actions: List[Any]  # 待执行的动作
    tool_results: List[Dict[str, Any]]  # 结构化工具结果
    loop_guard: int  # 循环计数器
    max_steps: int  # 最大步数
    context_vars: Dict[str, Any]  # 跨步共享数据
    next_action: Optional[str]  # 下一个动作决策


class MasterAgent(SubAgent):
    """DataFlow Master Agent - 基于 MyScaleKB-Agent 风格的 LangGraph 架构"""
    
    def __init__(self, ctx=None, llm=None, memory=None, *args, **kwargs):
        # 🔧 修复：如果llm为None，创建默认的LLM客户端
        if llm is None:
            class MockLLM:
                def __init__(self):
                    self.model = get_llm_client()
            llm = MockLLM()
        
        # 确保self.llm在super().__init__之前被设置
        self.llm = llm
        
        try:
            super().__init__(ctx, llm, memory, *args, **kwargs)
        except Exception as e:
            logger.error(f"❌ SubAgent初始化失败: {e}")
            # 如果SubAgent初始化失败，我们手动设置必要的属性
            self.ctx = ctx
            self.memory = memory

        self.forward_paths = {}
        self.sub_agents = {}
        self.conversation_sessions = {}  # 会话管理
        self.tools = []

        # 注册工具
        self._register_tools()
    
    @classmethod
    def name(cls) -> str:
        return "master_agent"
    
    @classmethod
    def description(cls) -> str:
        return "DataFlow主控智能体，支持多轮编排和工具调用，可以处理复杂的用户请求"
    
    def _register_tools(self):
        """注册工具"""
        try:
            # 导入Former工具（从former文件夹）
            self.tools = [
                APIKeyTool(),
                # 主要工作流工具
                FormerTool(),
                CodeWorkflowTool(),
                PipelineWorkflowTool(),
                # 其他Mock工具用于测试多轮编排
                SleepTool(),
                CSVProfileTool(), 
                CSVDetectTimeColumnsTool(), 
                CSVVegaSpecTool(), 
                ASTStaticCheckTool(), 
                UnitTestStubTool(), 
                LocalIndexBuildTool(), 
                LocalIndexQueryTool()
            ]
            
            logger.info(f"已注册 {len(self.tools)} 个可直接调用的工具")
            
        except Exception as e:
            logger.error(f"工具注册失败: {e}")
            self.tools = []
        
        # 确保 lc_tools 总是被设置
        self.lc_tools = [to_langchain_tool(t) for t in (self.tools or [])]
        
        # 初始化工具执行器
        try:
            self.tool_executor = ToolExecutor(self.lc_tools)
        except Exception as e:
            logger.error(f"ToolExecutor初始化失败: {e}")
            self.tool_executor = None

    def build_app(self):
        """构建代理工作流 - 类似 MyScaleKB-Agent 的实现"""
        workflow = self._build_graph(AgentState, compiled=False)
        
        # 设置条件入口点 - 直接进入planner，统一决策逻辑
        workflow.set_conditional_entry_point(
            self.entry,
            {
                "planner": "planner",
            }
        )
        
        # 添加条件边
        workflow.add_conditional_edges(
            "planner",
            self.planner_router,
            {
                "continue": "execute_tools",
                "finish": "summarize"
            }
        )
        
        # 🔥 关键修复：execute_tools 使用条件边而不是固定边
        workflow.add_conditional_edges(
            "execute_tools",
            self.action_forward,
            {
                "planner": "planner",
                "summarize": "summarize", 
                "end": GraphBuilder.END
            }
        )
        
        workflow.add_edge("summarize", GraphBuilder.END)
        
        return workflow.compile()
    
    @staticmethod
    async def entry(data):
        """入口点 - 直接路由到planner进行统一决策"""
        logger.info("🚪 进入Master Agent入口点，直接路由到planner")
        return "planner"
    
    @node
    async def execute_tools(self, data):
        """执行工具节点 - 确保每次只执行一个动作"""
        agent_outcome = data.get("agent_outcome")
        
        logger.info(f"🛠️ 进入execute_tools_node，agent_outcome类型: {type(agent_outcome)}")
        logger.info(f"🛠️ agent_outcome内容: {agent_outcome}")

        actions: List[LCAgentAction] = []
        if isinstance(agent_outcome, list):
            # 🔧 重要：即使agent_outcome是列表，也只取第一个动作
            if agent_outcome:
                actions = [agent_outcome[0]]  # 只取第一个
                logger.info(f"📋 从列表中取第一个动作: {actions[0].tool}")
            else:
                logger.warning("⚠️ agent_outcome是空列表")
                return data
        elif hasattr(agent_outcome, "tool"):
            actions = [agent_outcome]
            logger.info(f"📋 单个动作: {agent_outcome.tool}")
        else:
            logger.warning(f"⚠️ 没有可执行的动作，agent_outcome类型: {type(agent_outcome)}")
            return data

        if not data.get("tool_results"):
            data["tool_results"] = []

        # 🔧 关键：只执行这一个动作
        if actions:
            action = actions[0]
            logger.info(f"🛠️ 开始执行单个工具: {action.tool}")
            try:
                # 🔥 修复：特殊处理former工具 - 直接传递表单数据
                if action.tool == "former":
                    # 🔥 新逻辑：直接传递form_data参数
                    current_form_session = data.get("form_session")
                    if current_form_session and current_form_session.get("form_data"):
                        # 提取表单数据，只传递fields部分
                        form_data = current_form_session["form_data"].get("fields", {})
                        action.tool_input["form_data"] = form_data
                        logger.info(f"🔄 传递表单数据给Former工具: {list(form_data.keys())}")
                        logger.info(f"🔄 表单字段值: {form_data}")
                    else:
                        logger.info(f"🔄 没有现有表单数据，Former工具将从空白开始")
                        action.tool_input["form_data"] = {}
                
                # 统一使用tool_executor处理所有工具调用，确保触发LangGraph事件
                result = await self.tool_executor.ainvoke(action)
                
                # 特殊处理former工具的会话状态和跳转指令
                if action.tool == "former" and isinstance(result, dict):
                    # 更新 form_session 到 AgentState
                    if result.get("session_id"):
                        data["form_session"] = {
                            "session_id": result["session_id"],
                            "form_data": {"fields": result.get("form_data", {})},  # 包装在fields中以保持兼容性
                            "form_stage": result.get("form_stage"),
                            "requires_user_input": result.get("requires_user_input", True),
                            "target_workflow": result.get("target_workflow", "")  # 🔥 保存目标工作流
                        }
                    
                    # 🔥 关键修复：如果former需要等待用户输入，直接结束流程
                    if result.get("requires_user_input") is True:
                        logger.info("🛑 Former工具需要等待用户输入，直接结束流程")
                        
                        # 记录到intermediate_steps - 在提前返回之前保存
                        if not data.get("intermediate_steps"):
                            data["intermediate_steps"] = []
                        data["intermediate_steps"].append((action, result))
                        
                        # 记录到tool_results
                        data["tool_results"].append({
                            "tool": action.tool,
                            "ok": bool(result.get("success", True)) if isinstance(result, dict) else True,
                            "payload": result
                        })
                        
                        # 🔥 新增：立即同步到全局状态 (前置到提前返回之前)
                        try:
                            from ..websocket.server import global_agent_states
                            # 尝试从多个位置获取session_id
                            session_id = data.get('session_id') or getattr(data, 'session_id', None)
                            if not session_id:
                                # 从agent_metadata获取
                                agent_metadata = data.get('agent_metadata')
                                if agent_metadata and hasattr(agent_metadata, 'session_id'):
                                    session_id = agent_metadata.session_id
                                elif agent_metadata and isinstance(agent_metadata, dict):
                                    session_id = agent_metadata.get('session_id')
                            
                            logger.info(f"🔍 尝试同步(former等待输入)，session_id: {session_id}")
                            
                            if session_id and session_id in global_agent_states:
                                # 更新global_agent_states中的tool_results
                                global_agent_states[session_id]["tool_results"] = data["tool_results"]
                                if "form_session" in data:
                                    global_agent_states[session_id]["form_session"] = data["form_session"]
                                logger.info(f"🔄 已同步AgentState到全局状态(former等待): {session_id}")
                            else:
                                logger.warning(f"⚠️ 无法同步: session_id={session_id}, keys={list(global_agent_states.keys()) if global_agent_states else 'None'}")
                        except Exception as sync_error:
                            logger.warning(f"⚠️ 同步到全局状态失败: {sync_error}")
                            import traceback
                            logger.warning(traceback.format_exc())
                        
                        # 使用former的输出作为最终结果
                        data["final_result"] = result.get("message", "等待用户进一步输入")
                        data["agent_outcome"] = []  # 清空，表示结束
                        data["next_action"] = "finish"
                        return data
                
                # 记录到intermediate_steps
                if not data.get("intermediate_steps"):
                    data["intermediate_steps"] = []
                data["intermediate_steps"].append((action, result))
                
                # 记录到tool_results
                data["tool_results"].append({
                    "tool": action.tool,
                    "ok": bool(result.get("success", True)) if isinstance(result, dict) else True,
                    "payload": result
                })
                
                # 🔥 新增：立即同步到全局状态 (用于WebSocket前端显示)
                try:
                    from ..websocket.server import global_agent_states
                    # 尝试从多个位置获取session_id
                    session_id = data.get('session_id') or getattr(data, 'session_id', None)
                    if not session_id:
                        # 从agent_metadata获取
                        agent_metadata = data.get('agent_metadata')
                        if agent_metadata and hasattr(agent_metadata, 'session_id'):
                            session_id = agent_metadata.session_id
                        elif agent_metadata and isinstance(agent_metadata, dict):
                            session_id = agent_metadata.get('session_id')
                    
                    logger.info(f"🔍 尝试同步，session_id: {session_id}")
                    
                    if session_id and session_id in global_agent_states:
                        # 更新global_agent_states中的tool_results
                        global_agent_states[session_id]["tool_results"] = data["tool_results"]
                        if "form_session" in data:
                            global_agent_states[session_id]["form_session"] = data["form_session"]
                        logger.info(f"🔄 已同步AgentState到全局状态: {session_id}")
                except Exception as sync_error:
                    logger.warning(f"⚠️ 同步到全局状态失败: {sync_error}")
                    import traceback
                    logger.warning(traceback.format_exc())
                
                # 🗑️ 新增：如果执行的是非former工具，清除表单状态和final_result
                if action.tool != "former":
                    if data.get("form_session"):
                        logger.info(f"🗑️ 工作流工具 {action.tool} 执行完成，清除表单状态")
                        data["form_session"] = None
                    
                    if data.get("final_result"):
                        logger.info(f"🗑️ 工作流工具 {action.tool} 执行完成，清除final_result")
                        data["final_result"] = None
                    
                    # 同时清除全局状态中的表单
                    try:
                        # 获取session_id
                        session_id = data.get('session_id')
                        if not session_id:
                            agent_metadata = data.get('agent_metadata')
                            if agent_metadata and hasattr(agent_metadata, 'session_id'):
                                session_id = agent_metadata.session_id
                            elif agent_metadata and isinstance(agent_metadata, dict):
                                session_id = agent_metadata.get('session_id')
                        
                        if session_id and session_id in global_agent_states:
                            if "form_session" in global_agent_states[session_id]:
                                del global_agent_states[session_id]["form_session"]
                                logger.info(f"🗑️ 已清除全局状态中的表单: {session_id}")
                    except Exception as clear_error:
                        logger.warning(f"⚠️ 清除全局表单状态失败: {clear_error}")
                
            except Exception as e:
                logger.error(f"❌ 工具执行失败: {action.tool}, 错误: {e}")
                import traceback
                logger.error(f"❌ 详细错误信息:\n{traceback.format_exc()}")
                err = {"success": False, "error": str(e)}
                if not data.get("intermediate_steps"):
                    data["intermediate_steps"] = []
                data["intermediate_steps"].append((action, err))
                data["tool_results"].append({"tool": action.tool, "ok": False, "payload": err})
        # 清空agent_outcome，避免重复执行
        data["agent_outcome"] = []
        logger.info(f"🔄 工具执行完成，清空agent_outcome，进入planner节点")
        return data
    
    @node
    async def planner(self, data: AgentState) -> AgentState:
        """规划器节点 - 每次只规划下一个单独动作"""
        from langchain_core.agents import AgentAction as LCAgentAction
        
        # ✅ 步数护栏
        if data.get('loop_guard') is None:
            data["loop_guard"] = 0
        data["loop_guard"] += 1
        
        max_steps = data.get('max_steps', 20)
        if data["loop_guard"] >= max_steps:
            # 触发护栏直接结束
            logger.info(f"🛑 达到最大步骤数 {max_steps}，自动结束")
            data["agent_outcome"] = []  # 清空，让summarize处理
            data["next_action"] = "finish"
            return data
            
        logger.info(f"🎯 进入规划器节点 - 步骤 {data['loop_guard']}/{max_steps}")

        # 简化的上下文信息
        user_input = data.get('input', '')
        tool_results_count = len(data.get('tool_results', []))
        form_session = data.get('form_session')
        has_form_session = bool(form_session)
        
        logger.debug(f"📝 简化上下文: 用户输入='{user_input[:50]}...', 工具执行次数={tool_results_count}, 表单会话={has_form_session}")
        
        # 🔥 新增：优先检查是否存在正在进行的表单收集
        if form_session:
            # 检查表单是否需要用户输入
            requires_user_input = form_session.get('requires_user_input', False)
            form_stage = form_session.get('form_stage', '')
            
            if requires_user_input and form_stage == 'parameter_collection':
                logger.info(f"🎯 检测到正在进行的表单收集，继续使用former工具处理用户输入")
                
                # 直接创建former工具动作，跳过LLM决策
                single_action = LCAgentAction(
                    tool="former",
                    tool_input={
                        "user_query": user_input,
                        "action": "collect_user_response",
                        "session_id": form_session.get('session_id'),
                        "form_data": form_session.get('form_data', {})
                    },
                    log="继续表单收集: 处理用户补充信息"
                )
                data["agent_outcome"] = [single_action]
                data["next_action"] = "continue"
                
                logger.info(f"📋 继续表单收集，处理用户输入")
                return data
        
        # � 优先检查最近工具的后置建议
        tool_results = data.get("tool_results", [])
        if tool_results:
            last_result = tool_results[-1]  # 获取最后一个工具结果
            payload = last_result.get("payload", {})
            
            # 检查是否有后置工具建议
            if "followup_recommendation" in payload:
                rec = payload["followup_recommendation"]
                if isinstance(rec, dict) and rec.get("needs_followup"):
                    suggested_tool = rec.get("tool_name", "")
                    reason = rec.get("reason", "")
                    
                    # 检查表单完整性
                    form_complete = payload.get("form_complete", True)
                    
                    if not form_complete and suggested_tool:
                        logger.info(f"🎯 检测到表单不完整，直接采用工具建议: {suggested_tool}")
                        logger.info(f"📋 建议原因: {reason}")
                        
                        # 直接创建后置工具动作，跳过LLM决策
                        tool_input = {}
                        if suggested_tool == "continue_chat":
                            # 构建Former Tool的会话上下文
                            session_context = rec.get("session_context", {})
                            tool_input = {
                                "prompt": "请继续表单对话",
                                "context": json.dumps(session_context) if session_context else f"当前需求: {data.get('input', '')}"
                            }
                        
                        single_action = LCAgentAction(
                            tool=suggested_tool,
                            tool_input=tool_input,
                            log=f"工具建议: {suggested_tool}"
                        )
                        
                        data["agent_outcome"] = [single_action]
                        data["next_action"] = "continue"
                        
                        logger.info(f"📋 采用工具建议: {suggested_tool}")
                        return data
        
        # 🔥 新增：检查表单是否已完成并需要执行目标工作流
        if form_session:
            form_complete = False
            target_workflow = form_session.get("target_workflow", "")
            
            # 检查最近的former工具结果
            if tool_results:
                last_result = tool_results[-1]
                if last_result.get("tool") == "former":
                    payload = last_result.get("payload", {})
                    form_complete = payload.get("form_complete", False)
            
            if form_complete and target_workflow:
                logger.info(f"🎯 检测到表单已完成，目标工作流: {target_workflow}")
                
                # 从表单数据中提取参数
                form_data = form_session.get("form_data", {})
                if isinstance(form_data, dict) and "fields" in form_data:
                    form_fields = form_data["fields"]
                else:
                    form_fields = form_data
                
                logger.info(f"📋 使用表单数据构建工具参数: {list(form_fields.keys()) if form_fields else 'None'}")
                
                # 直接创建目标工具动作，使用表单数据
                single_action = LCAgentAction(
                    tool=target_workflow,
                    tool_input=form_fields or {},
                    log=f"表单收集完成，执行目标工作流: {target_workflow}"
                )
                
                data["agent_outcome"] = [single_action]
                data["next_action"] = "continue"
                
                logger.info(f"📋 Planner规划下一个动作: {target_workflow} (表单数据)")
                logger.info(f"📋 原因: 表单收集完成，使用收集的参数执行目标工作流")
                
                return data

        try:
            analysis = self._analyze_user_needs(data.get("input", ""), data.get("tool_results", []))
            logger.info(f"📋 需求分析: {analysis}")
            
            if analysis["should_continue"] and analysis["next_action"]:
                # 🔧 重要：只创建一个动作
                next_action = analysis["next_action"]
                
                # 🔥 新增：提取LLM的决策原因
                llm_reasoning = ""
                if "analysis" in analysis and "llm_decision" in analysis["analysis"]:
                    llm_reasoning = analysis["analysis"]["llm_decision"].get("reason", "")
                
                single_action = LCAgentAction(
                    tool=next_action.get("tool", ""),
                    tool_input=next_action.get("tool_input", {}),
                    log=llm_reasoning or f"Planner规划: {next_action.get('tool','')}"
                )
                
                data["agent_outcome"] = [single_action]  # 注意：只有一个动作
                data["next_action"] = "continue"
                
                logger.info(f"📋 Planner规划下一个动作: {next_action.get('tool', '')} (单次)")
                logger.info(f"📋 原因: {'; '.join(analysis['reasons']) if isinstance(analysis['reasons'], list) else analysis['reasons']}")
            else:
                # 🔧 修复：流转到summarize节点进行智能总结
                data["agent_outcome"] = []  # 清空，让summarize_node处理
                data["next_action"] = "finish"
                logger.info(f"🏁 Planner决定结束，流转到summarize节点")
            return data
            
        except Exception as e:
            logger.error(f"规划器错误: {e}")
            # 🔧 修复：异常时也流转到summarize节点
            data["agent_outcome"] = []
            data["next_action"] = "finish"
            data["error_message"] = f"规划错误: {str(e)}"
            return data

    def planner_router(self, data: AgentState) -> str:
        """规划器路由器 - 修复版本，兜底返回finish"""
        next_action = data.get("next_action")
        result = "continue" if next_action == "continue" else "finish"
        logger.info(f"� 路由决策: {next_action} -> {result}")
        return result
    
    @node
    @edge(target_node=GraphBuilder.END)
    async def summarize(self, data: AgentState) -> AgentState:
        """总结节点 - 处理工具执行结果总结或通用对话回复"""
        logger.info(f"📝 总结节点开始，intermediate_steps数量: {len(data.get('intermediate_steps', []))}")
        
        # 🔥 关键修复：如果有final_result（来自former等待用户输入），直接使用
        if data.get("final_result"):
            logger.info("📝 检测到final_result，直接使用former的输出")
            finish = LCAgentFinish(
                return_values={"output": data.get("final_result")},
                log="Former直接输出"
            )
            data["agent_outcome"] = finish
            return data
        
        # 🔥 新增：检查是否有former工具的输出，只有在需要用户输入时才直接使用
        former_output = None
        former_requires_input = False
        
        for action, result in data.get("intermediate_steps", []):
            if action.tool == "former" and isinstance(result, dict):
                former_output = result.get("message")
                former_requires_input = result.get("requires_user_input", False)
                if former_output and former_requires_input:
                    logger.info("📝 检测到former需要用户输入，直接使用former输出")
                    finish = LCAgentFinish(
                        return_values={"output": former_output},
                        log="Former工具等待用户输入"
                    )
                    data["agent_outcome"] = finish
                    return data
        
        # 检查是否有错误消息
        if data.get('error_message'):

            finish = LCAgentFinish(
                return_values={"output": data.get('error_message')},
                log="错误总结"
            )

            data["agent_outcome"] = finish
            return data
        
        # 如果已经是最终结果，直接返回
        agent_outcome = data.get('agent_outcome')
        if hasattr(agent_outcome, 'return_values'):
            # 已经是最终结果，直接返回
            logger.info("📝 检测到已有return_values，直接返回")
            return data
        
        final_output = await self._generate_conversation_response(data)

        finish = LCAgentFinish(
            return_values={"output": final_output},
            log="智能总结完成"
        )

        data["agent_outcome"] = finish
        
        return data
    
    async def action_forward(self, data: AgentState) -> str:
        """决定下一步动作 - 修复版本，正确处理工具执行后的路由"""
        agent_outcome = data.get('agent_outcome')
        logger.info(f"🔀 Action Forward开始，agent_outcome类型: {type(agent_outcome)}")
        logger.info(f"🔀 Agent outcome内容: {agent_outcome}")
        
        # 检查是否是结束状态
        if hasattr(agent_outcome, 'return_values'):
            logger.info("📝 检测到return_values，结束流程")
            return "end"

        # 🔧 关键修复：检查是否达到最大步数或有next_action标志
        if data.get('next_action') == "finish":
            logger.info("🏁 检测到finish标志，进入总结阶段")
            return "summarize"
        
        # 🔧 如果agent_outcome为空列表，根据上下文判断
        if isinstance(agent_outcome, list) and len(agent_outcome) == 0:
            # 检查是否有loop_guard（表示在planner中达到最大步数）
            if data.get('loop_guard', 0) >= data.get('max_steps', 8):
                logger.info("🛑 达到最大步数，进入总结阶段")
                return "end"
            else:
                logger.info("🔄 工具执行完成，回到planner继续决策")
                return "planner"  # 回到planner而不是execute_tools
        
        # 获取agent_action - 直接使用agent_outcome或从列表中取第一个
        if isinstance(agent_outcome, list):
            agent_action = agent_outcome[0] if agent_outcome else None
            logger.info(f"🎬 从列表获取agent_action: {agent_action}")
        else:
            agent_action = agent_outcome
            logger.info(f"🎬 直接获取agent_action: {agent_action}")
        
        if agent_action:
            # 检查LangGraph模式下的工具
            if hasattr(agent_action, 'tool'):
                tool_name = agent_action.tool
                logger.info(f"🔧 LangGraph模式 - 工具名: {tool_name}")
                # 🔧 简化：移除general_conversation，end工具直接到summarize
                if tool_name == "end":
                    logger.info("🏁 路由到: end (summarize)")
                    return "end"
                else:
                    logger.info(f"🛠️ 路由到: execute_tools (工具: {tool_name})")
                    return "execute_tools"
            
        logger.info("⚠️ 无匹配条件，默认路由到end")
        return "end"

    def _build_history_text(self, conversation_history: List[Dict[str, str]], k: int = 8, clip: int = 200) -> str:
        """把最近 k 条历史拼成统一文本；长消息裁剪到 clip 字符。"""
        if not conversation_history:
            return ""

        recent = conversation_history[-k:]
        lines = []
        for msg in recent:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            if len(content) > clip:
                content = content[:clip] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    
    async def _generate_conversation_response(self, data: AgentState) -> str:
        """基于工具执行结果和对话历史生成智能响应 - 让大模型对所有工具结果进行智能总结"""
        user_input = data.get("input", "")
        conversation_history = data.get("conversation_history", [])
        
        # 构建详细的工具执行结果 - 通用化处理，不硬编码特定字段
        detailed_tool_results = []
        tool_output_summary = {}  # 按工具类型汇总输出
        
        for i, (action, result) in enumerate(data.get("intermediate_steps", [])):
            tool_name = action.tool
            step_num = i + 1
            
            # 解析工具结果并收集详细信息
            if isinstance(result, dict):
                detailed_tool_results.append({
                    "step": step_num,
                    "tool": tool_name,
                    "result": result
                })
                
                # 通用化地收集每个工具的输出
                if tool_name not in tool_output_summary:
                    tool_output_summary[tool_name] = []
                
                # 动态提取结果中的重要信息
                important_fields = []
                for key, value in result.items():
                    # 特殊处理代码类字段，不限制长度
                    if key in ['current_code', 'generated_code', 'code', 'output']:
                        if isinstance(value, str) and value.strip():
                            important_fields.append(f"{key}: {value}")
                    # 其他字段保持原有限制
                    elif isinstance(value, (str, int, float, bool)) and len(str(value)) < 200:
                        important_fields.append(f"{key}: {value}")
                
                # 判断执行状态 - 更智能的状态判断
                is_success = (
                    result.get("success") is True or 
                    result.get("access_granted") is True or
                    result.get("ok") is True or
                    result.get("status") == "completed"
                )
                
                tool_output_summary[tool_name].append({
                    "step": step_num,
                    "status": "成功" if is_success else "失败",
                    "details": important_fields
                })
                    
            else:
                # 如果不是字典，也要记录
                detailed_tool_results.append({
                    "step": step_num,
                    "tool": tool_name,
                    "result": {"raw_output": str(result)}
                })
                
                if tool_name not in tool_output_summary:
                    tool_output_summary[tool_name] = []
                tool_output_summary[tool_name].append({
                    "step": step_num,
                    "status": "完成",
                    "details": [f"输出: {str(result)[:50]}"]
                })
        
        # 如果LLM不可用，使用增强的格式化输出
        if not self.llm.api_available:
            Exception("LLM服务不可用，无法执行任何工具或SubAgent")
        
        try:
            # 构建简洁实用的提示词
            system_prompt = """你是DataFlow智能助手。基于工具执行结果简洁回复用户。

要求：
0.如果没有工具执行结果，正常回复用户输入
1. 直接回答用户的问题，展示工具返回的核心结果
2. 如果工具生成了代码，直接展示代码
3. 如果工具返回了数据，直接提供数据
4. 语言要简洁自然，不要冗长的分析说明
5. 所有内容基于实际工具输出，不编造信息
"""

            # 构建对话历史文本
            history_text = self._build_history_text(conversation_history, k=10, clip=300)
            
            # 构建详细的工具执行报告
            try:
                execution_report = self._build_detailed_execution_report(detailed_tool_results, tool_output_summary)
            except Exception as report_error:
                logger.error(f"构建执行报告失败: {report_error}")
                execution_report = f"执行报告构建失败: {str(report_error)}"
            
            user_prompt = f"""用户请求: {user_input}
执行过程详细报告:
{execution_report}
对话历史:
{history_text}
"""

            logger.info(f"🚀 准备调用LLM，user_prompt长度: {len(user_prompt)}")
            
            # 调用LLM生成智能总结 - 增加超时控制
            try:
                
                
                def sync_llm_call():
                    try:
                        llm_service = self.llm._create_llm_service()
                        # 减少重试次数避免长时间阻塞
                        llm_service.max_retries = 1
                        return llm_service.generate_from_input(
                            user_inputs=[user_prompt],
                            system_prompt=system_prompt
                        )
                    except Exception as e:
                        logger.error(f"LLM服务内部错误: {e}")
                        return None
                
                # 异步执行，设置5秒超时
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(sync_llm_call)
                    try:
                        responses = await asyncio.wait_for(
                            asyncio.wrap_future(future), 
                            timeout=50.0
                        )
                        
                        if responses and responses[0]:
                            llm_response = responses[0].strip()
                            logger.info(f"🤖 LLM智能总结生成成功: {llm_response[:100]}...")
                            return llm_response
                        else:
                            logger.warning("⚠️ LLM返回空响应，使用fallback")
                            
                    except asyncio.TimeoutError:
                        logger.warning("⚠️ LLM调用超时（5秒），使用fallback")
                    except Exception as e:
                        logger.error(f"LLM异步调用错误: {e}")
                        
            except Exception as e:
                logger.error(f"LLM智能总结调用失败: {e}")
                import traceback
                logger.error(f"详细错误: {traceback.format_exc()}")
            
        except Exception as e:
            logger.error(f"LLM智能总结生成失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            # 直接抛出异常，不使用fallback
            raise e
      
    def _build_detailed_execution_report(self, detailed_tool_results: List[Dict], tool_output_summary: Dict) -> str:
        """构建详细的执行报告供LLM分析 - 精简版本，只显示关键的执行结果"""
        report_sections = []
        
        # 执行概况
        total_steps = len(detailed_tool_results)
        report_sections.append(f"执行概况: 总共{total_steps}个步骤")
        
        # 完整执行时序和结果 - 显示完整的工具返回数据
        report_sections.append(f"\n执行结果详情:")
        for tool_result in detailed_tool_results:
            step = tool_result["step"]
            tool = tool_result["tool"]
            result = tool_result["result"]
            
            # 显示完整的工具返回数据
            if isinstance(result, dict):
                # 更智能的状态判断
                is_success = (
                    result.get("success") is True or 
                    result.get("access_granted") is True or
                    result.get("ok") is True or
                    result.get("status") == "completed"
                )
                status = "成功" if is_success else "完成"
                
                # 显示完整的关键字段，特别是apikey
                key_info = []
                priority_fields = ["apikey", "result", "message"]  # 优先显示的字段
                
                # 先添加优先字段
                for field in priority_fields:
                    if field in result:
                        value = result[field]
                        if isinstance(value, (str, int, float, bool)):
                            key_info.append(f"{field}: {value}")
                
                # 再添加其他字段
                for key, value in result.items():
                    if key not in priority_fields and isinstance(value, (str, int, float, bool)) and len(str(value)) < 100:  # 只显示合理长度的字段
                        key_info.append(f"{key}: {value}")
                
                if key_info:
                    # 显示所有重要字段，不截断
                    info_text = "\n    ".join(key_info)
                    report_sections.append(f"  步骤{step}: [{tool}] {status}")
                    report_sections.append(f"    {info_text}")
                else:
                    report_sections.append(f"  步骤{step}: [{tool}] {status}")
            else:
                report_sections.append(f"  步骤{step}: [{tool}] 完成 → {str(result)[:100]}")
        
        return "\n".join(report_sections)
    
    def _analyze_user_needs(self, user_input: str, tool_results: List[Dict]) -> Dict[str, Any]:
        """让LLM智能分析用户需求和当前执行状态，决定下一步行动 - 完全基于LLM决策"""
        
        # 🔧 核心修复：让LLM来理解和决策
        if not self.llm.api_available:
            # LLM不可用时的简单fallback
            return {
                "should_continue": False,
                "next_action": None,
                "reasons": ["LLM不可用，无法进行智能决策"],
                "analysis": {}
            }
        
        try:
            # 构建可用工具的详细描述，包含参数信息
            available_tools = []
            for tool in self.tools:
                tool_info = {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": self._get_tool_parameters(tool)
                }
                available_tools.append(tool_info)
            
            # 构建执行历史 - 通用化处理，不硬编码特定字段
            execution_history = []
            for i, result in enumerate(tool_results, 1):
                tool_name = result.get("tool", "unknown")
                success = result.get("ok", False)
                payload = result.get("payload", {})
                
                step_info = f"步骤{i}: 执行了{tool_name}"
                if success:
                    if isinstance(payload, dict):
                        # 判断具体执行状态
                        is_tool_success = (
                            payload.get("success") is True or 
                            payload.get("access_granted") is True or
                            payload.get("ok") is True or
                            payload.get("status") == "completed"
                        )
                        
                        if is_tool_success:
                            step_info += " - 成功"
                            
                            # 特别提取工具建议信息
                            tool_recommendations = []
                            if "followup_recommendation" in payload:
                                rec = payload["followup_recommendation"]
                                if isinstance(rec, dict) and rec.get("needs_followup"):
                                    tool_name = rec.get("tool_name", "")
                                    reason = rec.get("reason", "")
                                    tool_recommendations.append(f"推荐后置工具: {tool_name} (原因: {reason})")
                            
                            # 提取表单完整性信息
                            form_info = []
                            if "form_complete" in payload:
                                form_complete = payload["form_complete"]
                                if not form_complete:
                                    form_info.append("表单信息不完整")
                            
                            # 🔥 关键修复：检查是否需要等待用户输入
                            waiting_status = []
                            if payload.get("requires_user_input") is True:
                                waiting_status.append("等待用户输入")
                            
                            # 优先显示等待状态，然后是工具建议和表单状态
                            important_info = waiting_status + tool_recommendations + form_info
                            
                            # 添加其他重要字段
                            for key, value in payload.items():
                                if key not in ["followup_recommendation", "form_complete"] and isinstance(value, (str, int, float, bool)) and len(str(value)) < 50:
                                    important_info.append(f"{key}: {value}")
                            
                            if important_info:
                                info_text = ", ".join(important_info[:5])  # 显示前5个重要信息
                                step_info += f" ({info_text})"
                        else:
                            step_info += " - 失败"
                    else:
                        step_info += " - 完成"
                else:
                    step_info += " - 失败"
                
                execution_history.append(step_info)
            
            # 构建LLM决策提示词 - 严格JSON格式
            system_prompt = """你是一个智能决策助手，需要根据用户需求和当前执行情况，决定下一步应该执行什么工具。

决策原则：
1. **简单问候直接回应**：对于"你好"、"hi"等简单问候，直接完成任务，不需要使用任何工具
2. **明确任务才使用工具**：只有当用户明确提出具体任务需求时，才选择合适的工具
3. **避免重复调用**：如果某个工具已经执行过，除非有明确的理由，否则不要重复调用相同工具

**重要：你必须只输出JSON格式，不要有任何额外的解释文字！**

返回格式（必须是纯JSON，无任何其他内容）：
{
    "decision": "continue" 或 "finish",
    "tool": "工具名称（仅当decision为continue时）",
    "tool_input": {"参数名": "参数值"},
    "finish_message": "任务完成说明（仅当decision为finish时）",
    "reason": "简短的决策原因"
}"""

            user_prompt = f"""用户原始需求: {user_input}

可用工具列表:
{chr(10).join([f"- {tool['name']}: {tool['description']} | 参数: {tool['parameters']}" for tool in available_tools])}

当前执行历史:
{chr(10).join(execution_history) if execution_history else "还没有执行任何工具"}

请分析当前情况，决定下一步应该：

**重要：只输出JSON格式，不要任何解释文字！**"""

            logger.info(f"🤖 调用LLM进行智能决策...")
            
            # 调用LLM进行决策 - 使用基础调用方式
            llm_service = self.llm._create_llm_service()
            responses = llm_service.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt
            )
            
            if responses and responses[0]:
                content = responses[0].strip()
                logger.info(f"🤖 LLM决策响应: {content}")
                
                # 尝试清理JSON格式 - 移除可能的markdown代码块标记
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                # 解析LLM响应
                try:
                    import json
                    decision = json.loads(content)
                    
                    decision_type = decision.get("decision", "finish")
                    tool_name = decision.get("tool")
                    tool_input = decision.get("tool_input", {})
                    finish_message = decision.get("finish_message", "")
                    reasoning = decision.get("reason", "")
                    
                    # 判断是否继续
                    should_continue = (decision_type == "continue")
                    
                    # 构建next_action
                    next_action = None
                    if should_continue and tool_name and tool_input:
                        next_action = {
                            "tool": tool_name,
                            "tool_input": tool_input
                        }
                    elif should_continue and tool_name:
                        # 如果只有工具名没有参数，使用用户输入作为fallback
                        next_action = {
                            "tool": tool_name,
                            "tool_input": {"user_message": user_input}
                        }
                    
                    result = {
                        "should_continue": should_continue,
                        "next_action": next_action,
                        "reasons": [reasoning or finish_message],
                        "analysis": {
                            "decision_type": decision_type,
                            "llm_decision": decision,
                            "execution_count": len(tool_results),
                            "json_parsed": True
                        }
                    }
                    
                    logger.info(f"🎯 LLM决策结果: decision={decision_type}, tool={tool_name}")
                    logger.info(f"🎯 工具参数: {tool_input}")
                    logger.info(f"🎯 决策原因: {reasoning}")
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"LLM响应JSON解析失败: {e}")
                    logger.error(f"原始响应: {content}")
                    
                    return 
            
        except Exception as e:
            logger.error(f"LLM智能决策失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # 出错时的fallback
        return {
            "should_continue": False,
            "next_action": None,
            "reasons": ["智能决策失败，结束任务"],
            "analysis": {}
        }
    
    def _get_tool_parameters(self, tool) -> str:
        """获取工具的参数信息 - 动态解析工具参数，不使用硬编码"""
        try:
            # 方法1：尝试调用工具的参数方法
            if hasattr(tool, 'params'):
                params_class = tool.params()
                if hasattr(params_class, '__annotations__'):
                    # 使用__annotations__获取类型注解
                    annotations = params_class.__annotations__
                    param_info = []
                    for field_name, field_type in annotations.items():
                        type_name = getattr(field_type, '__name__', str(field_type))
                        param_info.append(f"{field_name}: {type_name}")
                    return "{" + ", ".join(param_info) + "}"
                
                elif hasattr(params_class, '__dict__'):
                    # 尝试从类字典获取信息
                    class_dict = params_class.__dict__
                    param_info = []
                    for key, value in class_dict.items():
                        if not key.startswith('_'):
                            param_info.append(f"{key}: auto_detected")
                    if param_info:
                        return "{" + ", ".join(param_info) + "}"
            
            # 方法2：尝试检查工具的args_schema
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema
                if hasattr(schema, '__annotations__'):
                    annotations = schema.__annotations__
                    param_info = []
                    for field_name, field_type in annotations.items():
                        type_name = getattr(field_type, '__name__', str(field_type))
                        param_info.append(f"{field_name}: {type_name}")
                    return "{" + ", ".join(param_info) + "}"
            
            # 方法3：尝试inspect工具的run方法
            if hasattr(tool, 'run'):
                import inspect
                sig = inspect.signature(tool.run)
                param_info = []
                for param_name, param in sig.parameters.items():
                    if param_name not in ['self', 'kwargs', 'args']:
                        type_hint = param.annotation if param.annotation != inspect.Parameter.empty else "auto"
                        type_name = getattr(type_hint, '__name__', str(type_hint))
                        param_info.append(f"{param_name}: {type_name}")
                if param_info:
                    return "{" + ", ".join(param_info) + "}"
            
            # 方法4：最后fallback - 基于工具描述推断
            description = tool.description().lower()
            if "path" in description or "file" in description:
                return '{"path": "string", "additional_params": "auto"}'
            elif "query" in description or "search" in description:
                return '{"query": "string", "additional_params": "auto"}'
            elif "seconds" in description or "time" in description:
                return '{"seconds": "number", "additional_params": "auto"}'
            else:
                return '{"user_message": "string(通用参数)"}'
                
        except Exception as e:
            logger.debug(f"动态解析工具{tool.name()}参数失败: {e}")
            # 最终fallback
            return '{"user_message": "string(通用参数)"}'
