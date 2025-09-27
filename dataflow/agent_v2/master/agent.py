"""
DataFlow Master Agent
基于 MyScaleKB-Agent 架构的主控智能体 - 使用真正的LangGraph工作流
"""
import logging
import asyncio
import os
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

# 统一工具参数构建
from .tool_input_builder import build_tool_input, create_unified_action

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
from dataflow.agent_v2.subagents.former_tool import FormerTool
from dataflow.agent_v2.subagents.code_workflow_tool import CodeWorkflowTool
from dataflow.agent_v2.subagents.pipeline_workflow_tool import PipelineWorkflowTool

from concurrent.futures import ThreadPoolExecutor

def to_langchain_tool(tool: BaseTool) -> StructuredTool:
    """将自定义工具对象转换为LangChain的StructuredTool"""
    ArgsSchema = tool.params()

    return StructuredTool.from_function(
        coroutine=tool.execute,  # 直接使用工具的异步execute方法
        name=tool.name(),
        description=tool.description(),
        args_schema=ArgsSchema,
        return_direct=False,
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
    form_session: Optional[Dict[str, Any]]  # 通用会话状态补丁，任何工具都可更新
    awaiting_user_message: Optional[str]  # 通用等待用户输入消息
    xml_content: Optional[str]
    
    # 前端标签页内容
    generated_code: Optional[str]  # 生成的代码内容，用于前端代码标签页
    code_metadata: Optional[Dict[str, Any]]  # 代码元数据（文件名、语言等）
    
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
            self.tools = [
                APIKeyTool(),
                FormerTool(),
                CodeWorkflowTool(),
                PipelineWorkflowTool(),
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
                # 🎯 统一工具参数构建 - 替代所有工具特判
                
                llm_args = action.tool_input or {}
                state_view = dict(data)  # 提供完整状态视图
                user_input = data.get("input", "")
                
                # 统一构建工具输入参数
                unified_input = build_tool_input(action.tool, llm_args, state_view, user_input)
                action.tool_input = unified_input  # 覆盖为最终输入
                
                # 统一使用tool_executor处理所有工具调用，确保触发LangGraph事件
                result = await self.tool_executor.ainvoke(action)
                

                
                # 记录到intermediate_steps
                if not data.get("intermediate_steps"):
                    data["intermediate_steps"] = []
                data["intermediate_steps"].append((action, result))
                
                # 🎯 统一工具结果适配（正式功能）
                try:
                    from dataflow.agent_v2.tool_result import adapt_tool_result, ToolStatus
                    unified_result = adapt_tool_result(action.tool, result)
                    logger.info(f"🔄 工具结果已适配: {action.tool} -> {unified_result.status}")
                    
                    # 存储统一格式结果
                    data.setdefault("unified_tool_results", []).append(unified_result.model_dump())
                    
                    # 记录到tool_results（参考unified_result.status）
                    data["tool_results"].append({
                        "tool": action.tool,
                        "ok": unified_result.status not in ["ERROR", "FAILED"],
                        "payload": result
                    })
                    
                    # ✅ 统一会话写入（不看工具名、不读原始 result.*）
                    session = getattr(unified_result, "session", None)
                    if session:
                        data["form_session"] = session
                        logger.info(f"🔍 工具{action.tool}更新会话状态: {session}")
                    # ✅ 需要用户输入 → 只设置等待消息，不再写 final_result
                    if unified_result.status == ToolStatus.NEED_USER_INPUT:
                        data["awaiting_user_message"] = getattr(unified_result, "message", None)
                        data["agent_outcome"] = []
                        data["next_action"] = "finish"
                        logger.info(f"🛑 工具 {action.tool} 需要等待用户输入，结束流程")
                        
                        # 🔥 关键修复：在返回前同步到全局状态（用于WebSocket前端显示）
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
                            
                            logger.info(f"🔍 等待用户输入时同步，session_id: {session_id}")
                            
                            if session_id and session_id in global_agent_states:
                                # 更新global_agent_states中的tool_results
                                global_agent_states[session_id]["tool_results"] = data["tool_results"]
                                if "form_session" in data:
                                    global_agent_states[session_id]["form_session"] = data["form_session"]
                                    logger.info(f"🔍 已同步form_session (等待输入)，missing_params数量: {len(data['form_session'].get('missing_params', []))}")
                                logger.info(f"🔄 等待用户输入时已同步AgentState到全局状态: {session_id}")
                            else:
                                logger.warning(f"⚠️ 等待用户输入时无法同步到全局状态: session_id={session_id}, exists={session_id in global_agent_states if session_id else False}")
                        except Exception as sync_error:
                            logger.warning(f"⚠️ 等待用户输入时同步到全局状态失败: {sync_error}")
                        
                        return data
                    # ✅ 非等待场景的会话清理（不依赖工具名）
                    should_close = False
                    if session and session.get("closed", False) is True:
                        should_close = True
                    elif (not session) and unified_result.status in ("SUCCESS", "COMPLETED") and data.get("form_session"):
                        should_close = True
                    if should_close:
                        data.pop("form_session", None)
                        logger.info(f"🗑️ 工具 {action.tool} 执行完成，清理会话状态")
                    # 统一清理等待提示
                    data.pop("awaiting_user_message", None)
                        
                except ImportError:
                    logger.debug("📝 统一工具结果适配器不可用，使用传统逻辑")
                except Exception as adapter_error:
                    logger.warning(f"⚠️ 工具结果适配失败: {adapter_error}")
                
                # 🎯 新增：提取和存储生成的代码到state中（用于前端标签页）
                # 检查多个可能的代码字段
                code_content = ""
                code_metadata = {}
                
                if isinstance(result, dict):
                    # 优先使用 frontend_code_data 结构
                    if result.get("frontend_code_data"):
                        code_data = result["frontend_code_data"]
                        code_content = code_data.get("code_content", "")
                        code_metadata = {
                            "file_name": code_data.get("file_name", "generated_code.py"),
                            "file_path": code_data.get("file_path", ""),
                            "language": code_data.get("language", "python"),
                            "tool_source": code_data.get("tool_source", action.tool),
                            "timestamp": code_data.get("timestamp"),
                            "last_updated": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else None
                        }
                    
                    # 备选：使用 generated_pipeline_code
                    elif result.get("generated_pipeline_code"):
                        code_content = result["generated_pipeline_code"]
                        # 从其他字段推断元数据
                        file_path = ""
                        if action.tool == "pipeline_workflow_agent" and hasattr(action, 'tool_input'):
                            file_path = action.tool_input.get("python_file_path", "")
                        
                        code_metadata = {
                            "file_name": os.path.basename(file_path) if file_path else "generated_code.py",
                            "file_path": file_path,
                            "language": "python",
                            "tool_source": action.tool,
                            "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else None,
                            "last_updated": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else None
                        }
                
                # 如果有代码内容，存储到state
                if code_content:
                    # 将代码存储为列表格式（支持多个代码文件）
                    if not data.get("generated_code"):
                        data["generated_code"] = []
                    if not data.get("code_metadata"):
                        data["code_metadata"] = {}
                    
                    # 创建代码项
                    code_item = {
                        "content": code_content,
                        "filename": code_metadata.get("file_name", "generated_code.py"),
                        "language": code_metadata.get("language", "python"),
                        "tool_source": code_metadata.get("tool_source", action.tool),
                        "timestamp": code_metadata.get("timestamp"),
                        "file_path": code_metadata.get("file_path", "")
                    }
                    
                    data["generated_code"].append(code_item)
                    data["code_metadata"]["last_updated"] = code_metadata.get("last_updated")
                    data["code_metadata"]["total_files"] = len(data["generated_code"])
                    

                
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
                    logger.info(f"🔍 global_agent_states keys: {list(global_agent_states.keys())}")
                    logger.info(f"🔍 session_id in global_agent_states: {session_id in global_agent_states if session_id else False}")
                    
                    if session_id and session_id in global_agent_states:
                        # 更新global_agent_states中的tool_results
                        global_agent_states[session_id]["tool_results"] = data["tool_results"]
                        if "form_session" in data and data["form_session"] is not None:
                            global_agent_states[session_id]["form_session"] = data["form_session"]
                            logger.info(f"🔍 已同步form_session，missing_params数量: {len(data['form_session'].get('missing_params', []))}")
                        # 🎯 同步代码数据到全局状态
                        if "generated_code" in data:
                            global_agent_states[session_id]["generated_code"] = data["generated_code"]
                        if "code_metadata" in data:
                            global_agent_states[session_id]["code_metadata"] = data["code_metadata"]
                        logger.info(f"🔄 已同步AgentState到全局状态: {session_id}")
                    else:
                        logger.warning(f"⚠️ 无法同步到全局状态: session_id={session_id}, exists={session_id in global_agent_states if session_id else False}")
                except Exception as sync_error:
                    logger.warning(f"⚠️ 同步到全局状态失败: {sync_error}")
                    import traceback
                    logger.warning(traceback.format_exc())
                
                # 🎯 通用会话生命周期管理：根据工具和状态决定是否清理
                try:
                    # 如果有unified_result且不是NEED_USER_INPUT，考虑清理状态
                    if ('unified_tool_results' in data and 
                        data['unified_tool_results'] and 
                        data['unified_tool_results'][-1].get('status') not in ['NEED_USER_INPUT']):
                        
                        # 清理等待用户消息
                        if data.get("awaiting_user_message"):
                            logger.info(f"🗑️ 工具 {action.tool} 执行完成，清理等待用户消息")
                            data["awaiting_user_message"] = None
                        
                        # 向后兼容：清理final_result
                        if data.get("final_result"):
                            logger.info(f"🗑️ 工具 {action.tool} 执行完成，清理final_result")
                            data["final_result"] = None
                except Exception as clear_error:
                    # 如果统一逻辑失败，回退到原有逻辑（向后兼容）
                    logger.debug(f"🔄 统一清理逻辑失败，使用传统方式: {clear_error}")
                
            except Exception as e:
                import traceback
                logger.error(f"❌ 详细错误信息:\n{traceback.format_exc()}")
                err = {"success": False, "error": str(e)}
                if not data.get("intermediate_steps"):
                    data["intermediate_steps"] = []
                data["intermediate_steps"].append((action, err))
                data["tool_results"].append({"tool": action.tool, "ok": False, "payload": err})
        # 清空agent_outcome，避免重复执行
        data["agent_outcome"] = []
        return data
    
    @node
    async def planner(self, data: AgentState) -> AgentState:
        """规划器节点 - 每次只规划下一个单独动作"""
        
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
        form_session = data.get('form_session')
        
        # 🎯 检查是否有活跃会话需要继续（通用会话机制）
        active_session = data.get('form_session')
        logger.info(f"🔍 DEBUG: active_session = {active_session}")
        logger.info(f"🔍 DEBUG: user_input = {user_input}")
        if active_session:
            # 通用会话状态检查：任何工具都可以有活跃会话
            requires_input = active_session.get("requires_user_input", False)
            session_complete = active_session.get("form_complete", False)
            target_tool = active_session.get("target_workflow", "former")  # 默认former，但可扩展
            
            logger.info(f"� 会话状态详情: requires_input={requires_input}, complete={session_complete}, target={target_tool}")
            
            # ✅ 优先检查：表单完成且有目标工具，立即执行
            if (active_session.get("form_complete") and 
                active_session.get("target_workflow") and 
                active_session["target_workflow"] != (active_session.get("owner_tool") or "former")):
                
                # 提取表单参数，支持两种结构：直接字典或 {fields: {...}}
                form_data = active_session.get("form_data") or {}
                if isinstance(form_data, dict) and "fields" in form_data:
                    # 新结构: {fields: {...}}
                    params = form_data.get("fields", {})
                else:
                    # 旧结构: 直接字典 {...}
                    params = form_data
                
                act = create_unified_action(
                    tool_name=active_session["target_workflow"], 
                    llm_args=params,
                    state=data, 
                    user_input=user_input,
                    log_message=f"执行目标工具: {active_session['target_workflow']}"
                )
                
                # 在创建 action 后清理表单会话，避免重复
                data["form_session"] = None
                data["agent_outcome"] = [act]
                data["next_action"] = "continue"
                logger.info(f"🎯 表单完成，执行目标工具: {active_session['target_workflow']}")
                return data
            
            # ✅ 次要检查：需要用户输入，继续会话
            elif active_session.get("requires_user_input") and user_input:
                owner_tool = active_session.get("owner_tool") or "former"
                act = create_unified_action(
                    tool_name=owner_tool,
                    llm_args={"action": "collect_user_response"},
                    state=data, 
                    user_input=user_input,
                    log_message=f"继续会话: {owner_tool}"
                )
                data["agent_outcome"] = [act]
                data["next_action"] = "continue"
                logger.info(f"🎯 继续会话: {owner_tool}")
                return data
        
        # ✅ 统一工具结果后的处理
        unified_results = data.get("unified_tool_results", [])
        if unified_results:
            last_unified = unified_results[-1]
            status = last_unified.get("status")

            # 需要用户输入 → 回到 owner_tool/原工具收集
            if status == "NEED_USER_INPUT":
                owner_tool = (last_unified.get("session") or {}).get("owner_tool") \
                             or last_unified.get("tool_name") or "former"
                act = create_unified_action(
                    tool_name=owner_tool,
                    llm_args={"action": "collect_user_response"},
                    state=data,
                    user_input=user_input,
                    log_message=f"统一协议: {owner_tool}需要用户输入"
                )
                data["agent_outcome"] = [act]
                data["next_action"] = "continue"
                return data

            # 跟进建议 → 直接采纳
            followup = last_unified.get("followup") or {}
            if followup.get("needs_followup") and followup.get("suggested_tool"):
                act = create_unified_action(
                    tool_name=followup["suggested_tool"],
                    llm_args=followup.get("parameters", {}),
                    state=data,
                    user_input=user_input,
                    log_message=f"统一协议建议: {followup['suggested_tool']}"
                )
                data["agent_outcome"] = [act]
                data["next_action"] = "continue"
                return data
        
        # 🎯 检查是否已经处理完所有统一结果（无更多待办事项）
        # 如果最后一个unified_result的status是SUCCESS且没有followup，则进入正常LLM规划

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
                
                single_action = create_unified_action(
                    tool_name=next_action.get("tool", ""),
                    llm_args=next_action.get("tool_input", {}),
                    state=data,
                    user_input=user_input,
                    log_message=llm_reasoning or f"Planner规划: {next_action.get('tool','')}"
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
        
        # ✅ 优先展示统一等待提示（别再读/写 final_result）
        msg = data.get("awaiting_user_message")
        if msg:
            data["agent_outcome"] = LCAgentFinish(
                return_values={"output": msg}, 
                log="Awaiting user input"
            )
            logger.info("📝 使用统一等待用户输入消息")
            return data
        
        # ✅ 兜底：看最近一次 unified_tool_results 的 message
        unified_results = data.get("unified_tool_results", [])
        last = unified_results[-1] if unified_results else None
        if last and last.get("status") == "NEED_USER_INPUT" and last.get("message"):
            data["agent_outcome"] = LCAgentFinish(
                return_values={"output": last["message"]},
                log="Awaiting user input (unified)"
            )
            logger.info("📝 使用统一工具结果等待消息")
            return data
        
        # 🎯 临时向后兼容：读取 final_result 但不再写入
        if data.get("final_result"):
            logger.info("📝 向后兼容：检测到final_result")
            finish = LCAgentFinish(
                return_values={"output": data.get("final_result")},
                log="Tool output (legacy)"
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
            
            # 使用LLMClient的异步调用方法
            try:
                result = await self.llm.call_llm_async(system_prompt, user_prompt)
                
                if result and result.get("content"):
                    llm_response = result["content"].strip()
                    logger.info(f"🤖 LLM智能总结生成成功: {llm_response[:100]}...")
                    return llm_response
                else:
                    logger.warning("⚠️ LLM返回空响应")
                    
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
            
            # 使用LLMClient的同步调用方法
            result = self.llm.call_llm(system_prompt, user_prompt)
            
            if result and result.get("content"):
                content = result["content"].strip()
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
                        "reasons": [reasoning],
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
