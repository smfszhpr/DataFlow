"""
DataFlow Master Agent
基于 MyScaleKB-Agent 架构的主控智能体 - 使用真正的LangGraph工作流
"""
import logging
import asyncio
import time
import uuid
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
from dataflow.agent_v2.subagents.mock_tools import SleepTool, MockSearchTool, MockFormerTool, MockCodeGenTool
from dataflow.agent_v2.subagents.csvtools import CSVProfileTool, CSVDetectTimeColumnsTool, CSVVegaSpecTool, ASTStaticCheckTool, UnitTestStubTool, LocalIndexBuildTool, LocalIndexQueryTool

from concurrent.futures import ThreadPoolExecutor

def to_langchain_tool(tool: BaseTool) -> StructuredTool:
    ArgsSchema = tool.params()  # 你的工具已经提供了 Pydantic 参数类

    async def _arun(**kwargs):
        # 交给原工具执行（确保是 async）
        return await tool.execute(**kwargs)

    return StructuredTool.from_function(
        coroutine=_arun,                      # 异步函数
        name=tool.name(),
        description=tool.description(),
        args_schema=ArgsSchema,               # 参数校验
        return_direct=False,                  # 常规情况 False；需要时可 True
    )


def new_step_id() -> str:
    return f"step-{uuid.uuid4().hex[:8]}"


class PlannerOutput(BaseModel):
    decision: str              # "continue" | "finish"
    next_actions: list = []    # [ {"tool": "...", "tool_input": {...}} ... ]
    user_message: Optional[str] = None
    reasons: Optional[str] = None

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


class ActionType(Enum):
    """动作类型"""
    TOOL_EXECUTION = "tool_execution"
    SUB_AGENT_FORWARD = "sub_agent_forward"
    GENERAL_CONVERSATION = "general_conversation"
    END = "end"


class MasterAgent(SubAgent):
    """DataFlow Master Agent - 基于 MyScaleKB-Agent 风格的 LangGraph 架构"""
    
    def __init__(self, ctx=None, llm=None, memory=None, *args, **kwargs):
        # 如果没有传入 llm，创建一个模拟的 llm 对象
        if llm is None:
            class MockLLM:
                def __init__(self):
                    self.model = get_llm_client()
            llm = MockLLM()
        
        # 如果没有传入 ctx，创建一个模拟的 ctx 对象
        if ctx is None:
            class MockContext:
                def __init__(self):
                    self.embedding_model = None
                    self.myscale_client = None
                    self.variables = {"knowledge_scopes": []}
            ctx = MockContext()
        
        # 如果没有传入 memory，创建一个模拟的 memory 对象
        if memory is None:
            class MockMemory:
                pass
            memory = MockMemory()
        
        super().__init__(ctx, llm, memory, *args, **kwargs)
        
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
                # 添加Mock工具用于测试多轮编排
                SleepTool(),
                MockSearchTool(),
                MockFormerTool(),
                MockCodeGenTool(),
                CSVProfileTool(), 
                CSVDetectTimeColumnsTool(), 
                CSVVegaSpecTool(), 
                ASTStaticCheckTool(), 
                UnitTestStubTool(), 
                LocalIndexBuildTool(), 
                LocalIndexQueryTool()

            ]
            logger.info(f"已注册 {len(self.tools)} 个工具")
        except Exception as e:
            logger.error(f"工具注册失败: {e}")
            self.tools = []
        
        self.lc_tools = [to_langchain_tool(t) for t in self.tools]
        self.tool_executor = ToolExecutor(self.lc_tools)
    
    def build_app(self):
        """构建代理工作流 - 类似 MyScaleKB-Agent 的实现"""
        workflow = self._build_graph(AgentState, compiled=False)
        
        # 设置条件入口点
        workflow.set_conditional_entry_point(
            self.entry,
            {
                "bootstrap": "bootstrap",
            }
        )
        
        # 添加条件边
        workflow.add_conditional_edges(
            "bootstrap",
            self.action_forward,
            {
                "execute_tools": "execute_tools",
                "general_conversation": "general_conversation", 
                "end": "summarize"
            }
        )
        
        # 执行工具后进入规划器进行下一轮决策
        workflow.add_edge("execute_tools", "planner")
        workflow.add_edge("general_conversation", "planner")
        
        # 规划器决定继续执行还是结束
        workflow.add_conditional_edges(
            "planner",
            self.planner_router,
            {
                "continue": "execute_tools",  # 继续执行更多工具
                "finish": "summarize"        # 完成任务
            }
        )
        
        workflow.add_edge("summarize", GraphBuilder.END)
        
        return workflow.compile()
    
    @staticmethod
    async def entry(data):
        """入口点 - 决定路由到哪个节点"""
        logger.info("🚪 进入Master Agent入口点")
        # 默认进入bootstrap节点进行引导
        return "bootstrap"
    
    @node
    async def bootstrap(self, data):
        """引导节点 - 每次只规划第一个动作，后续通过planner节点逐步规划"""
        user_input = data.get("input", "")
        logger.info(f"🔄 Bootstrap节点: {user_input}")
        
        # 🔧 关键修复：Bootstrap只负责启动第一个动作，不进行复杂的多步骤规划
        # 获取可用工具列表
        available_tools = [tool.name() for tool in self.tools]
        logger.info(f"🔧 可用工具列表: {available_tools}")
        
        # 🔧 重要：Bootstrap阶段只选择启动动作，不考虑次数和间隔
        # 简化用户输入，只提取主要任务类型
        simplified_input = self._extract_main_task(user_input)
        logger.info(f"📝 简化任务: {simplified_input}")
        
        # 使用LLM分析用户意图，但只关注第一个动作
        try:
            intent_analysis = self.llm.analyze_user_intent(simplified_input, available_tools)
            
            selected_tool = intent_analysis.get("selected_tool")
            confidence = intent_analysis.get("confidence", 0.0)
            parameters = intent_analysis.get("parameters", {})
            
            logger.info(f"🎯 意图分析结果: 工具={selected_tool}, 置信度={confidence}")
            
            if selected_tool and confidence > 0.3:
                # 🔧 重要：只创建一个动作，让planner负责后续规划
                logger.info(f"✅ Bootstrap选择工具: {selected_tool} (单次执行)")
                
                action = LCAgentAction(
                    tool=selected_tool,
                    tool_input=parameters,
                    log=f"Bootstrap启动: {selected_tool}"
                )
                data["agent_outcome"] = [action]  # 注意：只有一个动作
                logger.info(f"🚀 Bootstrap创建单个Action: {action.tool}")
            else:
                # 没有合适的工具，标记为需要通用对话
                logger.info(f"❌ 没有合适工具，使用通用对话 (置信度: {confidence})")

                action = LCAgentAction(
                    tool="general_conversation",
                    tool_input={"user_input": user_input},
                    log="通用对话"
                )

                data["agent_outcome"] = [action]
                logger.info(f"💬 Bootstrap创建对话Action")
                
        except Exception as e:
            logger.error(f"LLM意图分析失败: {e}")
            
            raise Exception("LLM服务不可用，无法执行任何工具或SubAgent")
        return data
    
    def _extract_main_task(self, user_input: str) -> str:
        """直接返回用户输入，不做任何关键词匹配处理"""
        return user_input.strip()
    
    @node
    @edge(target_node="planner")
    async def execute_tools(self, data):
        """执行工具节点 - 确保每次只执行一个动作"""
        agent_outcome = data.get("agent_outcome")
        
        logger.info(f"🛠️ 进入execute_tools_node，agent_outcome类型: {type(agent_outcome)}")
        logger.info(f"🛠️ agent_outcome内容: {agent_outcome}")

        # 🔧 关键修复：确保每次只处理一个动作
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
                result = await self.tool_executor.ainvoke(action)
                
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
                
                logger.info(f"✅ 工具执行完成: {action.tool}")
                
            except Exception as e:
                logger.error(f"❌ 工具执行失败: {action.tool}, 错误: {e}")
                err = {"success": False, "error": str(e)}
                if not data.get("intermediate_steps"):
                    data["intermediate_steps"] = []
                data["intermediate_steps"].append((action, err))
                data["tool_results"].append({"tool": action.tool, "ok": False, "payload": err})

        # 🔧 关键：清空agent_outcome，避免重复执行
        data["agent_outcome"] = []
        logger.info(f"🔄 工具执行完成，清空agent_outcome，进入planner节点")
        
        return data
    
    @node
    @edge(target_node="planner")
    async def general_conversation(self, data: AgentState) -> AgentState:
        """通用对话节点 - 处理不需要工具的对话"""
        user_input = data.get("input", "")
        conversation_history = data.get("conversation_history", [])
        
        logger.info(f"💬 通用对话节点: {user_input}")
        
        # 如果LLM可用，使用智能对话
        if self.llm.api_available:
            try:
                # 构建通用对话提示词
                system_prompt = """你是DataFlow智能助手，一个专业、友好、智能的AI助手。你可以：

1. 回答各种通用问题（科学、技术、生活、学习等）
2. 提供专业的编程、数据分析建议
3. 协助解决问题和提供创意想法
4. 记住对话历史，保持连贯对话
5. 当用户需要时，建议使用专业工具（API密钥、表单生成、数据处理等）

你的特点：
- 知识丰富，善于分析和解释
- 回答准确、有条理
- 语言自然、友好
- 能够根据上下文理解用户真正的需求
- 如果用户问题可能需要专业工具，会主动建议

请用中文回答，保持专业但友好的语气。"""

                # 构建对话历史
                history_text = self._build_history_text(conversation_history, k=8, clip=200)
                
                user_prompt = f"""用户问题: {user_input}

对话历史:
{history_text}

请基于对话历史和当前问题，给出最合适的回答。如果用户询问之前的对话内容，请准确回忆。如果问题可能需要专业工具协助（如API密钥获取、表单生成、数据分析、代码生成等），请主动建议。"""

                # 调用LLM
                llm_service = self.llm._create_llm_service()
                responses = llm_service.generate_from_input(
                    user_inputs=[user_prompt],
                    system_prompt=system_prompt
                )
                
                if responses and responses[0]:
                    response = responses[0].strip()
                    
                    # 如果回复太短，尝试扩展
                    if len(response) < 30:
                        followup_prompt = f"""用户问题: {user_input}

请提供一个更详细、更有帮助的回答。即使问题简单，也要给出友好的回复和可能的扩展建议。如果问题涉及技术，可以提供一些相关背景知识。"""
                        
                        followup_responses = llm_service.generate_from_input(
                            user_inputs=[followup_prompt],
                            system_prompt=system_prompt
                        )
                        
                        if followup_responses and followup_responses[0]:
                            response = followup_responses[0].strip()
                    

                    finish = LCAgentFinish(
                        return_values={"output": response},
                        log="通用对话完成"
                    )

                    data["agent_outcome"] = finish
                    return data
                    
            except Exception as e:
                logger.error(f"通用对话LLM调用失败: {e}")
        
        # LLM不可用时的fallback逻辑
        raise Exception("LLM服务不可用，无法处理对话")
    
    @node
    async def planner(self, data: AgentState) -> AgentState:
        """规划器节点 - 每次只规划下一个单独动作"""
        # ✅ 步数护栏
        if data.get('loop_guard') is None:
            data["loop_guard"] = 0
        data["loop_guard"] += 1
        
        max_steps = data.get('max_steps', 8)
        if data["loop_guard"] >= max_steps:
            # 触发护栏直接结束
            logger.info(f"🛑 达到最大步骤数 {max_steps}，自动结束")
            data["agent_outcome"] = []  # 清空，让summarize处理
            data["next_action"] = "finish"
            return data
            
        logger.info(f"🎯 进入规划器节点 - 步骤 {data['loop_guard']}/{max_steps}")
        
        # 🔧 关键修复：每次只规划下一个单独动作
        try:
            analysis = self._analyze_user_needs(data.get("input", ""), data.get("tool_results", []))
            logger.info(f"📋 需求分析: {analysis}")
            
            if analysis["should_continue"] and analysis["next_action"]:
                # 🔧 重要：只创建一个动作
                next_action = analysis["next_action"]
                
                single_action = LCAgentAction(
                    tool=next_action.get("tool", ""),
                    tool_input=next_action.get("tool_input", {}),
                    log=f"Planner规划: {next_action.get('tool','')}"
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
                logger.info(f"📋 结束原因: {'; '.join(analysis['reasons']) if isinstance(analysis['reasons'], list) else analysis['reasons']}")
                
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
        """总结节点 - 智能总结所有工具执行结果，而不是简单的步骤计数"""
        logger.info(f"📝 总结节点开始，intermediate_steps数量: {len(data.get('intermediate_steps', []))}")
        
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
        
        # 🔧 核心修复：对所有工具执行结果进行LLM智能总结
        intermediate_steps = data.get('intermediate_steps', [])
        if intermediate_steps:
            logger.info(f"🤖 开始LLM智能总结，共{len(intermediate_steps)}个执行步骤")
            final_output = await self._generate_conversation_response(data)
            logger.info(f"✅ LLM智能总结完成: {final_output[:100]}...")
        else:
            # 如果没有工具执行，直接使用通用对话回复
            logger.info("💬 没有工具执行，使用通用对话回复")
            final_output = await self._get_direct_conversation_response(data)
        

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
                return "summarize"
            else:
                logger.info("🔄 工具执行完成，回到planner继续决策")
                return "planner"
        
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
                # 除了general_conversation外，所有工具都路由到execute_tools
                if tool_name == "general_conversation":
                    logger.info("💬 路由到: general_conversation")
                    return "general_conversation"
                else:
                    logger.info(f"🛠️ 路由到: execute_tools (工具: {tool_name})")
                    return "execute_tools"
            
        logger.info("⚠️ 无匹配条件，默认路由到general_conversation")
        return "general_conversation"

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
                    # 不跳过任何字段，让LLM看到完整的工具返回数据
                    if isinstance(value, (str, int, float, bool)) and len(str(value)) < 200:
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
            # 构建更智能的提示词，要求大模型进行深度总结
            system_prompt = """你是DataFlow智能助手。请基于工具执行结果进行总结。

要求：
1. 准确显示所有工具返回的数据，包括具体的API密钥值
2. 不要编造信息，所有内容都基于执行报告
3. 用自然的语言总结执行结果"""

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

请你作为智能助手，对这次执行结果进行全面、详细的总结。特别要注意：
1. 执行报告中的所有数据都是真实的工具返回结果，必须准确显示
2.不需要详细说明执行流程，只需总结结果和分析
3. 用户有权查看所有工具返回的数据，不要隐藏任何信息
4. 基于报告中的真实数据进行分析，不要编造"""

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
                    if key not in priority_fields and isinstance(value, (str, int, float, bool)):
                        if len(str(value)) < 100:  # 只显示合理长度的字段
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
    
    async def _get_direct_conversation_response(self, data: AgentState) -> str:
        """当没有工具执行时，获取直接对话回复"""
        user_input = data.get("input", "")
        conversation_history = data.get("conversation_history", [])
        
        # 直接使用LLM，不做fallback
        if self.llm.api_available:
            try:
                system_prompt = "你是DataFlow智能助手，请直接、自然地回答用户问题。"
                
                # 构建对话历史
                history_text = self._build_history_text(conversation_history, k=8, clip=200)
                
                user_prompt = f"""用户问题: {user_input}

对话历史:
{history_text}

请基于对话历史自然地回答用户问题。"""
                
                llm_service = self.llm._create_llm_service()
                responses = llm_service.generate_from_input(
                    user_inputs=[user_prompt],
                    system_prompt=system_prompt
                )
                
                if responses and responses[0]:
                    return responses[0].strip()
                    
            except Exception as e:
                logger.error(f"LLM调用失败: {e}")
                raise e
        
        # LLM不可用时抛出异常
        raise Exception("LLM服务不可用，无法处理对话")

    async def _planner(self, state: AgentState) -> PlannerOutput:
        """智能规划器：基于用户需求和已执行的工具结果，决定下一步行动"""
        user_input = state.get("input", "")
        tool_results = state.get("tool_results", [])
        
        # 分析用户原始需求中的关键信息
        needs_analysis = self._analyze_user_needs(user_input, tool_results)
        
        logger.info(f"🎯 规划器分析: {needs_analysis}")
        
        # 基于分析结果决定下一步
        if needs_analysis["should_continue"]:
            next_action = needs_analysis["next_action"]
            logger.info(f"✅ 规划器决定继续: {next_action}")
            return PlannerOutput(
                decision="continue",
                next_actions=[next_action],
                reasons="; ".join(needs_analysis["reasons"]) if isinstance(needs_analysis["reasons"], list) else needs_analysis["reasons"]
            )
        else:
            logger.info(f"🏁 规划器决定结束: {needs_analysis['reasons']}")
            # 生成简单的总结消息，避免复杂的LLM调用 - 通用化处理
            summary_msg = "任务已完成"
            tool_results = state.get("tool_results", [])
            if tool_results:
                latest_result = tool_results[-1]
                tool_name = latest_result.get("tool", "未知工具")
                result_data = latest_result.get("result", {})
                
                # 通用化地提取结果信息
                if isinstance(result_data, dict):
                    important_info = []
                    for key, value in result_data.items():
                        if key in ["success", "ok", "status"]:
                            continue
                        elif isinstance(value, (str, int, float)) and len(str(value)) < 50:
                            important_info.append(f"{key}: {value}")
                    
                    if important_info:
                        info_text = ", ".join(important_info[:1])  # 只显示第1个重要字段
                        summary_msg = f"已完成{tool_name}执行，结果: {info_text}"
                    else:
                        summary_msg = f"已完成{tool_name}执行"
                else:
                    summary_msg = f"已完成{tool_name}执行"
            
            return PlannerOutput(
                decision="finish",
                user_message=summary_msg,
                reasons="; ".join(needs_analysis["reasons"]) if isinstance(needs_analysis["reasons"], list) else needs_analysis["reasons"]
            )

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
                            
                            # 通用化地提取重要信息，显示所有关键字段
                            important_info = []
                            for key, value in payload.items():
                                if isinstance(value, (str, int, float, bool)) and len(str(value)) < 50:
                                    important_info.append(f"{key}: {value}")
                            
                            if important_info:
                                info_text = ", ".join(important_info[:3])  # 显示前3个重要字段
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
1. 仔细理解用户的完整需求，包括要执行多少次、是否需要间隔等
2. 分析当前的执行历史，了解已经完成了什么
3. 基于工具的功能描述，选择最合适的下一步动作
4. 每次只决策一个动作，不要一次性规划多个步骤
5. 如果任务已完成，应该选择结束

**重要：你必须只输出JSON格式，不要有任何额外的解释文字！**

返回格式（必须是纯JSON，无任何其他内容）：
{
    "decision": "continue" 或 "finish",
    "tool": "工具名称",
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
1. 继续执行某个工具（如果任务未完成）
2. 还是结束任务（如果已经满足用户需求）

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


