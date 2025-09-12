"""
DataFlow Master Agent
基于 MyScaleKB-Agent 架构的主控智能体 - 使用真正的LangGraph工作流
"""
import logging
import asyncio
import time
import uuid
from typing import Dict, List, Any, Union, Optional, Tuple, Protocol
from pydantic import BaseModel
from dataclasses import dataclass
from enum import Enum

# LangGraph核心组件
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import StructuredTool
from langchain_core.agents import AgentFinish as LCAgentFinish, AgentAction as LCAgentAction
from dataflow.agent_v2.base.core import SubAgent, GraphBuilder, BaseTool


from dataflow.agent_v2.llm_client import get_llm_client
from dataflow.agent_v2.subagents.apikey_agent import APIKeyTool
from dataflow.agent_v2.subagents.mock_tools import SleepTool, MockSearchTool, MockFormerTool, MockCodeGenTool
from dataflow.agent_v2.subagents.csvtools import CSVProfileTool, CSVDetectTimeColumnsTool, CSVVegaSpecTool, ASTStaticCheckTool, UnitTestStubTool, LocalIndexBuildTool, LocalIndexQueryTool

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolExecutor

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


# 事件协议定义
class EventType(str, Enum):
    RUN_STARTED = "run_started"
    PLAN_DECISION = "plan_decision"
    TOOL_STARTED = "tool_started"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    RUN_FINISHED = "run_finished"


class Event(BaseModel):
    session_id: str
    step_id: str         # 例如 "step-001" 方便前端做去重/排序
    event: EventType
    data: Dict[str, Any] # 载荷（工具名、参数摘要、结果摘要、决策等）
    ts: float            # time.time()


class EventSink(Protocol):
    async def emit(self, event: Event) -> None: ...

def new_step_id() -> str:
    return f"step-{uuid.uuid4().hex[:8]}"


class PlannerOutput(BaseModel):
    decision: str              # "continue" | "finish"
    next_actions: list = []    # [ {"tool": "...", "tool_input": {...}} ... ]
    user_message: Optional[str] = None
    reasons: Optional[str] = None

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """Master Agent 状态定义 - 支持多轮编排"""
    input: str = ""
    agent_outcome: Optional[Any] = None
    intermediate_steps: List[Tuple[Any, Any]] = []  # 修改为支持结构化结果
    session_id: Optional[str] = None
    current_step: str = "bootstrap"
    form_data: Optional[Dict[str, Any]] = None
    xml_content: Optional[str] = None
    execution_result: Optional[str] = None
    conversation_history: List[Dict[str, str]] = []  # 对话历史
    last_tool_results: Optional[Dict[str, Any]] = None  # 最近的工具结果
    
    # 多轮编排支持
    pending_actions: List[LCAgentAction] = []  # 待执行的动作
    tool_results: List[Dict[str, Any]] = []    # 结构化工具结果
    loop_guard: int = 0                        # 循环计数器
    max_steps: int = 8                         # 最大步数
    context_vars: Dict[str, Any] = {}          # 跨步共享数据
    next_action: Optional[str] = None          # 下一个动作决策
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)


class ActionType(Enum):
    """动作类型"""
    TOOL_EXECUTION = "tool_execution"
    SUB_AGENT_FORWARD = "sub_agent_forward"
    GENERAL_CONVERSATION = "general_conversation"
    END = "end"


class MasterAgent:
    """DataFlow Master Agent - 真正的LangGraph架构"""
    
    def __init__(self):
        self.llm = get_llm_client()  # 初始化真正的LLM客户端
        self.forward_paths = {}
        self.sub_agents = {}
        self.conversation_sessions = {}  # 会话管理
        self.tools = []
        self.compiled_graph = None
        
        # 注册工具
        self._register_tools()
        
        # 构建LangGraph
        self._build_langgraph()
    
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
    
    def _build_langgraph(self):
        """构建真正的LangGraph工作流 - 支持多轮编排"""
        try:
            # 创建StateGraph
            workflow = StateGraph(AgentState)
            
            # 添加节点 - 参照MyScaleKB-Agent的节点结构，增加planner节点
            workflow.add_node("bootstrap", self.bootstrap_node)
            workflow.add_node("execute_tools", self.execute_tools_node)
            workflow.add_node("general_conversation", self.general_conversation_node)
            workflow.add_node("planner", self.planner_node)  # 新增规划器节点
            workflow.add_node("summarize", self.summarize_node)
            
            # 设置入口点
            workflow.set_entry_point("bootstrap")
            
            # 添加条件边 - 参照MyScaleKB-Agent的action_forward逻辑
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
            
            workflow.add_edge("summarize", END)
            
            # 编译图
            self.compiled_graph = workflow.compile()
            logger.info("✅ LangGraph工作流构建成功 - 支持多轮编排")
            
        except Exception as e:
            logger.error(f"LangGraph构建失败: {e}")
            self.compiled_graph = None
    
    async def bootstrap_node(self, state: AgentState) -> AgentState:
        """引导节点 - 每次只规划第一个动作，后续通过planner节点逐步规划"""
        user_input = state.input
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
                state.agent_outcome = [action]  # 注意：只有一个动作
                logger.info(f"🚀 Bootstrap创建单个Action: {action.tool}")
            else:
                # 没有合适的工具，标记为需要通用对话
                logger.info(f"❌ 没有合适工具，使用通用对话 (置信度: {confidence})")

                action = LCAgentAction(
                    tool="general_conversation",
                    tool_input={"user_input": user_input},
                    log="通用对话"
                )

                state.agent_outcome = [action]
                logger.info(f"💬 Bootstrap创建对话Action")
                
        except Exception as e:
            logger.error(f"LLM意图分析失败: {e}")
            
            raise Exception("LLM服务不可用，无法执行任何工具或SubAgent")
        return state
    
    def _extract_main_task(self, user_input: str) -> str:
        """直接返回用户输入，不做任何关键词匹配处理"""
        return user_input.strip()
    
    async def execute_tools_node(self, state: AgentState) -> AgentState:
        """执行工具节点 - 确保每次只执行一个动作"""
        agent_outcome = state.agent_outcome
        
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
                return state
        elif hasattr(agent_outcome, "tool"):
            actions = [agent_outcome]
            logger.info(f"📋 单个动作: {agent_outcome.tool}")
        else:
            logger.warning(f"⚠️ 没有可执行的动作，agent_outcome类型: {type(agent_outcome)}")
            return state

        if state.tool_results is None:
            state.tool_results = []

        # 🔧 关键：只执行这一个动作
        if actions:
            action = actions[0]
            logger.info(f"🛠️ 开始执行单个工具: {action.tool}")
            
            try:
                result = await self.tool_executor.ainvoke(action)
                
                # 记录到intermediate_steps
                state.intermediate_steps.append((action, result))
                
                # 记录到tool_results
                state.tool_results.append({
                    "tool": action.tool,
                    "ok": bool(result.get("success", True)) if isinstance(result, dict) else True,
                    "payload": result
                })
                
                logger.info(f"✅ 工具执行完成: {action.tool}")
                
            except Exception as e:
                logger.error(f"❌ 工具执行失败: {action.tool}, 错误: {e}")
                err = {"success": False, "error": str(e)}
                state.intermediate_steps.append((action, err))
                state.tool_results.append({"tool": action.tool, "ok": False, "payload": err})

        # 🔧 关键：清空agent_outcome，避免重复执行
        state.agent_outcome = []
        logger.info(f"🔄 工具执行完成，清空agent_outcome，进入planner节点")
        
        return state
    
    async def general_conversation_node(self, state: AgentState) -> AgentState:
        """通用对话节点 - 处理不需要工具的对话"""
        user_input = state.input
        conversation_history = state.conversation_history
        
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

                    state.agent_outcome = finish
                    return state
                    
            except Exception as e:
                logger.error(f"通用对话LLM调用失败: {e}")
        
        # LLM不可用时的fallback逻辑
        response = self._get_fallback_response(user_input, conversation_history)
        

        finish = LCAgentFinish(
            return_values={"output": response},
            log="Fallback响应"
        )

        state.agent_outcome = finish
        return state
    
    async def planner_node(self, state: AgentState) -> AgentState:
        """规划器节点 - 每次只规划下一个单独动作"""
        # ✅ 步数护栏
        if not hasattr(state, 'loop_guard'):
            state.loop_guard = 0
        state.loop_guard += 1
        
        max_steps = getattr(state, 'max_steps', 8)
        if state.loop_guard >= max_steps:
            # 触发护栏直接结束
            logger.info(f"🛑 达到最大步骤数 {max_steps}，自动结束")
            state.agent_outcome = []  # 清空，让summarize处理
            state.next_action = "finish"
            return state
            
        logger.info(f"🎯 进入规划器节点 - 步骤 {state.loop_guard}/{max_steps}")
        
        # 🔧 关键修复：每次只规划下一个单独动作
        try:
            analysis = self._analyze_user_needs(state.input, state.tool_results or [])
            logger.info(f"📋 需求分析: {analysis}")
            
            if analysis["should_continue"] and analysis["next_action"]:
                # 🔧 重要：只创建一个动作
                next_action = analysis["next_action"]
                
                single_action = LCAgentAction(
                    tool=next_action.get("tool", ""),
                    tool_input=next_action.get("tool_input", {}),
                    log=f"Planner规划: {next_action.get('tool','')}"
                )
                
                state.agent_outcome = [single_action]  # 注意：只有一个动作
                state.next_action = "continue"
                
                logger.info(f"📋 Planner规划下一个动作: {next_action.get('tool', '')} (单次)")
                logger.info(f"📋 原因: {'; '.join(analysis['reasons']) if isinstance(analysis['reasons'], list) else analysis['reasons']}")
            else:
                # 🔧 修复：流转到summarize节点进行智能总结
                state.agent_outcome = []  # 清空，让summarize_node处理
                state.next_action = "finish"
                logger.info(f"🏁 Planner决定结束，流转到summarize节点")
                logger.info(f"📋 结束原因: {'; '.join(analysis['reasons']) if isinstance(analysis['reasons'], list) else analysis['reasons']}")
                
            return state
            
        except Exception as e:
            logger.error(f"规划器错误: {e}")
            # 🔧 修复：异常时也流转到summarize节点
            state.agent_outcome = []
            state.next_action = "finish"
            state.error_message = f"规划错误: {str(e)}"
            return state

    def planner_router(self, state: AgentState) -> str:
        """规划器路由器 - 修复版本，兜底返回finish"""
        next_action = getattr(state, "next_action", None)
        result = "continue" if next_action == "continue" else "finish"
        logger.info(f"� 路由决策: {next_action} -> {result}")
        return result
    
    async def summarize_node(self, state: AgentState) -> AgentState:
        """总结节点 - 智能总结所有工具执行结果，而不是简单的步骤计数"""
        logger.info(f"📝 总结节点开始，intermediate_steps数量: {len(state.intermediate_steps or [])}")
        
        # 检查是否有错误消息
        if hasattr(state, 'error_message'):

            finish = LCAgentFinish(
                return_values={"output": state.error_message},
                log="错误总结"
            )

            state.agent_outcome = finish
            return state
        
        # 如果已经是最终结果，直接返回
        if hasattr(state.agent_outcome, 'return_values'):
            # 已经是最终结果，直接返回
            logger.info("📝 检测到已有return_values，直接返回")
            return state
        
        # 🔧 核心修复：对所有工具执行结果进行LLM智能总结
        if state.intermediate_steps:
            logger.info(f"🤖 开始LLM智能总结，共{len(state.intermediate_steps)}个执行步骤")
            final_output = await self._generate_conversation_response(state)
            logger.info(f"✅ LLM智能总结完成: {final_output[:100]}...")
        else:
            # 如果没有工具执行，直接使用通用对话回复
            logger.info("💬 没有工具执行，使用通用对话回复")
            final_output = await self._get_direct_conversation_response(state)
        

        finish = LCAgentFinish(
            return_values={"output": final_output},
            log="智能总结完成"
        )

        state.agent_outcome = finish
        
        return state
    
    def action_forward(self, state: AgentState) -> str:
        """决定下一步动作 - 参照MyScaleKB-Agent的action_forward"""
        logger.info(f"🔀 Action Forward开始，agent_outcome类型: {type(state.agent_outcome)}")
        logger.info(f"🔀 Agent outcome内容: {state.agent_outcome}")
        
        # 检查是否是结束状态
        if hasattr(state.agent_outcome, 'return_values'):
            logger.info("📝 检测到return_values，结束流程")
            return "end"

        
        # 获取agent_action - 直接使用agent_outcome或从列表中取第一个
        if isinstance(state.agent_outcome, list):
            agent_action = state.agent_outcome[0] if state.agent_outcome else None
            logger.info(f"🎬 从列表获取agent_action: {agent_action}")
        else:
            agent_action = state.agent_outcome
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
    
    def _simple_keyword_fallback(self, user_input: str) -> Optional[Dict[str, Any]]:
        """当LLM不可用时，所有SubAgent都没有意义，直接抛出错误"""
        raise Exception("LLM服务不可用，无法执行任何工具或SubAgent")
    
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
    
    def _get_fallback_response(self, user_input: str, conversation_history: List[Dict[str, str]]) -> str:
        """Fallback响应 - 不做假装回复，直接抛出异常"""
        raise Exception("LLM服务不可用，无法处理对话")
    
    async def _generate_conversation_response(self, state: AgentState) -> str:
        """基于工具执行结果和对话历史生成智能响应 - 让大模型对所有工具结果进行智能总结"""
        user_input = state.input
        conversation_history = state.conversation_history
        
        # 构建详细的工具执行结果 - 包含所有执行步骤和具体结果
        detailed_tool_results = []
        api_keys_collected = []  # 专门收集API密钥
        sleep_records = []       # 专门收集睡眠记录
        
        for i, (action, result) in enumerate(state.intermediate_steps):
            tool_name = action.tool
            step_num = i + 1
            
            # 解析工具结果并收集详细信息
            if isinstance(result, dict):
                detailed_tool_results.append({
                    "step": step_num,
                    "tool": tool_name,
                    "result": result
                })
                
                # 专门收集API密钥
                if tool_name == "APIKey获取工具" and result.get("access_granted"):
                    api_key = result.get("apikey", "")
                    api_keys_collected.append(f"第{step_num}次: {api_key}")
                
                # 专门收集睡眠记录
                elif tool_name == "sleep_tool" and result.get("success"):
                    duration = result.get("duration", 0)
                    sleep_records.append(f"第{step_num}次睡眠: {duration}秒")
                    
            else:
                # 如果不是字典，也要记录
                detailed_tool_results.append({
                    "step": step_num,
                    "tool": tool_name,
                    "result": {"raw_output": str(result)}
                })
        
        # 如果LLM不可用，使用增强的格式化输出
        if not self.llm.api_available:
            return self._enhanced_format_results(detailed_tool_results, api_keys_collected, sleep_records, user_input)
        
        try:
            # 构建更智能的提示词，要求大模型进行深度总结
            system_prompt = """你是DataFlow智能助手。用户刚刚完成了一系列工具调用，你需要对执行结果进行智能总结和分析。

总结要求：
1. **必须包含所有具体结果** - 如果有多个API密钥，要全部列出；如果有睡眠间隔，要说明具体时长
2. **分析执行过程** - 说明调用顺序、间隔控制等
3. **回答用户关心的问题** - 比如"有什么不同"要具体对比分析
4. **语言自然友好** - 不要生硬地列举，要像真正的助手一样交流
5. **突出重点信息** - 把用户最关心的结果放在前面

特别注意：
- 如果用户问"有什么不同"，要仔细对比每次结果的差异
- 如果有多个API密钥，必须全部显示，不能遗漏
- 如果有时间间隔，要说明具体的等待时间和控制效果"""

            # 构建对话历史文本
            history_text = self._build_history_text(conversation_history, k=10, clip=300)
            
            # 构建详细的工具执行报告
            try:
                execution_report = self._build_detailed_execution_report(detailed_tool_results, api_keys_collected, sleep_records)
                logger.info(f"📊 执行报告构建成功，长度: {len(execution_report)}")
            except Exception as report_error:
                logger.error(f"构建执行报告失败: {report_error}")
                execution_report = f"执行报告构建失败: {str(report_error)}"
            
            user_prompt = f"""用户请求: {user_input}

执行过程详细报告:
{execution_report}

对话历史:
{history_text}

请你作为智能助手，对这次执行结果进行全面、详细的总结。特别要注意：
1. 如果用户问"有什么不同"，要仔细分析每次结果的具体差异
2. 所有获取的API密钥都要完整展示，不能遗漏
3. 如果有间隔控制，要说明具体的时间控制效果
4. 用自然的语言回答，像真正的助手一样"""

            logger.info(f"🚀 准备调用LLM，user_prompt长度: {len(user_prompt)}")
            
            # 调用LLM生成智能总结 - 增加超时控制
            try:
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                
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
        
        # fallback到增强格式化
        return self._enhanced_format_results(detailed_tool_results, api_keys_collected, sleep_records, user_input)
    
    def _simple_format_results(self, tool_results_summary: List[Dict[str, Any]]) -> str:
        """简单格式化工具结果（当LLM不可用时）"""
        for tool_summary in tool_results_summary:
            tool_name = tool_summary["tool"]
            result = tool_summary["result"]
            
            if tool_name == "APIKey获取工具":
                if result.get("access_granted"):
                    api_key = result.get("apikey", "")
                    return f"🔑 今天的秘密API密钥是: `{api_key}`"
                else:
                    return "❌ 无法获取API密钥"
        
        return "✅ 操作完成"
    
    def _enhanced_format_results(self, detailed_tool_results: List[Dict], api_keys_collected: List[str], sleep_records: List[str], user_input: str) -> str:
        """增强格式化工具结果（LLM不可用时的fallback）"""
        if not detailed_tool_results:
            return "✅ 任务完成，但没有工具执行记录"
        
        # 构建详细的执行报告
        report_lines = ["📋 执行报告："]
        
        # 如果有API密钥，优先展示
        if api_keys_collected:
            report_lines.append(f"\n🔑 获取到的API密钥（共{len(api_keys_collected)}个）：")
            for api_key_info in api_keys_collected:
                report_lines.append(f"  • {api_key_info}")
        
        # 如果有睡眠记录，展示间隔控制
        if sleep_records:
            report_lines.append(f"\n⏰ 间隔控制记录：")
            for sleep_info in sleep_records:
                report_lines.append(f"  • {sleep_info}")
        
        # 展示完整执行流程
        report_lines.append(f"\n📝 完整执行流程（共{len(detailed_tool_results)}步）：")
        for tool_result in detailed_tool_results:
            step = tool_result["step"]
            tool = tool_result["tool"]
            result = tool_result["result"]
            
            if tool == "APIKey获取工具" and result.get("access_granted"):
                api_key = result.get("apikey", "N/A")
                report_lines.append(f"  步骤{step}: 获取API密钥 → {api_key}")
            elif tool == "sleep_tool" and result.get("success"):
                duration = result.get("duration", 0)
                report_lines.append(f"  步骤{step}: 睡眠等待 → {duration}秒")
            else:
                report_lines.append(f"  步骤{step}: {tool} → 执行完成")
        
        # 如果用户问"有什么不同"，尝试简单分析
        if "不同" in user_input and api_keys_collected:
            report_lines.append(f"\n🔍 差异分析：")
            if len(api_keys_collected) > 1:
                report_lines.append("  • 每次获取的API密钥都是不同的（包含时间戳）")
                report_lines.append("  • 时间戳体现了执行的先后顺序")
            else:
                report_lines.append("  • 只有一次执行，无法进行差异对比")
        
        return "\n".join(report_lines)
    
    def _build_detailed_execution_report(self, detailed_tool_results: List[Dict], api_keys_collected: List[str], sleep_records: List[str]) -> str:
        """构建详细的执行报告供LLM分析"""
        report_sections = []
        
        # 执行概况
        total_steps = len(detailed_tool_results)
        api_count = len(api_keys_collected)
        sleep_count = len(sleep_records)
        
        report_sections.append(f"执行概况: 总共{total_steps}个步骤, 获取{api_count}个API密钥, {sleep_count}次睡眠间隔")
        
        # API密钥详情
        if api_keys_collected:
            report_sections.append("\nAPI密钥获取详情:")
            for i, api_key_info in enumerate(api_keys_collected, 1):
                report_sections.append(f"  {i}. {api_key_info}")
        
        # 睡眠间隔详情
        if sleep_records:
            report_sections.append("\n睡眠间隔详情:")
            for i, sleep_info in enumerate(sleep_records, 1):
                report_sections.append(f"  {i}. {sleep_info}")
        
        # 完整执行时序
        report_sections.append(f"\n完整执行时序:")
        for tool_result in detailed_tool_results:
            step = tool_result["step"]
            tool = tool_result["tool"]
            result = tool_result["result"]
            
            if tool == "APIKey获取工具":
                if result.get("access_granted"):
                    api_key = result.get("apikey", "N/A")
                    timestamp = api_key.split('_')[-1] if '_' in api_key else "无时间戳"
                    report_sections.append(f"  步骤{step}: [API密钥获取] 成功 → 密钥: {api_key} (时间戳: {timestamp})")
                else:
                    report_sections.append(f"  步骤{step}: [API密钥获取] 失败")
            elif tool == "sleep_tool":
                if result.get("success"):
                    duration = result.get("duration", 0)
                    label = result.get("label", "未知")
                    report_sections.append(f"  步骤{step}: [睡眠间隔] 成功 → 等待{duration}秒 (标签: {label})")
                else:
                    report_sections.append(f"  步骤{step}: [睡眠间隔] 失败")
            else:
                status = "成功" if result.get("success", True) else "失败"
                report_sections.append(f"  步骤{step}: [其他工具: {tool}] {status}")
        
        return "\n".join(report_sections)
    
    async def _get_direct_conversation_response(self, state: AgentState) -> str:
        """当没有工具执行时，获取直接对话回复"""
        user_input = state.input
        conversation_history = state.conversation_history
        
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
        user_input = state.input
        tool_results = state.tool_results
        
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
            # 生成简单的总结消息，避免复杂的LLM调用
            summary_msg = "任务已完成"
            if state.tool_results:
                latest_result = state.tool_results[-1]
                if latest_result.get("tool") == "APIKey获取工具":
                    summary_msg = f"已成功获取API密钥: {latest_result.get('result', {}).get('api_key', 'N/A')}"
                elif latest_result.get("tool") == "sleep_tool":
                    summary_msg = "已完成等待任务"
                else:
                    summary_msg = f"已完成{latest_result.get('tool', '工具')}执行"
            
            return PlannerOutput(
                decision="finish",
                user_message=summary_msg,
                reasons="; ".join(needs_analysis["reasons"]) if isinstance(needs_analysis["reasons"], list) else needs_analysis["reasons"]
            )

    def _analyze_user_needs(self, user_input: str, tool_results: List[Dict]) -> Dict[str, Any]:
        """让LLM智能分析用户需求和当前执行状态，决定下一步行动 - 完全基于LLM决策"""
        
        # 🔧 核心修复：完全去掉关键词匹配，让LLM来理解和决策
        if not self.llm.api_available:
            # LLM不可用时的简单fallback
            return {
                "should_continue": False,
                "next_action": None,
                "reasons": ["LLM不可用，无法进行智能决策"],
                "analysis": {}
            }
        
        try:
            # 构建可用工具的详细描述
            available_tools = []
            for tool in self.tools:
                tool_info = {
                    "name": tool.name(),
                    "description": tool.description()
                }
                available_tools.append(tool_info)
            
            # 构建执行历史
            execution_history = []
            for i, result in enumerate(tool_results, 1):
                tool_name = result.get("tool", "unknown")
                success = result.get("ok", False)
                payload = result.get("payload", {})
                
                step_info = f"步骤{i}: 执行了{tool_name}"
                if success:
                    if isinstance(payload, dict) and payload.get("success"):
                        step_info += " - 成功"
                        if "apikey" in payload:
                            step_info += f" (获得API密钥: {payload['apikey']})"
                        elif "duration" in payload:
                            step_info += f" (等待了{payload['duration']}秒)"
                    else:
                        step_info += " - 完成"
                else:
                    step_info += " - 失败"
                
                execution_history.append(step_info)
            
            # 构建LLM决策提示词
            system_prompt = """你是一个智能决策助手，需要根据用户需求和当前执行情况，决定下一步应该执行什么工具。

决策原则：
1. 仔细理解用户的完整需求，包括要执行多少次、是否需要间隔等
2. 分析当前的执行历史，了解已经完成了什么
3. 基于工具的功能描述，选择最合适的下一步动作
4. 每次只决策一个动作，不要一次性规划多个步骤
5. 如果任务已完成，应该选择结束

返回格式（必须是有效的JSON）：
{
    "should_continue": true/false,
    "next_tool": "工具名称" 或 null,
    "reasoning": "详细的决策原因",
    "task_progress": "当前任务进度分析"
}"""

            user_prompt = f"""用户原始需求: {user_input}

可用工具列表:
{chr(10).join([f"- {tool['name']}: {tool['description']}" for tool in available_tools])}

当前执行历史:
{chr(10).join(execution_history) if execution_history else "还没有执行任何工具"}

请分析当前情况，决定下一步应该：
1. 继续执行某个工具（如果任务未完成）
2. 还是结束任务（如果已经满足用户需求）

注意：每次只能选择一个下一步动作，不要同时规划多个步骤。"""

            logger.info(f"🤖 调用LLM进行智能决策...")
            
            # 调用LLM进行决策
            llm_service = self.llm._create_llm_service()
            responses = llm_service.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt
            )
            
            if responses and responses[0]:
                content = responses[0].strip()
                logger.info(f"🤖 LLM决策响应: {content}")
                
                # 解析LLM响应
                try:
                    import json
                    decision = json.loads(content)
                    
                    should_continue = decision.get("should_continue", False)
                    next_tool = decision.get("next_tool")
                    reasoning = decision.get("reasoning", "")
                    task_progress = decision.get("task_progress", "")
                    
                    # 构建next_action
                    next_action = None
                    if should_continue and next_tool:
                        next_action = {
                            "tool": next_tool,
                            "tool_input": {"user_message": user_input}
                        }
                    
                    result = {
                        "should_continue": should_continue,
                        "next_action": next_action,
                        "reasons": [reasoning],
                        "analysis": {
                            "task_progress": task_progress,
                            "llm_decision": decision,
                            "execution_count": len(tool_results)
                        }
                    }
                    
                    logger.info(f"🎯 LLM决策结果: continue={should_continue}, next_tool={next_tool}")
                    logger.info(f"🎯 决策原因: {reasoning}")
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"LLM响应JSON解析失败: {e}")
                    logger.error(f"原始响应: {content}")
                    
                    # 尝试简单解析
                    return self._simple_parse_llm_response(content, tool_results)
            
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
    
    def _simple_parse_llm_response(self, content: str, tool_results: List[Dict]) -> Dict[str, Any]:
        """简单解析LLM响应的fallback方法"""
        content_lower = content.lower()
        
        # 简单判断是否应该继续
        should_continue = False
        next_tool = None
        reasoning = "基于文本内容的简单解析"
        
        if "继续" in content or "continue" in content_lower or "true" in content_lower:
            should_continue = True
            
            # 尝试找到工具名
            for tool in self.tools:
                tool_name = tool.name()
                if tool_name in content:
                    next_tool = tool_name
                    break
        
        # 构建next_action
        next_action = None
        if should_continue and next_tool:
            next_action = {
                "tool": next_tool,
                "tool_input": {"user_message": "继续执行"}
            }
        
        return {
            "should_continue": should_continue,
            "next_action": next_action,
            "reasons": [reasoning],
            "analysis": {"simple_parse": True, "execution_count": len(tool_results)}
        }


class MasterAgentExecutor:
    """Master Agent 执行器 - 使用LangGraph执行"""
    
    def __init__(self, agent: MasterAgent):
        self.agent = agent
    
    async def execute(self, user_input: str, session_id: str = None) -> Dict[str, Any]:
        """执行用户请求 - 使用LangGraph或Fallback执行器"""
        try:
            # 获取或创建会话历史
            if session_id not in self.agent.conversation_sessions:
                self.agent.conversation_sessions[session_id] = []
            
            conversation_history = self.agent.conversation_sessions[session_id]
            
            # 初始化状态
            initial_state = {
                "input": user_input,
                "session_id": session_id,
                "conversation_history": conversation_history.copy(),
                "agent_outcome": None,
                "intermediate_steps": [],
                "current_step": "bootstrap"
            }
            
            if  self.agent.compiled_graph:
                # 使用真正的LangGraph执行
                logger.info("🚀 使用LangGraph执行")
                
                final_state = await self.agent.compiled_graph.ainvoke(initial_state)
                
                # 获取最终输出
                agent_outcome = final_state.get("agent_outcome")
                if agent_outcome and hasattr(agent_outcome, 'return_values'):
                    output = agent_outcome.return_values.get("output", "执行完成")
                else:
                    output = "执行完成，但未获取到输出"
                
                logger.info(f"✅ LangGraph执行完成")
                
            
            # 保存对话历史
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": output})
            
            # 保持历史长度在合理范围内（最近20轮对话）
            if len(conversation_history) > 40:
                conversation_history = conversation_history[-40:]
            
            self.agent.conversation_sessions[session_id] = conversation_history
            
            return {
                "success": True,
                "output": output,
                "session_id": session_id
            }
                
        except Exception as e:
            logger.error(f"Master Agent 执行失败: {e}")
            return {
                "success": False,
                "output": f"执行失败: {str(e)}",
                "session_id": session_id
            }
    
    async def run_with_events(
        self,
        user_input: str,
        session_id: str,
        sink: EventSink,
    ) -> Dict[str, Any]:
        """新的事件驱动多轮编排执行"""
        state = AgentState(
            input=user_input,
            session_id=session_id,
            conversation_history=self.agent.conversation_sessions.get(session_id, []),
            agent_outcome=None,
            intermediate_steps=[],
            current_step="bootstrap",
        )

        await sink.emit(Event(
            session_id=session_id, step_id=new_step_id(),
            event=EventType.RUN_STARTED, ts=time.time(),
            data={"input": user_input}
        ))

        # 1) bootstrap：初始路由或直接得到第一批 actions
        state = await self.agent.bootstrap_node(state)

        # 如果 bootstrap 直接给了 LCAgentFinish，就结束
        if hasattr(state.agent_outcome, "return_values"):
            await sink.emit(Event(
                session_id=session_id, step_id=new_step_id(),
                event=EventType.RUN_FINISHED, ts=time.time(),
                data={"output": state.agent_outcome.return_values}
            ))
            return {"success": True, "output": state.agent_outcome.return_values.get("output", "")}
            
        # 标准化成 pending_actions
        state.pending_actions = list(state.agent_outcome) if isinstance(state.agent_outcome, list) else []
        
        # 主循环
        state.tool_results = []
        state.loop_guard = 0
        MAX_STEPS = 8

        while state.loop_guard < MAX_STEPS:
            # 若没有待执行动作，进入 planner 决策（可能 finish 或追加 actions）
            if not state.pending_actions:
                plan = await self._planner(state)
                await sink.emit(Event(
                    session_id=session_id, step_id=new_step_id(),
                    event=EventType.PLAN_DECISION, ts=time.time(),
                    data=plan.model_dump()
                ))
                if plan.decision == "finish":
                    final_text = plan.user_message or await self._summarize(state)
                    await sink.emit(Event(
                        session_id=session_id, step_id=new_step_id(),
                        event=EventType.RUN_FINISHED, ts=time.time(),
                        data={"output": final_text}
                    ))
                    self._append_history(session_id, user_input, final_text)
                    return {"success": True, "output": final_text}
                else:
                    for na in plan.next_actions:
                        state.pending_actions.append(LCAgentAction(
                            tool=na["tool"], 
                            tool_input=na.get("tool_input", {}), 
                            log=na.get("log", "")
                        ))

            # 取出一个 action 执行
            action = state.pending_actions.pop(0)
            await sink.emit(Event(
                session_id=session_id, step_id=new_step_id(),
                event=EventType.TOOL_STARTED, ts=time.time(),
                data={"tool": action.tool, "tool_input": action.tool_input}
            ))

            try:
                # 用 ToolExecutor 统一执行
                result = await self.agent.tool_executor.ainvoke(action)
                norm = self._normalize_tool_result(action.tool, result)
                state.tool_results.append(norm)
                await sink.emit(Event(
                    session_id=session_id, step_id=new_step_id(),
                    event=EventType.TOOL_RESULT, ts=time.time(),
                    data={"tool": action.tool, "result": norm}
                ))
            except Exception as e:
                err = {"success": False, "error": str(e)}
                state.tool_results.append({"tool": action.tool, "ok": False, "payload": err})
                await sink.emit(Event(
                    session_id=session_id, step_id=new_step_id(),
                    event=EventType.TOOL_ERROR, ts=time.time(),
                    data={"tool": action.tool, "error": str(e)}
                ))

            state.loop_guard += 1

        # 超过最大步数，兜底结束
        final_text = await self._summarize(state)
        await sink.emit(Event(
            session_id=session_id, step_id=new_step_id(),
            event=EventType.RUN_FINISHED, ts=time.time(),
            data={"output": final_text, "reason": "max_steps_reached"}
        ))
        self._append_history(session_id, user_input, final_text)
        return {"success": True, "output": final_text}

    async def _summarize(self, state: AgentState) -> str:
        """基于工具结果生成总结"""
        return await self.agent._generate_conversation_response(state)

    def _append_history(self, session_id: str, user_input: str, output: str):
        """添加对话历史"""
        conv = self.agent.conversation_sessions.get(session_id, [])
        conv.append({"role": "user", "content": user_input})
        conv.append({"role": "assistant", "content": output})
        self.agent.conversation_sessions[session_id] = conv

    def _normalize_tool_result(self, tool: str, raw: Any) -> Dict[str, Any]:
        """标准化工具结果"""
        return {
            "tool": tool,
            "ok": bool(raw.get("success", True)) if isinstance(raw, dict) else True,
            "payload": raw,
            "summary": raw.get("analysis_result") if isinstance(raw, dict) and "analysis_result" in raw else str(raw)[:200]
        }


# 工厂函数
def create_master_agent() -> Tuple[MasterAgent, MasterAgentExecutor]:
    """创建 Master Agent 和执行器"""
    agent = MasterAgent()
    executor = MasterAgentExecutor(agent)
    return agent, executor
