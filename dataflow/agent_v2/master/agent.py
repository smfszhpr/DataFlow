"""
DataFlow Master Agent
基于 MyScaleKB-Agent 架构的主控智能体 - 使用真正的LangGraph工作流
"""
import logging
import asyncio
from typing import Dict, List, Any, Union, Optional, Tuple
from pydantic import BaseModel
from dataclasses import dataclass
from enum import Enum

# LangGraph核心组件
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import StructuredTool
from langchain_core.agents import AgentFinish as LCAgentFinish, AgentAction as LCAgentAction
LANGGRAPH_AVAILABLE = True
from dataflow.agent_v2.base.core import SubAgent, GraphBuilder, BaseTool, node, entry, conditional_edge
from dataflow.agent.agentrole.former import FormerAgent
from dataflow.agent.xmlforms.models import FormRequest, FormResponse
from dataflow.agent_v2.llm_client import get_llm_client
from dataflow.agent_v2.subagents.apikey_agent import APIKeyTool

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """Master Agent 状态定义 - 兼容LangGraph的TypedDict"""
    input: str = ""
    agent_outcome: Optional[Any] = None
    intermediate_steps: List[Tuple[Any, str]] = []
    session_id: Optional[str] = None
    current_step: str = "bootstrap"
    form_data: Optional[Dict[str, Any]] = None
    xml_content: Optional[str] = None
    execution_result: Optional[str] = None
    conversation_history: List[Dict[str, str]] = []  # 对话历史
    last_tool_results: Optional[Dict[str, Any]] = None  # 最近的工具结果
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)


class ActionType(Enum):
    """动作类型"""
    TOOL_EXECUTION = "tool_execution"
    SUB_AGENT_FORWARD = "sub_agent_forward"
    GENERAL_CONVERSATION = "general_conversation"
    END = "end"


@dataclass
class AgentAction:
    """代理动作"""
    tool: str
    tool_input: Dict[str, Any]
    action_type: ActionType = ActionType.TOOL_EXECUTION


@dataclass 
class AgentFinish:
    """代理结束"""
    return_values: Dict[str, Any]
    log: str = ""


class FormerAgentTool(BaseTool):
    """Former Agent 工具封装"""
    
    def __init__(self):
        self.former_agent = FormerAgent()
    
    @classmethod
    def name(cls) -> str:
        return "former_agent"
    
    @classmethod 
    def description(cls) -> str:
        return "处理用户对话，生成XML表单。适用于需要收集用户需求并生成结构化配置的场景。"
    
    def params(self) -> type[BaseModel]:
        class FormerParams(BaseModel):
            user_query: str
            session_id: Optional[str] = None
            conversation_history: List[Dict[str, str]] = []
        return FormerParams
    
    async def execute(self, user_query: str, session_id: str = None, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """执行 Former Agent"""
        try:
            form_request = FormRequest(
                user_query=user_query,
                session_id=session_id,
                conversation_history=conversation_history or []
            )
            
            response = await self.former_agent.process_conversation(form_request)
            
            return {
                "success": True,
                "need_more_info": response.need_more_info,
                "agent_response": response.agent_response,
                "xml_form": response.xml_form,
                "form_type": response.form_type,
                "session_id": session_id
            }
        except Exception as e:
            logger.error(f"Former Agent 执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_response": "抱歉，处理您的请求时发生错误。"
            }


class DataAnalysisTool(BaseTool):
    """数据分析工具"""
    
    @classmethod
    def name(cls) -> str:
        return "data_analysis"
    
    @classmethod
    def description(cls) -> str:
        return "分析数据集并提供洞察。可以处理CSV、JSON等格式的数据。"
    
    def params(self) -> type[BaseModel]:
        class AnalysisParams(BaseModel):
            data_path: str
            analysis_type: str = "basic"
            output_format: str = "summary"
        return AnalysisParams
    
    async def execute(self, data_path: str, analysis_type: str = "basic", output_format: str = "summary") -> Dict[str, Any]:
        """执行数据分析"""
        await asyncio.sleep(1)  # 模拟异步处理
        
        return {
            "success": True,
            "analysis_result": f"对 {data_path} 进行了 {analysis_type} 分析",
            "insights": [
                "数据质量良好",
                "发现3个主要模式",
                "建议进行进一步清洗"
            ],
            "output_format": output_format
        }


class CodeGeneratorTool(BaseTool):
    """代码生成工具"""
    
    @classmethod
    def name(cls) -> str:
        return "code_generator"
    
    @classmethod
    def description(cls) -> str:
        return "根据需求生成 DataFlow 算子代码。支持多种编程模式和数据处理任务。"
    
    def params(self) -> type[BaseModel]:
        class CodeGenParams(BaseModel):
            requirements: str
            operator_type: str = "processor"
            language: str = "python"
        return CodeGenParams
    
    async def execute(self, requirements: str, operator_type: str = "processor", language: str = "python") -> Dict[str, Any]:
        """生成代码"""
        await asyncio.sleep(1.5)  # 模拟代码生成时间
        
        generated_code = f'''
def {operator_type}_operator(data):
    """
    根据需求生成的算子: {requirements}
    """
    # TODO: 实现具体逻辑
    result = process_data(data)
    return result

def process_data(data):
    # 处理数据的核心逻辑
    return data
'''
        
        return {
            "success": True,
            "generated_code": generated_code,
            "operator_type": operator_type,
            "language": language,
            "file_path": f"generated_{operator_type}_operator.py"
        }


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
        if LANGGRAPH_AVAILABLE:
            self._build_langgraph()
        else:
            logger.warning("LangGraph不可用，使用简化版本")
    
    def _register_tools(self):
        """注册工具"""
        try:
            self.tools = [
                FormerAgentTool(),
                DataAnalysisTool(),
                CodeGeneratorTool(),
                APIKeyTool()  # 新增API密钥工具
            ]
            logger.info(f"已注册 {len(self.tools)} 个工具")
        except Exception as e:
            logger.error(f"工具注册失败: {e}")
            self.tools = []
    
    def _build_langgraph(self):
        """构建真正的LangGraph工作流 - 参照MyScaleKB-Agent"""
        if not LANGGRAPH_AVAILABLE:
            return
        
        try:
            # 创建StateGraph
            workflow = StateGraph(AgentState)
            
            # 添加节点 - 参照MyScaleKB-Agent的节点结构
            workflow.add_node("bootstrap", self.bootstrap_node)
            workflow.add_node("execute_tools", self.execute_tools_node)
            workflow.add_node("general_conversation", self.general_conversation_node)
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
            
            # 添加普通边
            workflow.add_edge("execute_tools", "summarize")
            workflow.add_edge("general_conversation", "summarize")
            workflow.add_edge("summarize", END)
            
            # 编译图
            self.compiled_graph = workflow.compile()
            logger.info("✅ LangGraph工作流构建成功")
            
        except Exception as e:
            logger.error(f"LangGraph构建失败: {e}")
            self.compiled_graph = None
    
    async def bootstrap_node(self, state: AgentState) -> AgentState:
        """引导节点 - 参照MyScaleKB-Agent的bootstrap模式"""
        user_input = state.input
        logger.info(f"🔄 Bootstrap节点: {user_input}")
        
        # 获取可用工具列表
        available_tools = [tool.name() for tool in self.tools]
        logger.info(f"🔧 可用工具列表: {available_tools}")
        
        # 使用LLM分析用户意图
        try:
            intent_analysis = self.llm.analyze_user_intent(user_input, available_tools)
            
            selected_tool = intent_analysis.get("selected_tool")
            confidence = intent_analysis.get("confidence", 0.0)
            parameters = intent_analysis.get("parameters", {})
            
            logger.info(f"🎯 意图分析结果: 工具={selected_tool}, 置信度={confidence}")
            logger.info(f"📋 完整意图分析: {intent_analysis}")
            
            if selected_tool and confidence > 0.3:
                # 有合适的工具，执行工具
                logger.info(f"✅ 选择执行工具: {selected_tool} (置信度: {confidence})")
                if LANGGRAPH_AVAILABLE:
                    action = LCAgentAction(
                        tool=selected_tool,
                        tool_input=parameters,
                        log=f"选择工具: {selected_tool}"
                    )
                else:
                    action = AgentAction(
                        tool=selected_tool,
                        tool_input=parameters,
                        action_type=ActionType.TOOL_EXECUTION
                    )
                state.agent_outcome = [action]
                logger.info(f"🚀 创建工具执行Action: {action}")
            else:
                # 没有合适的工具，标记为需要通用对话
                logger.info(f"❌ 没有合适工具，使用通用对话 (置信度: {confidence})")
                if LANGGRAPH_AVAILABLE:
                    action = LCAgentAction(
                        tool="general_conversation",
                        tool_input={"user_input": user_input},
                        log="通用对话"
                    )
                else:
                    action = AgentAction(
                        tool="general_conversation",
                        tool_input={"user_input": user_input},
                        action_type=ActionType.GENERAL_CONVERSATION
                    )
                state.agent_outcome = [action]
                logger.info(f"💬 创建通用对话Action: {action}")
                
        except Exception as e:
            logger.error(f"LLM意图分析失败: {e}")
            
            # fallback到简单的关键词匹配
            selected_tool = self._simple_keyword_fallback(user_input)
            if selected_tool:
                if LANGGRAPH_AVAILABLE:
                    action = LCAgentAction(
                        tool=selected_tool["name"],
                        tool_input=selected_tool["input"],
                        log=f"关键词匹配: {selected_tool['name']}"
                    )
                else:
                    action = AgentAction(
                        tool=selected_tool["name"],
                        tool_input=selected_tool["input"],
                        action_type=ActionType.TOOL_EXECUTION
                    )
                state.agent_outcome = [action]
            else:
                # fallback也没找到工具，标记为需要通用对话
                if LANGGRAPH_AVAILABLE:
                    action = LCAgentAction(
                        tool="general_conversation",
                        tool_input={"user_input": user_input},
                        log="fallback到通用对话"
                    )
                else:
                    action = AgentAction(
                        tool="general_conversation",
                        tool_input={"user_input": user_input},
                        action_type=ActionType.GENERAL_CONVERSATION
                    )
                state.agent_outcome = [action]
        
        return state
    
    async def execute_tools_node(self, state: AgentState) -> AgentState:
        """工具执行节点 - 参照MyScaleKB-Agent的execute_tools"""
        agent_outcome = state.agent_outcome
        if not isinstance(agent_outcome, list):
            return state
        
        intermediate_steps = []
        
        logger.info(f"🔧 执行工具节点，工具数量: {len(agent_outcome)}")
        
        for action in agent_outcome:
            if LANGGRAPH_AVAILABLE and hasattr(action, 'tool'):
                tool_name = action.tool
                tool_input = action.tool_input
            elif hasattr(action, 'tool'):
                tool_name = action.tool
                tool_input = action.tool_input
            else:
                continue
                
            logger.info(f"执行工具: {tool_name}, 参数: {tool_input}")
            
            # 查找并执行工具
            tool = self._find_tool(tool_name)
            if tool:
                try:
                    result = await tool.execute(**tool_input)
                    intermediate_steps.append((action, str(result)))
                    logger.info(f"工具 {tool_name} 执行成功")
                except Exception as e:
                    logger.error(f"工具执行失败: {e}")
                    intermediate_steps.append((action, f"执行失败: {str(e)}"))
            else:
                logger.warning(f"未找到工具: {tool_name}")
                intermediate_steps.append((action, f"未找到工具: {tool_name}"))
        
        state.intermediate_steps = intermediate_steps
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
                    
                    if LANGGRAPH_AVAILABLE:
                        finish = LCAgentFinish(
                            return_values={"output": response},
                            log="通用对话完成"
                        )
                    else:
                        finish = AgentFinish(
                            return_values={"output": response}
                        )
                    state.agent_outcome = finish
                    return state
                    
            except Exception as e:
                logger.error(f"通用对话LLM调用失败: {e}")
        
        # LLM不可用时的fallback逻辑
        response = self._get_fallback_response(user_input, conversation_history)
        
        if LANGGRAPH_AVAILABLE:
            finish = LCAgentFinish(
                return_values={"output": response},
                log="Fallback响应"
            )
        else:
            finish = AgentFinish(
                return_values={"output": response}
            )
        state.agent_outcome = finish
        return state
    
    async def summarize_node(self, state: AgentState) -> AgentState:
        """总结节点 - 参照MyScaleKB-Agent的summarize模式"""
        logger.info(f"📝 总结节点")
        
        if LANGGRAPH_AVAILABLE and hasattr(state.agent_outcome, 'return_values'):
            # 如果已经是最终结果，直接返回
            return state
        elif hasattr(state, 'agent_outcome') and isinstance(state.agent_outcome, AgentFinish):
            # 如果已经是最终结果，直接返回
            return state
        
        # 使用LLM基于工具执行结果进行智能对话
        if state.intermediate_steps:
            final_output = await self._generate_conversation_response(state)
        else:
            # 如果没有工具执行，直接使用通用对话回复
            final_output = await self._get_direct_conversation_response(state)
        
        if LANGGRAPH_AVAILABLE:
            finish = LCAgentFinish(
                return_values={"output": final_output},
                log="总结完成"
            )
        else:
            finish = AgentFinish(
                return_values={"output": final_output}
            )
        state.agent_outcome = finish
        
        return state
    
    def action_forward(self, state: AgentState) -> str:
        """决定下一步动作 - 参照MyScaleKB-Agent的action_forward"""
        logger.info(f"🔀 Action Forward开始，agent_outcome类型: {type(state.agent_outcome)}")
        logger.info(f"🔀 Agent outcome内容: {state.agent_outcome}")
        
        # 检查是否是结束状态
        if LANGGRAPH_AVAILABLE and hasattr(state.agent_outcome, 'return_values'):
            logger.info("📝 检测到return_values，结束流程")
            return "end"
        elif hasattr(state, 'agent_outcome') and isinstance(state.agent_outcome, AgentFinish):
            logger.info("🏁 检测到AgentFinish，结束流程")
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
            if LANGGRAPH_AVAILABLE and hasattr(agent_action, 'tool'):
                tool_name = agent_action.tool
                logger.info(f"🔧 LangGraph模式 - 工具名: {tool_name}")
                # 除了general_conversation外，所有工具都路由到execute_tools
                if tool_name == "general_conversation":
                    logger.info("💬 路由到: general_conversation")
                    return "general_conversation"
                else:
                    logger.info(f"🛠️ 路由到: execute_tools (工具: {tool_name})")
                    return "execute_tools"
            
            # 检查传统模式下的工具
            elif hasattr(agent_action, 'tool'):
                tool_name = agent_action.tool
                logger.info(f"🔧 传统模式 - 工具名: {tool_name}")
                # 有工具就执行工具
                logger.info(f"🛠️ 路由到: execute_tools (工具: {tool_name})")
                return "execute_tools"
        
        logger.info("⚠️ 无匹配条件，默认路由到general_conversation")
        return "general_conversation"
    def _simple_keyword_fallback(self, user_input: str) -> Optional[Dict[str, Any]]:
        """当LLM不可用时，所有SubAgent都没有意义，直接抛出错误"""
        raise Exception("LLM服务不可用，无法执行任何工具或SubAgent")
    
    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        """查找工具"""
        for tool in self.tools:
            if tool.name() == tool_name:
                return tool
        return None
    
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
        """基于工具执行结果和对话历史生成智能响应"""
        user_input = state.input
        conversation_history = state.conversation_history
        
        # 构建工具执行摘要
        tool_results_summary = []
        for action, result_str in state.intermediate_steps:
            tool_name = action.tool
            
            # 解析工具结果
            import ast
            try:
                if result_str.startswith("{'") or result_str.startswith('{"'):
                    result_dict = ast.literal_eval(result_str)
                    tool_results_summary.append({
                        "tool": tool_name,
                        "result": result_dict
                    })
                else:
                    tool_results_summary.append({
                        "tool": tool_name,
                        "result": {"raw_output": result_str}
                    })
            except (ValueError, SyntaxError):
                tool_results_summary.append({
                    "tool": tool_name,
                    "result": {"raw_output": result_str}
                })
        
        # 如果LLM不可用，使用简单格式化
        if not self.llm.api_available:
            return self._simple_format_results(tool_results_summary)
        
        try:
            # 构建智能对话提示词
            system_prompt = """你是DataFlow智能助手，一个专业、友好的AI助手。你可以：

1. 调用专业工具处理特定任务（API密钥获取、表单生成、数据分析、代码生成等）
2. 进行通用智能对话，回答各种问题
3. 记住对话历史，提供连贯的对话体验

当前对话情况：
- 用户刚才提出了一个请求
- 我已经调用了相关工具并获得结果（如果有的话）
- 现在需要基于工具结果和对话历史，自然地回答用户

回答要求：
1. 优先基于工具结果提供准确信息
2. 如果用户问及对话历史，要准确回忆
3. 保持对话自然流畅，像真正的助手
4. 如果没有工具结果，就进行正常的AI对话
5. 用中文回答，语气友好专业

请根据用户问题和上下文，给出最合适的回答。"""

            # 构建对话历史文本
            history_text = self._build_history_text(conversation_history, k=10, clip=300)
            
            # 构建工具结果描述
            tools_info = ""
            if tool_results_summary:
                tools_info_list = []
                for tool_summary in tool_results_summary:
                    tool_name = tool_summary["tool"]
                    result = tool_summary["result"]
                    
                    if tool_name == "APIKey获取工具":
                        if result.get("access_granted"):
                            api_key = result.get("apikey", "")
                            tools_info_list.append(f"成功获取API密钥: {api_key}")
                        else:
                            tools_info_list.append("API密钥获取失败")
                    elif tool_name == "former_agent":
                        if result.get("success"):
                            response = result.get("agent_response", "")
                            tools_info_list.append(f"表单生成结果: {response}")
                        else:
                            tools_info_list.append("表单生成失败")
                    else:
                        tools_info_list.append(f"工具 {tool_name} 执行完成")
                
                tools_info = f"\n\n刚刚执行的工具结果:\n" + "\n".join(tools_info_list)
            
            user_prompt = f"""当前用户问题: {user_input}

对话历史:
{history_text}

工具执行结果:
{tools_info}

请基于以上信息自然地回答用户的问题。如果用户询问对话历史中的内容，请准确回忆。"""

            # 调用LLM生成回复
            llm_service = self.llm._create_llm_service()
            responses = llm_service.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt
            )
            
            if responses and responses[0]:
                return responses[0].strip()
            
        except Exception as e:
            logger.error(f"LLM对话生成失败: {e}")
        
        # fallback到简单格式化
        return self._simple_format_results(tool_results_summary)
    
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
            
            if LANGGRAPH_AVAILABLE and self.agent.compiled_graph:
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
                
            else:
                # 使用Fallback执行器
                logger.info("🔄 使用Fallback执行器")
                output = await self._fallback_execute(initial_state)
            
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
    
    async def _fallback_execute(self, initial_state: Dict[str, Any]) -> str:
        """Fallback执行器（当LangGraph不可用时）"""
        logger.info("执行Fallback逻辑")
        
        # 转换为AgentState对象
        state = AgentState.from_dict(initial_state)
        
        # 执行引导阶段
        state = await self.agent.bootstrap_node(state)
        
        # 决定下一步
        next_action = self.agent.action_forward(state)
        
        if next_action == "execute_tools":
            # 执行工具
            state = await self.agent.execute_tools_node(state)
            # 总结
            state = await self.agent.summarize_node(state)
        elif next_action == "general_conversation":
            # 通用对话
            state = await self.agent.general_conversation_node(state)
        else:
            # 直接总结
            state = await self.agent.summarize_node(state)
        
        # 获取最终输出
        if isinstance(state.agent_outcome, AgentFinish):
            return state.agent_outcome.return_values.get("output", "执行完成")
        else:
            return "执行完成，但未获取到输出"


# 工厂函数
def create_master_agent() -> Tuple[MasterAgent, MasterAgentExecutor]:
    """创建 Master Agent 和执行器"""
    agent = MasterAgent()
    executor = MasterAgentExecutor(agent)
    return agent, executor
