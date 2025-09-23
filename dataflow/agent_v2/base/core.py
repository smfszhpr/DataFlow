"""
DataFlow Agent-V2 Base Classes
基于 myscalekb-agent 架构的基础组件，实现类似 myscalekb-agent-base 的功能
"""
import asyncio
import datetime
import functools
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Type, Any, Dict, List, Optional, Union, TypedDict, Callable
from pydantic import BaseModel
from enum import Enum

from langchain_core.agents import AgentFinish
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph
from langchain_core.runnables.base import RunnableLike

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """节点类型枚举"""
    ENTRY = "entry"
    PROCESSOR = "processor"
    CONDITIONAL = "conditional"
    END = "end"


class AgentMetadata(BaseModel):
    """代理元数据，类似 myscalekb-agent-base.schemas.agent_metadata"""
    name: str
    step: str
    timestamp: datetime.datetime = datetime.datetime.now()


class BaseTool(ABC):
    """基础工具类，模仿 myscalekb-agent-base.tool.BaseTool，支持前置/后置工具架构"""
    
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """工具名称"""
        pass
    
    @classmethod
    @abstractmethod
    def description(cls) -> str:
        """工具描述"""
        pass
    
    @classmethod
    def prerequisite_tools(cls) -> List[str]:
        """前置工具列表 - 建议在调用此工具前先调用的工具"""
        return []
    
    @classmethod
    def suggested_followup_tools(cls) -> List[str]:
        """建议的后置工具列表 - 此工具完成后建议调用的工具"""
        return []
    
    @classmethod
    def get_tool_metadata(cls) -> Dict[str, Any]:
        """获取工具的完整元数据"""
        return {
            "name": cls.name(),
            "description": cls.description(),
            "prerequisite_tools": cls.prerequisite_tools(),
            "suggested_followup_tools": cls.suggested_followup_tools()
        }
    
    @abstractmethod
    def params(self) -> Type[BaseModel]:
        """工具参数模型"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        """执行工具"""
        pass


def node(func):
    """添加新节点到图中，类似 myscalekb-agent-base 的 @node 装饰器
    
    示例:
        @node
        async def run_query(self, data):
            # 代码逻辑
    
    添加以下横切关注点:
    - 记录节点执行的开始和结束
    - 性能计时
    - 代理元数据创建
    - 一致的配置处理
    """
    
    @functools.wraps(func)
    async def wrapper(self, data):
        # 处理不同的数据类型
        if hasattr(data, 'get'):
            # 字典类型
            trace_id = data.get("trace_id", "unknown")
        else:
            # Pydantic 模型类型
            trace_id = getattr(data, "trace_id", "unknown")
        
        agent_metadata = AgentMetadata(name=self.name(), step=func.__name__)
        
        logger.info(
            "QueryTrace[%s] Beginning to run %s.%s step.",
            trace_id,
            self.name(),
            func.__name__,
        )
        start_time = datetime.datetime.now()
        
        try:
            # 为 Pydantic 模型添加元数据
            if hasattr(data, 'agent_metadata'):
                # 对于 Pydantic 模型，使用 setattr 或直接赋值
                data.agent_metadata = agent_metadata
            elif hasattr(data, '__dict__'):
                # 对于普通对象，直接设置属性
                data.agent_metadata = agent_metadata
            
            # 执行原始函数
            result = await func(self, data)
            
            duration = (datetime.datetime.now() - start_time).total_seconds()
            
            logger.info(
                "QueryTrace[%s] Completed running %s.%s step, duration: %.3f seconds",
                trace_id,
                self.name(),
                func.__name__,
                duration,
            )
            
            if not result:
                if hasattr(data, 'agent_metadata'):
                    data.agent_metadata = agent_metadata
                    return data
                elif hasattr(data, '__dict__'):
                    data.agent_metadata = agent_metadata
                    return data
                else:
                    return {"agent_metadata": agent_metadata}
            elif isinstance(result, dict):
                result["agent_metadata"] = agent_metadata
                
            return result
            
        except Exception as e:
            logger.error(
                "QueryTrace[%s] Error in %s step: %s", trace_id, func.__name__, str(e)
            )
            raise
    
    wrapper._is_node = True
    return wrapper


def edge(target_node: str):
    """在源节点上注释以建立到目标节点的单向边
    
    示例:
        @node
        @edge(target_node="summary_contexts")
        async def retrieve_contexts(self, data):
            # 代码逻辑
        
        @node
        @edge(target_node=GraphBuilder.END)
        async def summary_contexts(self, data):
            # 代码逻辑
    """
    
    def decorator(func):
        func._is_edge = True
        func._target = target_node
        return func
    
    return decorator


def conditional_edge(path: RunnableLike, path_map: dict):
    """在源节点上注释并定义不同的条件以到达不同的节点
    
    示例:
        @conditional_edge(
            path=lambda data: "continue" if should_continue(data) else "end",
            path_map={"continue": "retrieve_contexts", "end": GraphBuilder.END}
        )
        def should_continue(self, data):
            # 条件逻辑
    """
    
    def decorator(func):
        func._is_conditional_edge = True
        func._path = path
        func._path_map = path_map
        return func
    
    return decorator


def entry(func):
    """将节点标记为图的入口点
    一个图只能有一个入口点，此注释不能与 @conditional_entry 一起使用
    
    示例:
        @node
        @entry
        async def run_query(self, data):
            # 代码逻辑
    """
    func._is_entry = True
    return func


def conditional_entry(path_map: dict):
    """将边标记为图的入口点
    某些特殊图需要在入口处确定分支到哪个节点
    
    示例:
        @conditional_entry(path_map={"form": "init_form", "query": "run_query"})
        async def entry(self, data):
            # 代码逻辑
    """
    
    def decorator(func):
        func._is_conditional_entry = True
        func._path_map = path_map
        return func
    
    return decorator


class GraphBuilder:
    """图构建器，类似 myscalekb-agent-base.graph_builder.GraphBuilder"""
    
    END = "__end__"
    
    def _build_graph(self, state_definition: type, compiled: bool = True):
        """构建 LangGraph 图"""
        workflow = StateGraph(state_definition)
        
        # 首先添加所有节点
        node_methods = {}
        entry_points = []
        
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_is_node"):
                workflow.add_node(name, method)
                node_methods[name] = method
                logger.debug(f"Added node: {name}")
                
                if hasattr(method, "_is_entry"):
                    entry_points.append(name)
                    logger.debug(f"Marked entry point: {name}")
        
        # 然后添加边和条件边
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_is_edge"):
                target = method._edge_target if hasattr(method, '_edge_target') else method._target
                workflow.add_edge(name, target)
                logger.debug(f"Added edge from {name} to {target}")
            
            if hasattr(method, "_is_conditional_edge"):
                # 条件边应该从对应的节点开始
                source_node = name.replace('_decision', '') if name.endswith('_decision') else name
                if source_node in node_methods:
                    workflow.add_conditional_edges(
                        source=source_node, path=method._path, path_map=method._path_map
                    )
                    logger.debug(f"Added conditional edge from {source_node}")
                else:
                    logger.warning(f"Conditional edge {name} has no corresponding node")
        
        # 设置入口点
        if entry_points:
            workflow.set_entry_point(entry_points[0])
            logger.debug(f"Set entry point: {entry_points[0]}")
        
        if compiled:
            graph = workflow.compile()
            logger.info("Build graph successfully")
            return graph
        
        logger.info("Return workflow not compiled.")
        return workflow


class SubAgent(ABC, GraphBuilder):
    """子代理抽象基类，创建可以构建和处理图的专用子代理
    类似 myscalekb-agent-base.sub_agent.SubAgent
    
    属性:
        llm: 用于自然语言处理任务的底层聊天模型
        embedding_model: 用于从文本生成向量嵌入的模型
        knowledge_scopes: 代理可用的知识域列表
    """
    
    def __init__(self, llm=None, embedding_model=None, knowledge_scopes=None, *args, **kwargs):
        """初始化新的 SubAgent 实例
        
        Args:
            llm: 聊天模型实例
            embedding_model: 嵌入模型实例
            knowledge_scopes: 知识域列表
            *args: 可变长度参数列表，用于未来扩展
            **kwargs: 任意关键字参数，用于未来扩展
        """
        logger.info("Initializing SubAgent - %s", self.__class__.__name__)
        
        self.llm = llm
        self.embedding_model = embedding_model
        self.knowledge_scopes = knowledge_scopes or []
        
        # 收集装饰器标记的方法
        self._collect_decorated_methods()
    
    @classmethod
    def register(cls, master_builder, llm=None, embedding_model=None, knowledge_scopes=None, *args, **kwargs):
        """工厂方法，用于创建 SubAgent 实例以确保图构建器工作"""
        agent = cls(
            llm=llm, 
            embedding_model=embedding_model, 
            knowledge_scopes=knowledge_scopes,
            *args, **kwargs
        )
        agent._finalize_initialization(master_builder)
        return agent
    
    def _finalize_initialization(self, master_builder):
        """完成初始化，注册到主构建器"""
        forward_func = StructuredTool(
            name=self.name(),
            description=self.description(),
            args_schema=self.forward_schema(),
            func=self.__placeholder_func,
        )
        
        graph = self.graph()
        master_builder.register_sub_agent(forward_func, (self.name(), graph))
    
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """子类必须实现此方法以向 MasterAgent 注册"""
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def description(cls) -> str:
        """子类必须实现此方法以向 MasterAgent 注册"""
        raise NotImplementedError
    
    @classmethod
    def state_definition(cls) -> type:
        """子类必须实现此方法以定义图状态数据
        当然，开发者可以使用默认状态 - `AgentState`
        """
        # 动态导入以避免循环导入
        try:
            from dataflow.agent_v2.master.agent import AgentState
            return AgentState
        except ImportError:
            # 如果无法导入，使用基本的 TypedDict
            class BasicAgentState(TypedDict):
                input: str
                output: Optional[str]
            return BasicAgentState
    
    @classmethod
    def forward_schema(cls) -> Type[BaseModel]:
        """子类可以继承此方法以实现 Forward 可以包含的参数
        此结构可以从 state["forward_args"] 获取
        
        默认为空
        """
        return cls.EmptySchema
    
    def graph(self):
        """子类可以使用此方法自定义图的实现，而不是使用注释来定义它"""
        return self._build_graph(self.state_definition())
    
    class EmptySchema(BaseModel):
        """没有参数的工具的空模式"""
        pass
    
    def __placeholder_func(self, **kwargs):
        raise NotImplementedError("This tool is a placeholder and cannot be called.")
    
    @staticmethod
    def _get_tool_args(agent_outcome) -> dict:
        """工具方法：从先前的 agent_outcome 获取工具调用参数"""
        from langchain.agents.output_parsers.tools import ToolAgentAction
        
        if isinstance(agent_outcome, ToolAgentAction):
            return agent_outcome.tool_input
        if isinstance(agent_outcome, list) and isinstance(agent_outcome[0], ToolAgentAction):
            return agent_outcome[0].tool_input
        
        return {}
    
    @staticmethod
    def _make_agent_finish(output: Any) -> AgentFinish:
        """创建 AgentFinish 对象"""
        return AgentFinish(return_values={"output": output}, log="")
    
    def _collect_decorated_methods(self):
        """收集装饰器标记的方法"""
        self._node_methods = []
        self._edge_methods = []
        self._conditional_edge_methods = []
        self._entry_methods = []
        
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, '_is_node'):
                self._node_methods.append(method)
            if hasattr(method, '_is_edge'):
                self._edge_methods.append(method)
            if hasattr(method, '_is_conditional_edge'):
                self._conditional_edge_methods.append(method)
            if hasattr(method, '_is_entry'):
                self._entry_methods.append(method)
    
    def build_graph(self, agent=None):
        """构建图 - 提供与测试兼容的接口"""
        if agent is None:
            agent = self
        return self._build_graph(self.state_definition())
    
    def graph(self):
        """获取代理的图实例"""
        return self._build_graph(self.state_definition())


class SubAgentRegistry:
    """子代理注册表，管理所有可用的子代理"""
    _agents = {}
    
    @classmethod
    def register(cls, agent_class):
        """注册代理类"""
        agent_name = agent_class.name() if hasattr(agent_class, 'name') else agent_class.__name__
        cls._agents[agent_name] = agent_class
        logger.info(f"Registered agent: {agent_name}")
    
    @classmethod
    def get_agent(cls, name: str):
        """通过名称获取代理类"""
        return cls._agents.get(name)
    
    @classmethod
    def filter_enabled_agents(cls, disabled_agent_names: List[str]) -> List[Type[SubAgent]]:
        """通过类名获取启用的代理"""
        enabled_agents = []
        
        for name, agent_class in cls._agents.items():
            if name not in disabled_agent_names:
                enabled_agents.append(agent_class)
        
        return enabled_agents
    
    @classmethod
    def get_all_agents(cls) -> Dict[str, Type[SubAgent]]:
        """获取所有注册的代理"""
        return cls._agents.copy()


class Prompt(ABC):
    """提示模板基类"""
    
    def prompt_template(self, messages: List, with_history: bool = True, 
                       with_user_query: bool = True, with_agent_scratchpad: bool = False):
        """构建提示模板"""
        # 这里应该返回 ChatPromptTemplate，暂时返回字典
        return {
            'messages': messages,
            'with_history': with_history,
            'with_user_query': with_user_query,
            'with_agent_scratchpad': with_agent_scratchpad
        }
