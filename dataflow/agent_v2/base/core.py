"""
DataFlow Agent-V2 Base Classes
基于 myscalekb-agent 架构的基础组件
"""
from abc import ABC, abstractmethod
from typing import Type, Any, Dict, List, Optional, Union
from pydantic import BaseModel
from enum import Enum
import functools


class NodeType(Enum):
    """节点类型枚举"""
    ENTRY = "entry"
    PROCESSOR = "processor"
    CONDITIONAL = "conditional"
    END = "end"


class BaseTool(ABC):
    """基础工具类，模仿 myscalekb-agent-base.tool.BaseTool"""
    
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
    
    @abstractmethod
    def params(self) -> Type[BaseModel]:
        """工具参数模型"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        """执行工具"""
        pass


class GraphBuilder:
    """图构建器，管理节点和边的连接"""
    
    END = "END"
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.entry_point = None
    
    def add_node(self, name: str, func, node_type: NodeType = NodeType.PROCESSOR):
        """添加节点"""
        self.nodes[name] = {
            'func': func,
            'type': node_type
        }
    
    def add_edge(self, from_node: str, to_node: str):
        """添加边"""
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)
    
    def add_conditional_edge(self, from_node: str, condition_func, edge_map: Dict[str, str]):
        """添加条件边"""
        self.edges[from_node] = {
            'type': 'conditional',
            'condition': condition_func,
            'map': edge_map
        }
    
    def set_entry_point(self, node_name: str):
        """设置入口点"""
        self.entry_point = node_name


class SubAgent(ABC):
    """子代理基类，模仿 myscalekb-agent 的 SubAgent 模式"""
    
    def __init__(self):
        self.graph_builder = GraphBuilder()
        self.tools = []
        self._setup_graph()
    
    @abstractmethod
    def _setup_graph(self):
        """设置图结构，子类需要实现"""
        pass
    
    @abstractmethod
    def state_definition(self) -> Type[BaseModel]:
        """状态定义，返回 TypedDict 或 BaseModel"""
        pass
    
    def add_tool(self, tool: BaseTool):
        """添加工具"""
        self.tools.append(tool)
    
    def get_tools(self) -> List[BaseTool]:
        """获取所有工具"""
        return self.tools


def node(node_type: NodeType = NodeType.PROCESSOR):
    """节点装饰器"""
    def decorator(func):
        func._node_type = node_type
        return func
    return decorator


def entry(func):
    """入口点装饰器"""
    func._node_type = NodeType.ENTRY
    func._is_entry = True
    return func


def conditional_edge(condition_func):
    """条件边装饰器"""
    def decorator(func):
        func._condition_func = condition_func
        func._node_type = NodeType.CONDITIONAL
        return func
    return decorator


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
