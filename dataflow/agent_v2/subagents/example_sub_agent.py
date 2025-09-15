"""
DataFlow Agent-V2 示例 SubAgent
使用类似 myscalekb-agent-plugin 的风格和装饰器
"""
from typing import TypedDict, List

from langchain.agents import create_openai_tools_agent
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.runnables import RunnableConfig

from dataflow.agent_v2.base.core import SubAgent, node, edge, entry, conditional_edge, GraphBuilder, SubAgentRegistry
from dataflow.agent_v2.master.agent import AgentState
from dataflow.agent_v2.llm_client import get_llm_client


class ExampleSubAgent(SubAgent):
    """示例子代理，展示如何使用装饰器风格的开发"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化工具和其他组件
        self.llm = self.llm or get_llm_client()
    
    @classmethod
    def name(cls) -> str:
        return "example_sub_agent"
    
    @classmethod
    def description(cls) -> str:
        return "An example sub-agent that demonstrates the decorator style"
    
    @classmethod
    def state_definition(cls) -> type:
        """定义自定义状态"""
        from dataflow.agent_v2.master.agent import AgentState
        from dataflow.agent_v2.base.core import AgentMetadata
        from typing import Optional
        
        class ExampleState(AgentState):
            # 添加示例特定的状态字段
            processing_stage: str = "init"
            intermediate_results: List[str] = []
            agent_metadata: Optional[AgentMetadata] = None
        
        return ExampleState
    
    @node
    @entry
    @edge(target_node="process_input")
    async def entry_point(self, data):
        """入口点节点，决定如何处理输入"""
        
        # 初始化处理阶段
        data.processing_stage = "entry"
        data.intermediate_results = []
        
        # 简单的决策逻辑
        user_input = getattr(data, "input", "")
        
        data.processing_stage = "entry_completed"
        data.intermediate_results = [f"Entry processed: {user_input[:50]}..."]
        
        return data
    
    @node
    @edge(target_node="finalize_results")
    async def process_input(self, data):
        """处理用户输入"""
        
        user_input = getattr(data, "input", "")
        intermediate_results = getattr(data, "intermediate_results", [])
        
        # 模拟处理过程
        processed_result = f"Processed: {user_input}"
        intermediate_results.append(processed_result)
        
        # 如果有 LLM，可以使用它进行处理
        if self.llm:
            # 这里可以添加实际的 LLM 调用
            llm_result = f"LLM analyzed: {user_input}"
            intermediate_results.append(llm_result)
        
        data.processing_stage = "processing_completed"
        data.intermediate_results = intermediate_results
        
        return data
    
    @node
    async def finalize_results(self, data):
        """最终化结果"""
        
        intermediate_results = getattr(data, "intermediate_results", [])
        
        # 生成最终输出
        final_output = "\\n".join([
            "=== Example SubAgent Results ===",
            f"Processing completed with {len(intermediate_results)} steps:",
            *[f"- {result}" for result in intermediate_results],
            "=== End Results ==="
        ])
        
        # 直接设置到 agent_outcome 而不是返回 AgentFinish
        data.agent_outcome = self._make_agent_finish(final_output)
        
        return data


# 创建工厂函数，类似 myscalekb-agent-plugin 的使用方式
def create_example_sub_agent(llm=None, **kwargs):
    """创建示例子代理实例"""
    return ExampleSubAgent(llm=llm, **kwargs)


# 注册子代理
SubAgentRegistry.register(ExampleSubAgent)
