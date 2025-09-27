"""
Master Agent 工具执行器模块
负责工具的实际执行和结果处理
"""
from typing import Dict, Any, List, Tuple
from langchain_core.agents import AgentAction
from dataflow import get_logger
from .adapters.tool_adapter import ToolResultAdapter

logger = get_logger()


class ToolExecutor:
    """工具执行器 - 负责工具执行和结果处理"""
    
    def __init__(self, all_tools: List):
        self.all_tools = all_tools
        self.tool_dict = {tool.name: tool for tool in all_tools}
        self.tool_adapter = ToolResultAdapter()
    
    async def execute_tools(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具并返回更新后的数据"""
        # 首先检查agent_outcome（优先）
        agent_outcome = data.get("agent_outcome")
        if agent_outcome:
            # 如果是列表，取第一个
            if isinstance(agent_outcome, list) and agent_outcome:
                action = agent_outcome[0]
                tool_input = action.tool_input
            elif hasattr(agent_outcome, 'tool'):
                action = agent_outcome  
                tool_input = agent_outcome.tool_input
            else:
                logger.warning(f"execute_tools: 无法解析agent_outcome {agent_outcome}")
                return data
        else:
            # 回退到next_action
            next_action = data.get("next_action")
            if not next_action:
                logger.warning("execute_tools: 没有找到 agent_outcome 或 next_action")
                return data
            
            # 解析动作
            action, tool_input = self._parse_action(next_action)
            if not action:
                logger.warning(f"execute_tools: 无法解析动作 {next_action}")
                return data
        
        # 执行工具
        result = await self._execute_single_tool(action, tool_input)
        
        # 更新数据
        intermediate_steps = data.get("intermediate_steps", [])
        intermediate_steps.append((action, result))
        data["intermediate_steps"] = intermediate_steps
        
        # 清除 agent_outcome 和 next_action
        data["agent_outcome"] = []
        data.pop("next_action", None)
        
        logger.info(f"工具执行完成: {action.tool}, 结果类型: {type(result)}")
        return data
    
    def _parse_action(self, next_action: Any) -> Tuple[AgentAction, Dict[str, Any]]:
        """解析 next_action 为 AgentAction 和 tool_input"""
        if isinstance(next_action, dict):
            tool_name = next_action.get("tool")
            tool_input = next_action.get("tool_input", {})
            
            if tool_name:
                action = AgentAction(
                    tool=tool_name,
                    tool_input=tool_input,
                    log=f"调用工具: {tool_name}"
                )
                return action, tool_input
        
        elif isinstance(next_action, AgentAction):
            return next_action, next_action.tool_input
        
        return None, None
    
    async def _execute_single_tool(self, action: AgentAction, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个工具并统一结果格式"""
        tool_name = action.tool
        
        if tool_name not in self.tool_dict:
            error_msg = f"工具 '{tool_name}' 不存在，可用工具: {list(self.tool_dict.keys())}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        tool = self.tool_dict[tool_name]
        
        try:
            logger.info(f"开始执行工具: {tool_name}, 输入: {tool_input}")
            
            # 执行工具
            if hasattr(tool, 'ainvoke'):
                # 异步工具
                raw_result = await tool.ainvoke(tool_input)
            elif hasattr(tool, 'invoke'):
                # 同步工具
                raw_result = tool.invoke(tool_input)
            elif callable(tool):
                # 直接可调用
                raw_result = tool(tool_input)
            else:
                error_msg = f"工具 {tool_name} 不可调用"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            logger.info(f"工具 {tool_name} 原始结果类型: {type(raw_result)}")
            
            # 使用适配器统一结果格式
            unified_result = self.tool_adapter.adapt_tool_result(raw_result, tool_name)
            
            logger.info(f"工具 {tool_name} 统一结果: success={unified_result.get('success')}")
            return unified_result
            
        except Exception as e:
            error_msg = f"工具 {tool_name} 执行失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}
    
    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return list(self.tool_dict.keys())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """获取工具信息"""
        if tool_name not in self.tool_dict:
            return {"error": f"工具 {tool_name} 不存在"}
        
        tool = self.tool_dict[tool_name]
        
        info = {
            "name": tool_name,
            "description": getattr(tool, 'description', ''),
        }
        
        # 尝试获取参数信息
        if hasattr(tool, 'args_schema'):
            info["args_schema"] = str(tool.args_schema)
        
        return info
