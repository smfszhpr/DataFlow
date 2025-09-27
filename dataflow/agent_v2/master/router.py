"""
Master Agent 路由器
处理决策路由和节点转发
"""
from typing import Dict, Any
from dataflow import get_logger

logger = get_logger()


class MasterRouter:
    """Master Agent路由器"""
    
    @staticmethod
    def planner_router(data: Dict[str, Any]) -> str:
        """规划器路由器 - 决定下一个节点"""
        next_action = data.get("next_action")
        result = "continue" if next_action == "continue" else "finish"
        logger.info(f"🔀 路由决策: {next_action} -> {result}")
        return result
    
    @staticmethod
    def action_forward_router(data: Dict[str, Any]) -> str:
        """动作转发路由器"""
        agent_outcome = data.get("agent_outcome", [])
        
        if not agent_outcome or (isinstance(agent_outcome, list) and len(agent_outcome) == 0):
            logger.info(f"🔀 Action Forward开始，agent_outcome类型: {type(agent_outcome)}")
            logger.info(f"🔀 Agent outcome内容: {agent_outcome}")
            logger.info("🔄 工具执行完成，回到planner继续决策")
            return "planner"
        else:
            logger.info("🔄 发现待执行动作，流转到execute_tools")
            return "execute_tools"


def decide_next_hop(data: Dict[str, Any], current_node: str) -> str:
    """统一的决策跳转函数"""
    
    if current_node == "planner":
        return MasterRouter.planner_router(data)
    elif current_node == "action_forward":
        return MasterRouter.action_forward_router(data)
    else:
        logger.warning(f"未知节点路由: {current_node}")
        return "finish"
