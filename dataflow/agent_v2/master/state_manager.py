"""
Master Agent 状态管理器模块
负责AgentState的创建、更新和维护
"""
from typing import Dict, Any, List
from dataclasses import dataclass, field
from dataflow import get_logger

logger = get_logger()


@dataclass
class AgentState:
    """Agent状态类 - 保持与原来完全一致"""
    input: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # 工具执行相关
    next_action: Dict[str, Any] = None
    intermediate_steps: List = field(default_factory=list)
    
    # Agent运行状态
    iteration: int = 0
    max_iterations: int = 15
    force_finish: bool = False
    
    # 输出和总结
    agent_outcome: str = ""
    execution_summary: str = ""


class StateManager:
    """状态管理器 - 负责AgentState的管理和更新"""
    
    def __init__(self):
        pass
    
    def create_initial_state(self, user_input: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """创建初始状态"""
        state = AgentState(
            input=user_input,
            conversation_history=conversation_history or [],
            iteration=0,
            max_iterations=15,
            force_finish=False
        )
        
        # 转换为字典格式（LangGraph需要）
        state_dict = {
            "input": state.input,
            "conversation_history": state.conversation_history,
            "next_action": state.next_action,
            "intermediate_steps": state.intermediate_steps,
            "iteration": state.iteration,
            "max_iterations": state.max_iterations,
            "force_finish": state.force_finish,
            "agent_outcome": state.agent_outcome,
            "execution_summary": state.execution_summary,
        }
        
        logger.info(f"创建初始状态: input='{user_input}', history_len={len(conversation_history or [])}")
        return state_dict
    
    def update_iteration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """更新迭代计数"""
        current_iteration = data.get("iteration", 0)
        data["iteration"] = current_iteration + 1
        
        # 检查是否超过最大迭代次数
        max_iterations = data.get("max_iterations", 15)
        if data["iteration"] >= max_iterations:
            data["force_finish"] = True
            logger.warning(f"达到最大迭代次数 {max_iterations}，强制结束")
        
        return data
    
    def set_agent_outcome(self, data: Dict[str, Any], outcome: str) -> Dict[str, Any]:
        """设置代理输出结果"""
        data["agent_outcome"] = outcome
        logger.info(f"设置代理结果: {outcome[:100]}..." if len(outcome) > 100 else outcome)
        return data
    
    def set_execution_summary(self, data: Dict[str, Any], summary: str) -> Dict[str, Any]:
        """设置执行摘要"""
        data["execution_summary"] = summary
        return data
    
    def clear_next_action(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """清除下一个动作"""
        data["next_action"] = None
        return data
    
    def set_next_action(self, data: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """设置下一个动作"""
        data["next_action"] = action
        logger.info(f"设置下一动作: {action.get('tool', 'unknown')} - {action.get('tool_input', {})}")
        return data
    
    def should_continue(self, data: Dict[str, Any]) -> bool:
        """判断是否应该继续执行"""
        # 强制结束
        if data.get("force_finish"):
            return False
        
        # 有下一个动作就继续
        if data.get("next_action"):
            return True
        
        # 检查迭代次数
        iteration = data.get("iteration", 0)
        max_iterations = data.get("max_iterations", 15)
        if iteration >= max_iterations:
            return False
        
        return False
    
    def get_state_summary(self, data: Dict[str, Any]) -> str:
        """获取状态摘要"""
        iteration = data.get("iteration", 0)
        max_iterations = data.get("max_iterations", 15)
        steps_count = len(data.get("intermediate_steps", []))
        has_next_action = bool(data.get("next_action"))
        force_finish = data.get("force_finish", False)
        
        summary = f"迭代 {iteration}/{max_iterations}, 已执行 {steps_count} 步"
        
        if has_next_action:
            next_tool = data["next_action"].get("tool", "unknown")
            summary += f", 下一步: {next_tool}"
        
        if force_finish:
            summary += ", 强制结束"
        
        return summary
