"""
Master Agent 策略模块
处理步数护栏、成本控制、重复检测等策略
"""
from typing import Dict, Any
from dataflow import get_logger

logger = get_logger()


class ExecutionPolicy:
    """执行策略"""
    
    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
    
    def check_step_guard(self, data: Dict[str, Any]) -> bool:
        """步数护栏检查"""
        if data.get('loop_guard') is None:
            data["loop_guard"] = 0
        data["loop_guard"] += 1
        
        current_step = data["loop_guard"]
        if current_step >= self.max_steps:
            logger.info(f"🛑 达到最大步骤数 {self.max_steps}，自动结束")
            data["agent_outcome"] = []  # 清空，让summarize处理
            data["next_action"] = "finish"
            return False  # 需要结束
        
        logger.info(f"🎯 执行步骤 {current_step}/{self.max_steps}")
        return True  # 可以继续
    
    def check_repetition(self, data: Dict[str, Any]) -> bool:
        """重复检测 - 防止无限循环"""
        tool_results = data.get("tool_results", [])
        
        # 检查最近3次是否都是相同工具
        if len(tool_results) >= 3:
            recent_tools = [result.get("tool") for result in tool_results[-3:]]
            if len(set(recent_tools)) == 1 and recent_tools[0] == "former":
                logger.warning("⚠️ 检测到可能的无限循环（连续3次former调用）")
                # 可以在这里添加打断逻辑
                return False
        
        return True
    
    def estimate_cost(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """成本估算"""
        tool_results = data.get("tool_results", [])
        llm_calls = sum(1 for result in tool_results if result.get("tool") in ["former", "pipeline_workflow_agent"])
        
        return {
            "total_tools": len(tool_results),
            "llm_calls": llm_calls,
            "estimated_tokens": llm_calls * 1000,  # 粗略估算
            "current_step": data.get("loop_guard", 0)
        }


class FormPolicy:
    """表单策略"""
    
    @staticmethod
    def should_continue_form(form_session: Dict[str, Any]) -> bool:
        """判断是否应该继续表单收集"""
        if not form_session:
            return False
        
        requires_user_input = form_session.get('requires_user_input', False)
        form_stage = form_session.get('form_stage', '')
        
        return requires_user_input and form_stage == 'parameter_collection'
    
    @staticmethod
    def should_execute_workflow(unified_results: list) -> bool:
        """判断是否应该执行工作流"""
        if not unified_results:
            return False
        
        last_result = unified_results[-1]
        followup = last_result.get("followup")
        
        return (followup and 
                followup.get("needs_followup") and 
                followup.get("suggested_tool"))


# 全局策略实例
execution_policy = ExecutionPolicy()
form_policy = FormPolicy()


def apply_execution_policies(data: Dict[str, Any]) -> bool:
    """应用执行策略，返回是否可以继续"""
    
    # 步数护栏
    if not execution_policy.check_step_guard(data):
        return False
    
    # 重复检测
    if not execution_policy.check_repetition(data):
        logger.warning("🛑 重复检测失败，结束执行")
        data["agent_outcome"] = []
        data["next_action"] = "finish"
        return False
    
    return True


def get_execution_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    """获取执行统计信息"""
    return execution_policy.estimate_cost(data)
