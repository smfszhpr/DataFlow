"""
Master Agent 总结器模块
负责对话总结、执行报告生成等
"""
from typing import Dict, Any, List
from dataflow import get_logger

logger = get_logger()


class Summarizer:
    """总结器 - 负责各种总结和报告生成"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def summarize(self, data: Dict[str, Any]) -> str:
        """生成对话总结"""
        user_input = data.get("input", "")
        tool_results = data.get("intermediate_steps", [])
        
        if not tool_results:
            return f"用户输入：{user_input}\n状态：等待工具执行"
        
        # 统计工具执行情况
        tool_count = len(tool_results)
        tool_types = []
        for action, result in tool_results:
            if hasattr(action, 'tool'):
                tool_types.append(action.tool)
        
        unique_tools = list(set(tool_types))
        
        summary_parts = [
            f"用户输入：{user_input}",
            f"执行了 {tool_count} 个工具步骤",
            f"涉及工具：{', '.join(unique_tools)}",
        ]
        
        # 检查是否有代码生成
        for action, result in tool_results:
            if isinstance(result, dict):
                if any(key in result for key in ['generated_code', 'current_code', 'code']):
                    summary_parts.append("包含代码生成/编辑")
                    break
        
        return " | ".join(summary_parts)
    
    def build_execution_summary(self, tool_results: List) -> Dict[str, Any]:
        """构建执行摘要"""
        if not tool_results:
            return {
                "total_steps": 0,
                "tools_used": [],
                "has_errors": False,
                "has_code": False,
                "summary": "没有工具执行"
            }
        
        total_steps = len(tool_results)
        tools_used = []
        has_errors = False
        has_code = False
        error_messages = []
        
        for action, result in tool_results:
            if hasattr(action, 'tool'):
                tools_used.append(action.tool)
            
            if isinstance(result, dict):
                # 检查错误
                if result.get('success') is False or 'error' in result:
                    has_errors = True
                    if 'error' in result:
                        error_messages.append(str(result['error']))
                
                # 检查代码
                if any(key in result for key in ['generated_code', 'current_code', 'code']):
                    has_code = True
        
        unique_tools = list(set(tools_used))
        
        summary = f"执行了 {total_steps} 步，使用工具：{', '.join(unique_tools)}"
        if has_code:
            summary += "，包含代码操作"
        if has_errors:
            summary += f"，发现 {len(error_messages)} 个错误"
        
        return {
            "total_steps": total_steps,
            "tools_used": unique_tools,
            "has_errors": has_errors,
            "has_code": has_code,
            "error_messages": error_messages,
            "summary": summary
        }
