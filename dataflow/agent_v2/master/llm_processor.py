"""
Master Agent LLM 处理模块
负责与LLM的交互、对话生成、用户需求分析等
"""
from typing import Dict, Any, List
import json
from dataflow import get_logger

logger = get_logger()


class LLMProcessor:
    """LLM处理器 - 负责所有与LLM相关的逻辑"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def build_history_text(self, conversation_history: List[Dict[str, str]], k: int = 8, clip: int = 200) -> str:
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
    
    async def generate_conversation_response(self, data: Dict[str, Any]) -> str:
        """基于工具执行结果和对话历史生成智能响应 - 让大模型对所有工具结果进行智能总结"""
        user_input = data.get("input", "")
        conversation_history = data.get("conversation_history", [])
        
        # 构建详细的工具执行结果 - 通用化处理，不硬编码特定字段
        detailed_tool_results = []
        tool_output_summary = {}  # 按工具类型汇总输出
        
        for i, (action, result) in enumerate(data.get("intermediate_steps", [])):
            tool_name = action.tool
            step_num = i + 1
            
            # 解析工具结果并收集详细信息
            if isinstance(result, dict):
                detailed_tool_results.append({
                    "step": step_num,
                    "tool": tool_name,
                    "result": result
                })
                
                # 通用化地收集每个工具的输出
                if tool_name not in tool_output_summary:
                    tool_output_summary[tool_name] = []
                
                # 动态提取结果中的重要信息
                important_fields = []
                for key, value in result.items():
                    # 特殊处理代码类字段，不限制长度
                    if key in ['current_code', 'generated_code', 'code', 'output']:
                        if isinstance(value, str) and value.strip():
                            important_fields.append(f"{key}: {value}")
                    # 其他字段保持原有限制
                    elif isinstance(value, (str, int, float, bool)) and len(str(value)) < 200:
                        important_fields.append(f"{key}: {value}")
                
                tool_output_summary[tool_name].append(" | ".join(important_fields))
        
        # 构建详细执行报告
        execution_report = self._build_detailed_execution_report(detailed_tool_results, tool_output_summary)
        
        # 构建历史上下文
        history_text = self.build_history_text(conversation_history)
        
        # 构建LLM提示词
        llm_prompt = f"""你是一个专业的AI助手，需要基于工具执行结果为用户生成智能、专业的回复。

当前用户输入: {user_input}

对话历史:
{history_text}

工具执行详情:
{execution_report}

请根据以上信息，生成一个智能、有用的回复。要求：
1. 直接回答用户的问题或需求
2. 总结工具执行的关键结果
3. 如果有生成的代码，要重点展示
4. 如果遇到错误，要说明原因和建议
5. 语气友好、专业
6. 不要重复工具的详细执行过程，只总结关键结果"""

        try:
            # 使用LLMClient的异步调用方法
            response = await self.llm.call_llm_async(
                system_prompt="你是一个专业的AI助手，需要基于工具执行结果为用户生成智能、专业的回复。",
                user_prompt=llm_prompt
            )
            
            response_content = response.get("content", "") if isinstance(response, dict) else str(response)
            return response_content.strip() if response_content else "抱歉，我无法生成合适的回复，请稍后再试。"
            
        except Exception as e:
            logger.error(f"LLM生成对话回复失败: {e}")
            return f"抱歉，在生成回复时遇到问题：{str(e)}"
    
    def _build_detailed_execution_report(self, detailed_tool_results: List[Dict], tool_output_summary: Dict) -> str:
        """构建详细的执行报告"""
        if not detailed_tool_results:
            return "没有工具执行结果。"
        
        report_lines = []
        report_lines.append("=== 工具执行报告 ===")
        
        # 按步骤显示详细结果
        for result in detailed_tool_results:
            step_num = result["step"]
            tool_name = result["tool"]
            tool_result = result["result"]
            
            report_lines.append(f"\n步骤 {step_num} - 工具: {tool_name}")
            
            # 动态显示工具结果的重要字段
            if isinstance(tool_result, dict):
                for key, value in tool_result.items():
                    if key in ['success', 'error', 'message', 'output']:
                        if isinstance(value, (str, int, float, bool)):
                            report_lines.append(f"  {key}: {value}")
                    elif key in ['current_code', 'generated_code', 'code'] and value:
                        # 代码字段特殊处理：显示前几行和总行数
                        code_lines = str(value).split('\n')
                        if len(code_lines) > 5:
                            preview = '\n'.join(code_lines[:3])
                            report_lines.append(f"  {key} (共{len(code_lines)}行): \n{preview}\n  ...")
                        else:
                            report_lines.append(f"  {key}: \n{value}")
        
        # 按工具类型汇总
        if tool_output_summary:
            report_lines.append("\n=== 工具输出汇总 ===")
            for tool_name, outputs in tool_output_summary.items():
                if outputs:
                    report_lines.append(f"\n{tool_name}:")
                    for i, output in enumerate(outputs, 1):
                        if output:  # 只显示非空输出
                            report_lines.append(f"  执行 {i}: {output}")
        
        return "\n".join(report_lines)
    
    def analyze_user_needs(self, user_input: str, tool_results: List[Dict], available_tools: List[str] = None) -> Dict[str, Any]:
        """分析用户需求并决定下一步动作 - 从实际可用工具中选择"""
        
        # 如果没有提供可用工具，使用默认工具列表
        if not available_tools:
            available_tools = [
                'APIKey获取工具', 'former', 'code_workflow_agent', 
                'pipeline_workflow_agent', 'sleep_tool', 'csv_profile', 
                'csv_detect_time_columns', 'csv_vega_spec', 'code_static_check', 
                'code_test_stub', 'local_index_build', 'local_index_query'
            ]
        
        try:
            # 构建工具列表描述
            tools_desc = "\n".join([f"- {tool}" for tool in available_tools])
            
            analysis_prompt = f"""分析以下用户输入，判断用户需要什么帮助：

用户输入: {user_input}

工具执行历史: {len(tool_results)} 个工具已执行

可用工具列表（只能从中选择）:
{tools_desc}

请严格按照以下JSON格式输出决策：
{{
    "should_continue": true/false,
    "reasons": ["原因1", "原因2"],
    "next_action": {{
        "tool": "必须从可用工具列表中选择",
        "tool_input": {{键值对}}
    }} 或 null,
    "analysis": {{
        "user_intent": "用户意图描述",
        "confidence": 0.8,
        "llm_decision": {{
            "reasoning": "决策理由",
            "risk_assessment": "风险评估",
            "reason": "选择该工具的原因"
        }}
    }}
}}

重要规则：
1. 如果用户询问general问题或需要对话，设置should_continue=false
2. 如果用户需要代码生成，选择'code_workflow_agent'工具
3. 如果用户需要数据管道，选择'pipeline_workflow_agent'工具  
4. 如果用户需要表单收集，选择'former'工具
5. tool字段必须完全匹配可用工具列表中的名称"""

            response = self.llm.call_llm(
                system_prompt="你是一个专业的需求分析助手。",
                user_prompt=analysis_prompt
            )
            
            # 解析JSON响应
            try:
                response_content = response.get("content", "") if isinstance(response, dict) else str(response)
                analysis = json.loads(response_content.strip())
                
                # 验证必需字段
                required_fields = ["should_continue", "reasons"]
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = False if field == "should_continue" else ["LLM分析不完整"]
                
                return analysis
                
            except json.JSONDecodeError as je:
                logger.error(f"LLM分析响应JSON解析失败: {je}")
                logger.error(f"原始响应: {response}")
                return {
                    "should_continue": False,
                    "reasons": ["LLM响应格式错误"],
                    "next_action": None,
                    "analysis": {"error": f"JSON解析错误: {str(je)}"}
                }
                
        except Exception as e:
            logger.error(f"用户需求分析失败: {e}")
            return {
                "should_continue": False,
                "reasons": [f"分析过程出错: {str(e)}"],
                "next_action": None,
                "analysis": {"error": str(e)}
            }
