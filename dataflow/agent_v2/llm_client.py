"""
LLM客户端，基于现有DataFlow项目的LLM调用方式
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional

from dataflow.serving import APILLMServing_request
from dataflow.agent.eventengine.config_manager import get_llm_config

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM客户端，使用DataFlow现有的LLM服务"""
    
    def __init__(self):
        self.llm_config = get_llm_config()
        self.api_available = bool(self.llm_config.api_key and self.llm_config.api_url)
        
        if not self.api_available:
            logger.warning("LLM API配置不可用，将使用fallback模式")
        
        logger.info(f"LLM客户端初始化完成，使用模型: {self.llm_config.model}, API可用: {self.api_available}")
    
    def _create_llm_service(self) -> APILLMServing_request:
        """创建LLM服务实例"""
        # 设置环境变量（为了兼容现有API）
        os.environ["DF_API_KEY"] = self.llm_config.api_key
        
        return APILLMServing_request(
            api_url=self.llm_config.api_url,
            key_name_of_api_key="DF_API_KEY",
            model_name=self.llm_config.model
        )
    
    def analyze_user_intent(self, user_input: str, available_tools: List[str]) -> Dict[str, Any]:
        """分析用户意图并选择合适的工具"""
        
        if not self.api_available:
            logger.warning("LLM API不可用，使用关键词匹配fallback")
            return self._keyword_fallback(user_input, available_tools)
        
        try:
            # 构建工具描述
            tools_description = self._build_tools_description(available_tools)
            
            # 构建提示词
            system_prompt = f"""你是一个智能助手决策系统。根据用户输入，选择最合适的工具来处理用户请求。

可用工具：
{tools_description}

请分析用户意图，选择最合适的工具，并提供调用参数。

返回格式（必须是有效的JSON）：
{{
    "selected_tool": "工具名称",
    "confidence": 0.9,
    "reasoning": "选择原因",
    "parameters": {{
        "user_message": "用户原始消息"
    }}
}}

如果没有合适的工具，返回：
{{
    "selected_tool": null,
    "confidence": 0.0,
    "reasoning": "没有找到合适的工具",
    "parameters": {{}}
}}"""

            user_prompt = f"用户请求: {user_input}"
            
            # 创建LLM服务并调用
            llm_service = self._create_llm_service()
            
            responses = llm_service.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt
            )
            
            if responses and responses[0]:
                content = responses[0].strip()
                
                # 尝试解析JSON响应
                try:
                    result = json.loads(content)
                    logger.info(f"LLM意图分析成功: {result.get('selected_tool', 'None')}")
                    return result
                except json.JSONDecodeError:
                    logger.error(f"LLM返回的JSON格式无效: {content}")
                    return self._fallback_parse(content, available_tools)
            
        except Exception as e:
            logger.error(f"LLM意图分析失败: {e}")
        
        # fallback到关键词匹配
        return self._keyword_fallback(user_input, available_tools)
    
    def _build_tools_description(self, available_tools: List[str]) -> str:
        """构建工具描述"""
        tool_descriptions = {
            "APIKey获取工具": "获取今天的秘密API密钥，用于系统认证。适用于用户请求API密钥、认证码、秘密信息等场景。",
            "former_agent": "处理用户对话，生成XML表单。适用于创建算子配置、生成表单、收集需求等场景。",
            "data_analysis": "分析数据集并提供洞察。适用于数据分析、统计报告、数据处理等场景。",
            "code_generator": "根据需求生成DataFlow算子代码。适用于代码生成、编程实现、算法开发等场景。"
        }
        
        descriptions = []
        for tool in available_tools:
            desc = tool_descriptions.get(tool, "未知工具")
            descriptions.append(f"- {tool}: {desc}")
        
        return "\n".join(descriptions)
    
    def _fallback_parse(self, content: str, available_tools: List[str]) -> Dict[str, Any]:
        """当JSON解析失败时的fallback解析"""
        content_lower = content.lower()
        
        for tool in available_tools:
            if tool.lower() in content_lower:
                return {
                    "selected_tool": tool,
                    "confidence": 0.5,
                    "reasoning": f"通过文本匹配找到工具: {tool}",
                    "parameters": {"user_message": "解析失败，使用fallback"}
                }
        
        return {
            "selected_tool": None,
            "confidence": 0.0,
            "reasoning": "无法解析LLM响应",
            "parameters": {}
        }
    
    def _keyword_fallback(self, user_input: str, available_tools: List[str]) -> Dict[str, Any]:
        """关键词匹配fallback"""
        user_input_lower = user_input.lower()
        
        # API密钥相关关键词
        if any(keyword in user_input_lower for keyword in ["apikey", "api key", "密钥", "秘密", "认证"]):
            if "APIKey获取工具" in available_tools:
                return {
                    "selected_tool": "APIKey获取工具",
                    "confidence": 0.8,
                    "reasoning": "关键词匹配：API密钥相关",
                    "parameters": {"user_message": user_input}
                }
        
        # 表单生成相关关键词
        if any(keyword in user_input_lower for keyword in ["表单", "配置", "算子", "创建", "生成"]):
            if "former_agent" in available_tools:
                return {
                    "selected_tool": "former_agent",
                    "confidence": 0.8,
                    "reasoning": "关键词匹配：表单生成相关",
                    "parameters": {
                        "user_query": user_input,
                        "session_id": None,
                        "conversation_history": []
                    }
                }
        
        return {
            "selected_tool": None,
            "confidence": 0.0,
            "reasoning": "未找到匹配的工具",
            "parameters": {}
        }


# 单例实例
_llm_client = None

def get_llm_client() -> LLMClient:
    """获取LLM客户端单例"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
