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
        
        # 立即设置环境变量，确保后续调用可以使用
        if self.api_available:
            os.environ["DF_API_KEY"] = self.llm_config.api_key
            logger.info(f"已设置DF_API_KEY环境变量，长度: {len(self.llm_config.api_key)}")
        
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
    
    def call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """同步调用LLM"""
        if not self.api_available:
            return {"content": "API不可用，fallback模式"}
            
        try:
            llm_service = self._create_llm_service()
            responses = llm_service.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt
            )
            
            if responses and len(responses) > 0 and responses[0]:
                return {"content": responses[0]}
            else:
                return {"content": "LLM返回空响应"}
                
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            return {"content": f"LLM调用失败: {str(e)}"}
    
    async def call_llm_async(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """异步调用LLM（当前实现为同步调用的包装）"""
        return self.call_llm(system_prompt, user_prompt)
    
    def analyze_user_intent(self, user_input: str, available_tools: List[str]) -> Dict[str, Any]:
        """分析用户意图并选择合适的工具"""
        
        if not self.api_available:
            logger.warning("LLM API不可用，使用关键词匹配fallback")
            return 
        
        try:
            # 构建工具描述
            tools_description = self._build_tools_description(available_tools)
            
            # 构建提示词
            system_prompt = f"""你是一个智能助手决策系统。根据用户输入，选择最合适的工具来处理用户请求。

可用工具：
{tools_description}

**工具选择规则：**
1. **直接代码需求**: 用户明确要求"写代码"、"实现算法"、"编程"等 → 直接选择 code_workflow_agent
2. **需求分析场景**: 用户描述复杂业务需求，需要详细分析 → 选择 former 进行需求收集
3. **数据处理场景**: 涉及CSV、数据分析等 → 选择对应的数据工具
4. **其他工具**: 根据具体功能匹配

**重要：** 
- 如果用户直接要求编程/写代码，优先选择 code_workflow_agent，不要选择 former
- former 主要用于复杂需求的分析和表单收集，不适用于直接的编程请求

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
        return 
    
    def _build_tools_description(self, available_tools: List[str]) -> str:
        """构建工具描述 - 动态获取实际工具描述"""
        descriptions = []
        
        # 尝试从Master Agent获取实际的工具描述
        try:
            from dataflow.agent_v2.master.agent import MasterAgent
            # 创建临时的Master Agent实例来获取工具信息
            temp_master = MasterAgent()
            
            # 构建工具名到描述的映射
            tool_map = {}
            for tool in temp_master.direct_tools:
                tool_map[tool.name()] = tool.description()
            
            # 构建描述列表
            for tool_name in available_tools:
                if tool_name in tool_map:
                    descriptions.append(f"- {tool_name}: {tool_map[tool_name]}")
                else:
                    # fallback描述
                    descriptions.append(f"- {tool_name}: 工具功能未知")
                    
        except Exception as e:
            logger.warning(f"无法获取动态工具描述，使用fallback: {e}")
            # fallback到静态描述
            tool_descriptions = {
                "APIKey获取工具": "获取今天的秘密API密钥，用于系统认证。适用于用户请求API密钥、认证码、秘密信息等场景。",
                "former": "智能表单生成和用户交互处理工具，用于收集和整理用户需求。适用于需要分析和收集用户具体需求的场景。",
                "code_workflow_agent": "代码生成、测试、调试循环工具，自动化完成代码开发全流程。适用于编程、算法开发、代码实现等场景。",
                "pipeline_workflow_agent": "数据处理流水线推荐工具，自动分析数据、推荐算子、生成完整的数据处理流水线代码。适用于数据处理、数据分析、流水线构建等场景。",
                "sleep_tool": "等待N秒以模拟耗时操作",
                "csv_profile": "对CSV进行快速画像：缺失率、数值列统计、类别Top等",
                "csv_detect_time_columns": "采样解析，自动检测CSV中的日期时间列",
                "csv_vega_spec": "根据指定参数生成 Vega-Lite 图表配置（柱状图/折线图等）",
                "code_static_check": "AST 静态检查：函数、导入、近似复杂度、TODO 扫描",
                "code_test_stub": "从代码中提取函数并生成 pytest 单测骨架",
                "local_index_build": "从文档列表构建简易倒排索引，返回可传递的索引对象",
                "local_index_query": "在 local_index_build 返回的索引上做简单检索"
            }
            
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
    
# 单例实例
_llm_client = None

def get_llm_client() -> LLMClient:
    """获取LLM客户端单例"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
