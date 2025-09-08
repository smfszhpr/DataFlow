"""
XML表单Former Agent
负责通过对话收集用户需求，生成XML表单
"""

import os
import uuid
import asyncio
import xml.etree.ElementTree as ET
from typing import Dict, Any, List

from ..xmlforms.models import FormRequest, FormResponse
from ..xmlforms.form_templates import FormTemplateManager
from ..promptstemplates.prompt_template import PromptsTemplateGenerator
from ..eventengine.config_manager import get_llm_config, get_former_config
from dataflow.serving import APILLMServing_request
from dataflow import get_logger

logger = get_logger()

class FormerAgent:
    """XML表单Former Agent - 对话式表单生成器"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.template_manager = FormTemplateManager()
        self.prompt_template = PromptsTemplateGenerator(output_language="zh")
        self.prompts = self.prompt_template  # 别名，兼容旧代码
        self.max_history = get_former_config().max_history
        self.session_states = {}  # 存储会话状态
        self.conversations = {}  # 存储对话历史
        
        # 获取配置
        self.llm_config = get_llm_config()
        self.former_config = get_former_config()
        self.api_available = bool(self.llm_config.api_key and self.llm_config.api_url)
        
        if not self.api_available:
            logger.warning("API配置不可用，将使用模板回复模式")
        
        logger.info(f"Former Agent 初始化完成，会话ID: {self.session_id}, API可用: {self.api_available}")
        
    def _build_form_selection_prompt(self, query: str) -> str:
        """构建表单类型选择提示词"""
        available_forms = self.template_manager.get_available_forms()
        forms_list = "\\n".join([f"- {form_type}: {desc}" for form_type, desc in available_forms.items()])
        
        # 使用提示词模板
        return self.prompts.get_xml_form_selection_prompt(query, forms_list)

    def _build_conversation_prompt(self, query: str, history: List[Dict[str, str]], 
                                 form_type: str) -> str:
        """构建对话提示词"""
        
        template_name = self.template_manager.get_template_name(form_type)
        xml_schema = self.template_manager.get_xml_schema(form_type)
        conversation_guide = self.template_manager.get_conversation_guide(form_type)
        
        conversation_context = ""
        if history:
            conversation_context = "\\n".join([
                f"{'用户' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in history[-6:]  # 最近6轮对话
            ])
        
        # 动态构建XML字段
        required_fields = xml_schema.get('required_fields', [])
        optional_fields = xml_schema.get('optional_fields', [])
        
        # 使用提示词模板
        return self.prompts.get_xml_form_conversation_prompt(
            template_name=template_name,
            conversation_context=conversation_context,
            query=query,
            form_type=form_type,
            required_fields=required_fields,
            optional_fields=optional_fields,
            conversation_guide=conversation_guide
        )

    async def _detect_form_type_with_llm(self, query: str) -> str:
        """使用大模型检测表单类型"""
        try:
            # 如果API不可用，使用关键字匹配作为fallback
            if not self.api_available:
                logger.warning("API不可用，使用关键字匹配检测表单类型")
                return self._detect_form_type_by_keywords(query)
            
            prompt = self._build_form_selection_prompt(query)
            
            # 临时设置环境变量（为了兼容现有API）
            import os
            os.environ["DF_API_KEY"] = self.llm_config.api_key
            
            # 创建LLM服务实例
            llm_service = APILLMServing_request(
                api_url=self.llm_config.api_url,
                key_name_of_api_key="DF_API_KEY",
                model_name=self.llm_config.model
            )
            
            # 使用generate_from_input方法
            responses = llm_service.generate_from_input(
                user_inputs=[prompt],
                system_prompt="你是一个表单类型分类专家，请根据用户查询选择最合适的表单类型。"
            )
            
            if responses and responses[0]:
                detected_type = responses[0].strip()
                # 验证返回的类型是否有效
                if detected_type in self.template_manager.templates:
                    return detected_type
            
        except Exception as e:
            logger.error(f"大模型表单类型检测失败: {e}")
        
        # fallback到关键字匹配
        return self._detect_form_type_by_keywords(query)
    
    def _detect_form_type_by_keywords(self, query: str) -> str:
        """基于关键字的表单类型检测（fallback方法）"""
        query_lower = query.lower()
        
        # 关键字映射
        if "算子" in query or "operator" in query_lower:
            return "create_operator"
        elif "pipeline" in query_lower or "流水线" in query or "推荐" in query:
            return "recommend_pipeline"
        elif "优化" in query or "optimize" in query_lower:
            return "optimize_operator"
        elif "知识库" in query or "knowledge" in query_lower:
            return "knowledge_base"
        else:
            return "create_operator"  # 默认算子创建

    async def process_conversation(self, request: FormRequest) -> FormResponse:
        """处理对话并返回响应"""
        session_id = request.session_id or str(uuid.uuid4())
        
        # 检测或获取表单类型
        if session_id not in self.session_states:
            form_type = await self._detect_form_type_with_llm(request.user_query)
            self.session_states[session_id] = {"form_type": form_type}
        else:
            form_type = self.session_states[session_id]["form_type"]
        
        # 获取或创建对话历史
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        conversation_history = self.conversations[session_id]
        conversation_history.extend(request.conversation_history)
        
        # 添加用户输入
        conversation_history.append({
            "role": "user",
            "content": request.user_query
        })
        
        # 构建提示词
        prompt = self._build_conversation_prompt(request.user_query, conversation_history, form_type)
        
        # 调用LLM
        try:
            # 使用环境变量中的API配置
            # 如果API不可用，使用模板化响应
            if not self.api_available:
                logger.warning("API不可用，使用模板化响应")
                agent_response = self._generate_template_response(request.user_query, form_type, conversation_history)
            else:
                # 临时设置环境变量（为了兼容现有API）
                import os
                os.environ["DF_API_KEY"] = self.llm_config.api_key
                
                # 创建LLM服务实例
                llm_service = APILLMServing_request(
                    api_url=self.llm_config.api_url,
                    key_name_of_api_key="DF_API_KEY",
                    model_name=self.llm_config.model
                )
                
                # 使用generate_from_input方法
                responses = llm_service.generate_from_input(
                    user_inputs=[prompt],
                    system_prompt="你是一个专业的用户需求分析师，负责通过对话收集用户需求并生成XML表单。"
                )
                
                agent_response = responses[0] if responses and responses[0] else "抱歉，暂时无法响应您的请求。"
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            # fallback到模板化响应
            agent_response = self._generate_template_response(request.user_query, form_type, conversation_history)
        
        # 添加助手响应
        conversation_history.append({
            "role": "assistant",
            "content": agent_response
        })
        
        # 更新会话历史
        self.conversations[session_id] = conversation_history
        
        # 检查是否包含XML表单
        xml_form = None
        need_more_info = True
        
        if "```xml" in agent_response and "</workflow>" in agent_response:
            try:
                xml_start = agent_response.find("```xml") + 6
                xml_end = agent_response.find("```", xml_start)
                xml_content = agent_response[xml_start:xml_end].strip()
                
                # 验证XML格式
                ET.fromstring(xml_content)
                xml_form = xml_content
                need_more_info = False
                
            except ET.ParseError:
                logger.warning("生成的XML格式有误")
        
        return FormResponse(
            need_more_info=need_more_info,
            agent_response=agent_response,
            xml_form=xml_form,
            form_type=form_type,
            conversation_history=conversation_history
        )
    
    def _generate_template_response(self, user_query: str, form_type: str, conversation_history: List[Dict[str, str]]) -> str:
        """生成模板化响应（当没有API key时使用）"""
        
        # 分析用户查询
        query_lower = user_query.lower()
        
        # 根据表单类型和查询内容生成响应
        if form_type == "create_operator":
            if len(conversation_history) <= 2:  # 第一次交互
                return f"""我理解您想要创建一个算子。根据您的描述："{user_query}"

为了帮您生成合适的算子，我需要了解以下信息：
1. 算子的具体功能是什么？
2. 输入数据的格式和字段名称
3. 期望的输出格式
4. 是否有特殊的处理逻辑要求？

请您详细描述这些需求，我会为您生成相应的XML配置。"""
            else:
                # 生成基本的XML表单
                return f"""基于您提供的信息，我为您生成了算子配置：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<workflow>
    <form_type>create_operator</form_type>
    <user_requirements>{user_query}</user_requirements>
    <dataset_path>请提供数据集路径</dataset_path>
    <example_input>请提供输入示例</example_input>
    <example_output>请提供输出示例</example_output>
    <output_format>json</output_format>
</workflow>
```

这是一个基础配置，您可以根据具体需求进行调整。"""
        
        elif form_type == "recommend_pipeline":
            if len(conversation_history) <= 2:
                return f"""我理解您需要Pipeline推荐。根据您的描述："{user_query}"

为了为您推荐最合适的数据处理流水线，请告诉我：
1. 您的数据类型（文本、图像、表格等）
2. 数据处理的目标
3. 数据量大小
4. 性能要求

请提供更多详细信息。"""
            else:
                return f"""基于您的需求，我推荐以下Pipeline配置：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<workflow>
    <form_type>recommend_pipeline</form_type>
    <user_requirements>{user_query}</user_requirements>
    <target_goal>数据处理和分析</target_goal>
    <dataset_path>请指定数据路径</dataset_path>
    <processing_constraints>标准处理流程</processing_constraints>
    <performance_requirements>中等性能要求</performance_requirements>
</workflow>
```"""
        
        elif form_type == "optimize_operator":
            return f"""我理解您想要优化现有算子。

请提供：
1. 当前算子的代码
2. 遇到的具体问题
3. 优化目标（性能、准确性等）

我会帮您分析并生成优化建议。"""
        
        else:
            return f"""我理解您的需求："{user_query}"

这是一个{form_type}类型的请求。请提供更多详细信息，以便我为您生成合适的配置。

您可以描述：
- 具体的功能需求
- 数据格式要求
- 预期的处理结果"""
