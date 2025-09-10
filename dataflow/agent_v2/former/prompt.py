"""
Former Agent 提示模板
包含需求分析、字段验证和XML生成的提示
"""
from typing import List
from ..base.core import Prompt


class FormerAgentPrompt(Prompt):
    """Former Agent 提示模板类"""
    
    def entry_prompt(self):
        """入口提示模板"""
        system_prompt = """你是一个专业的表单生成助手，负责分析用户需求并生成相应的XML表单。你的主要任务包括：

1. 需求分析：理解用户的业务需求，确定需要生成的表单类型
2. 字段提取：从用户描述中提取关键字段信息
3. 表单验证：确保所有必要字段都已收集完整
4. XML生成：根据模板生成标准化的XML表单

核心能力：
- 准确识别不同类型的业务表单需求（如数据分析、机器学习、API调用等）
- 智能提取和验证表单字段
- 生成符合业务规范的XML格式

工作流程：
1. 分析用户需求，确定表单类型
2. 提取必要的字段信息
3. 验证字段完整性
4. 生成最终XML表单

交互准则：
- 专注于表单生成，不进行对话交流
- 结果导向，确保输出的XML格式正确
- 如有字段缺失，明确指出需要补充的信息
- 保持专业性和准确性

请使用工具来完成表单生成任务。"""

        messages = [{"role": "system", "content": system_prompt}]
        return self.prompt_template(
            messages=messages, 
            with_history=True, 
            with_user_query=True, 
            with_agent_scratchpad=True
        )
    
    def analysis_prompt(self, user_requirement: str):
        """需求分析提示"""
        prompt = f"""分析以下用户需求，确定需要生成的表单类型：

用户需求：{user_requirement}

请识别：
1. 业务场景类型（数据分析、机器学习、API调用等）
2. 所需的表单类型
3. 关键字段和参数
4. 推荐的模板

输出格式为JSON：
{{
    "form_type": "确定的表单类型",
    "confidence": 0.9,
    "reasoning": "分析推理过程",
    "key_fields": ["字段1", "字段2"],
    "recommended_template": "推荐模板名称"
}}"""

        messages = [{"role": "user", "content": prompt}]
        return self.prompt_template(
            messages=messages, 
            with_history=False, 
            with_user_query=False
        )
    
    def validation_prompt(self, form_type: str, fields: dict, missing_fields: List[str]):
        """字段验证提示"""
        prompt = f"""验证表单字段的完整性：

表单类型：{form_type}
当前字段：{fields}
缺失字段：{missing_fields}

请评估：
1. 字段是否完整
2. 字段值是否有效
3. 是否需要补充信息
4. 提供改进建议

输出格式为JSON：
{{
    "is_complete": true/false,
    "validation_status": "完整/不完整",
    "missing_info": ["需要补充的信息"],
    "suggestions": ["改进建议"]
}}"""

        messages = [{"role": "user", "content": prompt}]
        return self.prompt_template(
            messages=messages, 
            with_history=False, 
            with_user_query=False
        )
    
    def generation_prompt(self, form_type: str, validated_fields: dict, template_name: str):
        """XML生成提示"""
        prompt = f"""根据验证的字段生成XML表单：

表单类型：{form_type}
模板名称：{template_name}
字段数据：{validated_fields}

生成要求：
1. 使用正确的XML格式
2. 包含所有必要的字段
3. 符合模板规范
4. 确保XML语法正确

输出格式：
{{
    "xml_content": "生成的XML内容",
    "validation_passed": true,
    "field_count": 字段数量,
    "template_used": "使用的模板"
}}"""

        messages = [{"role": "user", "content": prompt}]
        return self.prompt_template(
            messages=messages, 
            with_history=False, 
            with_user_query=False
        )
