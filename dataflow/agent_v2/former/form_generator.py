"""
表单生成器 - 基于模板和用户需求智能生成表单
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
import logging

from ..llm_client import get_llm_client

logger = logging.getLogger(__name__)


class FormGenerator:
    """表单生成器，负责基于模板和用户需求生成智能填充的表单"""
    
    def __init__(self):
        """初始化表单生成器"""
        self.llm = get_llm_client()
    
    def generate_form(self, template: Dict[str, Any], user_prompt: str) -> Dict[str, Any]:
        """生成表单并智能填充字段
        
        Args:
            template: 表单模板
            user_prompt: 用户原始需求
            
        Returns:
            包含表单数据的字典
        """
        if not template or 'xml_schema' not in template:
            logger.error("无效的表单模板")
            return {}
        
        schema = template['xml_schema']
        required_fields = schema.get('required_fields', [])
        optional_fields = schema.get('optional_fields', [])
        
        # 生成表单基础结构
        form_data = {
            'task_type': template.get('name', ''),
            'description': template.get('description', ''),
            'fields': {},
            'metadata': {
                'required_fields': required_fields,
                'optional_fields': optional_fields,
                'generated_at': self._get_timestamp(),
                'user_prompt': user_prompt
            }
        }
        
        # 智能填充字段
        filled_fields = self._intelligent_field_filling(
            required_fields + optional_fields, 
            user_prompt, 
            template
        )
        
        form_data['fields'].update(filled_fields)
        
        return form_data
    
    def _intelligent_field_filling(self, fields: List[str], user_prompt: str, template: Dict[str, Any]) -> Dict[str, Any]:
        """智能填充表单字段
        
        Args:
            fields: 需要填充的字段列表
            user_prompt: 用户原始需求
            template: 表单模板
            
        Returns:
            填充后的字段字典
        """
        if not fields:
            return {}
        
        # 构建字段描述
        field_descriptions = self._build_field_descriptions(fields)
        
        prompt = f"""基于用户需求，智能填充表单字段。

任务类型：{template.get('name', '')}
任务描述：{template.get('description', '')}

需要填充的字段：
{field_descriptions}

用户需求："{user_prompt}"

请分析用户需求，为每个字段提供合适的值。如果用户没有明确提供某个字段的信息，请基于上下文进行合理推测或建议。

注意事项：
1. user_requirements字段必须详细描述用户的具体需求
2. 示例字段（example_input/example_output）要具体、实用
3. 路径字段要符合实际情况
4. 如果某个字段无法从用户需求中推断，可以留空或给出建议值
5. 不要使用硬编码的示例值

返回JSON格式，字段名作为key，填充值作为value：
```json
{{
    "field_name": "填充值",
    ...
}}
```"""

        try:
            response = self.llm.call_llm("", prompt)
            
            result_text = response.get('content', '')
            
            # 提取JSON部分
            json_match = re.search(r'```json\s*\n(.*?)\n```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 如果没有找到```json块，尝试直接解析整个响应
                json_str = result_text.strip()
            
            # 解析JSON
            filled_data = json.loads(json_str)
            
            # 验证和清理数据
            cleaned_data = {}
            for field in fields:
                if field in filled_data:
                    value = filled_data[field]
                    if isinstance(value, str):
                        value = value.strip()
                    if value:  # 只添加非空值
                        cleaned_data[field] = value
            
            return cleaned_data
            
        except json.JSONDecodeError as e:
            logger.error(f"解析LLM返回的JSON失败: {e}")
            logger.error(f"原始响应: {result_text}")
            # 返回基础填充
            return self._fallback_field_filling(fields, user_prompt)
        except Exception as e:
            logger.error(f"智能字段填充失败: {e}")
            return self._fallback_field_filling(fields, user_prompt)
    
    def _build_field_descriptions(self, fields: List[str]) -> str:
        """构建字段描述"""
        descriptions = {
            'user_requirements': '用户需求描述 - 详细描述用户的具体需求和期望',
            'dataset': '数据集路径 - 数据文件的路径或数据源',
            'output_format': '输出格式 - 期望的输出格式（如JSON、CSV、TXT等）',
            'example_input': '输入示例 - 具体的输入数据示例',
            'example_output': '输出示例 - 期望的输出结果示例',
            'existing_code': '现有代码 - 需要优化的现有代码',
            'optimization_goal': '优化目标 - 具体的优化目标（如性能、内存、可读性等）',
            'dataset_path': '数据集路径 - 需要处理的数据集文件路径',
            'target_goal': '治理目标 - 数据治理的具体目标',
            'data_quality_issues': '数据质量问题 - 已知的数据质量问题',
            'processing_constraints': '处理约束 - 处理过程中的约束条件',
            'performance_requirements': '性能要求 - 对处理性能的具体要求',
            'document_types': '文档类型 - 需要处理的文档类型（如PDF、Word、TXT等）',
            'target_database': '目标数据库 - 数据入库的目标数据库',
            'cleaning_rules': '清洗规则 - 数据清洗的具体规则',
            'chunking_strategy': '分块策略 - 文档分块的策略',
            'embedding_model': '向量化模型 - 使用的向量化模型'
        }
        
        result = []
        for field in fields:
            desc = descriptions.get(field, f'{field} - 相关字段')
            result.append(f"- {field}: {desc}")
        
        return '\n'.join(result)
    
    def _fallback_field_filling(self, fields: List[str], user_prompt: str) -> Dict[str, Any]:
        """备用字段填充方法"""
        result = {}
        
        # 至少填充user_requirements字段
        if 'user_requirements' in fields:
            result['user_requirements'] = user_prompt
        
        return result
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def validate_form(self, form_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证表单数据的完整性
        
        Args:
            form_data: 表单数据
            
        Returns:
            (is_valid, error_messages): 验证结果和错误信息
        """
        errors = []
        
        if not form_data:
            errors.append("表单数据为空")
            return False, errors
        
        metadata = form_data.get('metadata', {})
        required_fields = metadata.get('required_fields', [])
        fields = form_data.get('fields', {})
        
        # 检查必填字段
        for field in required_fields:
            if field not in fields or not fields[field]:
                errors.append(f"必填字段 '{field}' 未填写")
        
        # 检查user_requirements字段的质量
        user_req = fields.get('user_requirements', '')
        if user_req and len(user_req.strip()) < 10:
            errors.append("用户需求描述过于简单，请提供更详细的描述")
        
        return len(errors) == 0, errors
    
    def update_field(self, form_data: Dict[str, Any], field_name: str, field_value: Any) -> Dict[str, Any]:
        """更新表单字段
        
        Args:
            form_data: 表单数据
            field_name: 字段名
            field_value: 字段值
            
        Returns:
            更新后的表单数据
        """
        if not form_data:
            return {}
        
        if 'fields' not in form_data:
            form_data['fields'] = {}
        
        form_data['fields'][field_name] = field_value
        
        # 更新修改时间
        if 'metadata' not in form_data:
            form_data['metadata'] = {}
        
        form_data['metadata']['last_updated'] = self._get_timestamp()
        
        return form_data
