"""
表单模板管理器
负责加载和管理多种表单类型，使用大模型智能选择
"""

import yaml
import os
from pathlib import Path
from typing import Dict, List
from dataflow import get_logger

logger = get_logger()

class FormTemplateManager:
    """表单模板管理器，使用大模型智能选择表单类型"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # 从promptstemplates目录加载模板
            config_path = Path(__file__).parent.parent / "promptstemplates" / "xml_form_templates.yaml"
        
        self.config_path = config_path
        self.templates = {}
        self.default_config = {}
        self.load_templates()
    
    def load_templates(self):
        """加载表单模板配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.templates = config.get('templates', {})
            self.default_config = config.get('default_config', {})
            
            logger.info(f"加载了 {len(self.templates)} 个表单模板")
            
        except Exception as e:
            logger.error(f"加载表单模板失败: {e}")
            # 使用默认模板
            self._load_default_templates()
    
    def _load_default_templates(self):
        """加载默认表单模板"""
        self.templates = {
            "create_operator": {
                "name": "创建算子",
                "description": "根据用户需求创建新的数据处理算子",
                "xml_schema": {
                    "required_fields": ["user_requirements", "dataset", "output_format"],
                    "optional_fields": ["example_input", "example_output"]
                }
            },
            "optimize_operator": {
                "name": "优化算子", 
                "description": "对现有算子进行性能优化或功能增强",
                "xml_schema": {
                    "required_fields": ["user_requirements", "dataset", "existing_code", "optimization_goal"],
                    "optional_fields": ["example_input", "example_output"]
                }
            },
            "recommend_pipeline": {
                "name": "推荐Pipeline",
                "description": "根据数据治理需求推荐合适的pipeline组合",
                "xml_schema": {
                    "required_fields": ["user_requirements", "dataset_path", "target_goal"],
                    "optional_fields": ["data_quality_issues", "processing_constraints"]
                }
            },
            "knowledge_base": {
                "name": "知识库构建",
                "description": "构建知识库数据清洗与入库pipeline",
                "xml_schema": {
                    "required_fields": ["user_requirements", "document_types", "target_database"],
                    "optional_fields": ["cleaning_rules", "chunking_strategy", "embedding_model"]
                }
            }
        }
        
        self.default_config = {
            "execution": {
                "timeout": 3600,
                "max_debug_round": 3,
                "stream_output": True,
                "execute_the_operator": True,
                "execute_the_pipeline": True
            },
            "api": {
                "model": "deepseek-v3",
                "backend_url": "http://localhost:8081"
            }
        }
    
    def get_available_forms(self) -> Dict[str, str]:
        """获取可用的表单类型和描述"""
        return {
            form_type: template.get('description', template.get('name', form_type))
            for form_type, template in self.templates.items()
        }
    
    def get_xml_schema(self, form_type: str) -> Dict[str, List[str]]:
        """获取表单的XML模式定义"""
        template = self.templates.get(form_type, {})
        return template.get('xml_schema', {})
    
    def get_conversation_guide(self, form_type: str) -> str:
        """获取指定表单类型的对话引导"""
        template = self.templates.get(form_type, {})
        return template.get('conversation_guide', '')
    
    def get_template_name(self, form_type: str) -> str:
        """获取表单模板名称"""
        template = self.templates.get(form_type, {})
        return template.get('name', form_type)
