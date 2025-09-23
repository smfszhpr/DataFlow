"""
表单分析器 - 分析用户需求并确定合适的表单模板
"""

import yaml
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from ..llm_client import get_llm_client
from ..base.core import SubAgent, node, entry

logger = logging.getLogger(__name__)


class FormAnalyzer:
    """表单分析器，负责分析用户需求并选择合适的表单模板"""
    
    def __init__(self, templates_path: str = None):
        """初始化表单分析器
        
        Args:
            templates_path: 表单模板文件路径
        """
        if templates_path is None:
            templates_path = Path(__file__).parent / "form_templates.yaml"
        
        self.templates_path = Path(templates_path)
        self.templates = self._load_templates()
        self.llm = get_llm_client()
    
    def _load_templates(self) -> Dict[str, Any]:
        """加载表单模板"""
        try:
            with open(self.templates_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get('templates', {})
        except Exception as e:
            logger.error(f"加载表单模板失败: {e}")
            return {}
    
    def analyze_user_intent(self, user_prompt: str) -> Tuple[str, float]:
        """分析用户意图，返回最匹配的任务类型和置信度
        完全基于 LLM 分析，不使用关键词匹配
        
        Args:
            user_prompt: 用户输入的需求描述
            
        Returns:
            (task_type, confidence): 任务类型和置信度
        """
        logger.info("开始LLM意图分析，不使用关键词匹配...")
        
        # 直接使用LLM进行意图分析
        llm_result = self._llm_intent_analysis(user_prompt)
        
        logger.info(f"LLM分析结果: {llm_result}")
        return llm_result
    
    def _llm_intent_analysis(self, user_prompt: str) -> Tuple[str, float]:
        """使用LLM进行意图分析"""
        template_descriptions = []
        for task_type, template in self.templates.items():
            template_descriptions.append(
                f"- {task_type}: {template.get('name', '')} - {template.get('description', '')}"
            )
        
        prompt = f"""分析用户需求，确定最合适的任务类型。

可用的任务类型：
{chr(10).join(template_descriptions)}

用户需求："{user_prompt}"

请分析用户需求，选择最合适的任务类型，并给出置信度(0-1)。

返回格式：
任务类型: <task_type>
置信度: <confidence>
理由: <reason>"""

        try:
            response = self.llm.call_llm("", prompt)
            
            result_text = response.get('content', '')
            
            # 解析响应
            task_type = self._extract_field(result_text, r'任务类型[:：]\s*(\w+)')
            confidence_str = self._extract_field(result_text, r'置信度[:：]\s*([\d.]+)')
            
            # 验证任务类型是否有效
            if task_type not in self.templates:
                logger.warning(f"LLM返回了无效的任务类型: {task_type}")
                return 'usual_code', 0.3
            
            confidence = float(confidence_str) if confidence_str else 0.5
            confidence = max(0.0, min(1.0, confidence))  # 限制在0-1范围内
            
            return task_type, confidence
            
        except Exception as e:
            logger.error(f"LLM意图分析失败: {e}")
            return 'usual_code', 0.3
    
    def _extract_field(self, text: str, pattern: str) -> Optional[str]:
        """从文本中提取字段"""
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None
    
    def get_template(self, task_type: str) -> Dict[str, Any]:
        """获取指定任务类型的模板"""
        return self.templates.get(task_type, {})
    
    def get_available_tasks(self) -> List[str]:
        """获取所有可用的任务类型"""
        return list(self.templates.keys())
