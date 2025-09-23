"""
Former模块 - 智能表单生成和用户交互

主要功能：
1. 分析用户需求，确定合适的任务类型
2. 根据模板智能生成和填充表单
3. 处理用户交互和表单修改
4. 提交表单到对应的工作流
"""

from .form_analyzer import FormAnalyzer
from .form_generator import FormGenerator

__all__ = [
    'FormAnalyzer', 
    'FormGenerator'
]
