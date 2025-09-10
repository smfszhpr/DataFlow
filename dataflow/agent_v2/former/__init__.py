"""
Former Agent V2 模块
提供表单生成和XML配置功能
"""

from .agent import FormerAgentV2
from .compat import FormerAgentCompat, FormRequest, FormResponse
from .tools import RequirementAnalysis, FieldValidation, XMLGeneration

# 为了与master agent兼容，导出兼容层
FormerAgent = FormerAgentCompat

__all__ = [
    'FormerAgentV2',
    'FormerAgent', 
    'FormerAgentCompat',
    'FormRequest',
    'FormResponse',
    'RequirementAnalysis',
    'FieldValidation', 
    'XMLGeneration'
]