"""
XML表单数据模型定义
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class FormRequest(BaseModel):
    """Former Agent请求模型"""
    user_query: str
    conversation_history: List[Dict[str, str]] = []
    session_id: Optional[str] = None

class FormResponse(BaseModel):
    """Former Agent响应模型"""
    need_more_info: bool
    agent_response: str
    xml_form: Optional[str] = None
    form_type: Optional[str] = None
    conversation_history: List[Dict[str, str]] = []

class SimpleWorkflowXML(BaseModel):
    """XML工作流模型，支持动态字段"""
    form_type: str
    user_requirements: str  # 用户需求（必需）
    
    # 通用字段
    dataset: str = ""  # 数据集路径或描述
    dataset_path: str = ""  # 数据集路径（兼容）
    output_format: str = "json"  # 输出格式
    
    # 算子相关字段
    example_input: str = ""  # 输入示例
    example_output: str = ""  # 输出示例
    existing_code: str = ""  # 现有代码（优化算子用）
    optimization_goal: str = ""  # 优化目标
    
    # Pipeline相关字段
    target_goal: str = ""  # 治理目标
    data_quality_issues: str = ""  # 数据质量问题
    processing_constraints: str = ""  # 处理约束
    performance_requirements: str = ""  # 性能要求
    
    # 知识库相关字段
    document_types: str = ""  # 文档类型
    target_database: str = ""  # 目标数据库
    cleaning_rules: str = ""  # 清洗规则
    chunking_strategy: str = ""  # 分块策略
    embedding_model: str = ""  # 向量化模型
