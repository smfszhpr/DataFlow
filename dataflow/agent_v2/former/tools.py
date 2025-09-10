"""
Former Agent 工具集
用于表单分析、字段检查和XML生成
"""
from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel, Field
import yaml
import os

from ..base.core import BaseTool


class RequirementAnalysis(BaseTool):
    """需求分析工具"""
    
    @classmethod
    def name(cls) -> str:
        return "requirement_analysis"
    
    @classmethod
    def description(cls) -> str:
        return "分析用户需求并确定需要生成的表单类型"
    
    def params(self) -> Type[BaseModel]:
        class ToolParams(BaseModel):
            user_requirement: str = Field(description="用户的需求描述")
            context: Optional[str] = Field(default=None, description="上下文信息")
        
        return ToolParams
    
    def execute(self, user_requirement: str, context: str = None) -> Dict[str, Any]:
        """执行需求分析"""
        # 简化的需求分析逻辑，实际应该调用原 Former Agent
        # TODO: 集成原 Former Agent 的分析能力
        
        # 基于关键词的简单分析
        form_type = self._analyze_form_type(user_requirement)
        confidence = 0.8
        reasoning = f"基于关键词分析，识别为{form_type}类型"
        
        return {
            "success": True,
            "form_type": form_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "suggested_template": form_type.lower().replace(" ", "_"),
            "context": context
        }
    
    def _analyze_form_type(self, requirement: str) -> str:
        """改进的表单类型分析"""
        requirement_lower = requirement.lower()
        
        # 数据分析关键词
        data_analysis_keywords = ["数据分析", "分析数据", "统计", "报表", "销售数据", "趋势", "分布"]
        # 机器学习关键词  
        ml_keywords = ["机器学习", "训练模型", "预测", "算法", "分类", "回归", "客户流失"]
        # API调用关键词
        api_keywords = ["api", "接口", "调用", "请求", "天气", "外部"]
        # 流水线关键词
        pipeline_keywords = ["流水线", "pipeline", "工作流", "数据处理"]
        
        if any(keyword in requirement_lower for keyword in data_analysis_keywords):
            return "数据分析"
        elif any(keyword in requirement_lower for keyword in ml_keywords):
            return "机器学习"  
        elif any(keyword in requirement_lower for keyword in api_keywords):
            return "API调用"
        elif any(keyword in requirement_lower for keyword in pipeline_keywords):
            return "数据流水线"
        else:
            return "通用表单"


class FieldValidation(BaseTool):
    """字段验证工具"""
    
    @classmethod
    def name(cls) -> str:
        return "field_validation"
    
    @classmethod
    def description(cls) -> str:
        return "验证表单字段的完整性和有效性"
    
    def params(self) -> Type[BaseModel]:
        class ToolParams(BaseModel):
            form_type: str = Field(description="表单类型")
            extracted_fields: Dict[str, Any] = Field(description="提取的字段信息")
            user_input: str = Field(description="用户输入")
        
        return ToolParams
    
    def execute(self, form_type: str, extracted_fields: Dict[str, Any], 
                user_input: str) -> Dict[str, Any]:
        """执行字段验证"""
        # 简化的字段验证逻辑
        # TODO: 集成原 Former Agent 的验证能力
        
        validation_result = self._validate_fields(form_type, extracted_fields, user_input)
        
        return {
            "success": True,
            "is_complete": validation_result.get("is_complete", False),
            "missing_fields": validation_result.get("missing_fields", []),
            "validated_fields": validation_result.get("validated_fields", {}),
            "suggestions": validation_result.get("suggestions", [])
        }
    
    def _validate_fields(self, form_type: str, fields: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """简单的字段验证"""
        # 从用户输入中提取字段
        extracted = self._extract_fields_from_input(user_input)
        
        # 根据表单类型定义必需字段
        required_fields = self._get_required_fields(form_type)
        
        # 检查缺失字段
        missing = [field for field in required_fields if field not in extracted]
        
        return {
            "is_complete": len(missing) == 0,
            "missing_fields": missing,
            "validated_fields": extracted,
            "suggestions": [f"请提供{field}的信息" for field in missing]
        }
    
    def _extract_fields_from_input(self, user_input: str) -> Dict[str, Any]:
        """改进的字段提取"""
        fields = {}
        input_lower = user_input.lower()
        
        # 数据源相关
        if any(keyword in input_lower for keyword in ["mysql", "数据库", "数据源", "csv", "excel"]):
            if "mysql" in input_lower:
                fields["data_source"] = "MySQL数据库"
            elif "csv" in input_lower:
                fields["data_source"] = "CSV文件"
            elif "excel" in input_lower:
                fields["data_source"] = "Excel文件"
            else:
                fields["data_source"] = "数据库"
        
        # 分析类型
        if any(keyword in input_lower for keyword in ["月度", "趋势", "分布", "报表"]):
            analysis_types = []
            if "月度" in input_lower:
                analysis_types.append("月度分析")
            if "趋势" in input_lower:
                analysis_types.append("趋势分析")
            if "地区" in input_lower or "分布" in input_lower:
                analysis_types.append("分布分析")
            fields["analysis_type"] = "、".join(analysis_types)
        
        # 模型相关
        if any(keyword in input_lower for keyword in ["分类", "预测", "模型", "训练"]):
            if "分类" in input_lower:
                fields["model_type"] = "分类模型"
            elif "回归" in input_lower:
                fields["model_type"] = "回归模型"
            elif "预测" in input_lower:
                fields["model_type"] = "预测模型"
            else:
                fields["model_type"] = "机器学习模型"
        
        # 目标变量
        if any(keyword in input_lower for keyword in ["流失", "目标变量", "标签"]):
            if "流失" in input_lower:
                fields["target_variable"] = "客户流失标识"
            else:
                fields["target_variable"] = "目标变量"
        
        # API相关
        if any(keyword in input_lower for keyword in ["api", "接口", "天气", "城市"]):
            if "天气" in input_lower:
                fields["endpoint"] = "天气API接口"
                fields["method"] = "GET"
            else:
                fields["endpoint"] = "API端点"
                fields["method"] = "GET"
        
        if any(keyword in input_lower for keyword in ["参数", "城市", "传入"]):
            if "城市" in input_lower:
                fields["parameters"] = "城市名称参数"
            else:
                fields["parameters"] = "请求参数"
        
        # 通用描述
        if any(keyword in input_lower for keyword in ["分析", "处理", "生成", "调用"]):
            fields["description"] = user_input[:100]  # 截取前100个字符作为描述
            
        return fields
    
    def _get_required_fields(self, form_type: str) -> List[str]:
        """获取表单类型的必需字段"""
        field_map = {
            "数据分析": ["data_source", "analysis_type"],
            "机器学习": ["data_source", "model_type", "target_variable"],
            "API调用": ["endpoint", "method", "parameters"],
            "数据流水线": ["input_source", "output_target", "operations"],
            "通用表单": ["description"]
        }
        return field_map.get(form_type, ["description"])


class XMLGeneration(BaseTool):
    """XML生成工具"""
    
    @classmethod
    def name(cls) -> str:
        return "xml_generation"
    
    @classmethod
    def description(cls) -> str:
        return "根据验证的字段生成最终的XML表单"
    
    def params(self) -> Type[BaseModel]:
        class ToolParams(BaseModel):
            form_type: str = Field(description="表单类型")
            validated_fields: Dict[str, Any] = Field(description="已验证的字段")
            template_name: str = Field(description="模板名称")
        
        return ToolParams
    
    def execute(self, form_type: str, validated_fields: Dict[str, Any], 
                template_name: str) -> Dict[str, Any]:
        """执行XML生成"""
        # 简化的XML生成逻辑
        # TODO: 集成原 Former Agent 的XML生成能力
        
        xml_content = self._generate_xml_content(form_type, validated_fields, template_name)
        
        return {
            "success": True,
            "xml_content": xml_content,
            "form_type": form_type,
            "template_used": template_name,
            "field_count": len(validated_fields),
            "generation_timestamp": self._get_timestamp()
        }
    
    def _generate_xml_content(self, form_type: str, fields: Dict[str, Any], template_name: str) -> str:
        """生成XML内容"""
        # 简单的XML生成
        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_lines.append(f'<form type="{form_type}" template="{template_name}">')
        
        for field_name, field_value in fields.items():
            xml_lines.append(f'  <field name="{field_name}">{field_value}</field>')
        
        xml_lines.append('</form>')
        
        return '\n'.join(xml_lines)
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        import datetime
        return datetime.datetime.now().isoformat()
