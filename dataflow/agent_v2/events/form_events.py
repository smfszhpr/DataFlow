#!/usr/bin/env python3
"""
表单事件系统 - 实现前端表单实时更新
支持双向同步：用户在前端修改表单 + LLM对话修改表单
"""

from typing import Dict, Any, Optional, List
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from . import Event, EventType, EventBuilder

logger = logging.getLogger(__name__)


class FormEventType(str, Enum):
    """表单专用事件类型"""
    FORM_SCHEMA_UPDATED = "form_schema_updated"  # 表单结构更新
    FORM_DATA_UPDATED = "form_data_updated"      # 表单数据更新  
    FORM_FIELD_CHANGED = "form_field_changed"    # 单个字段更新
    FORM_VALIDATION_ERROR = "form_validation_error"  # 表单验证错误
    FORM_SUBMITTED = "form_submitted"            # 表单提交
    FORM_READY_FOR_EXECUTION = "form_ready_for_execution"  # 准备执行工作流


@dataclass
class FormSchema:
    """表单结构定义"""
    workflow_name: str
    workflow_description: str
    fields: Dict[str, Any]  # 字段定义
    required_fields: List[str]
    optional_fields: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "workflow_description": self.workflow_description,
            "fields": self.fields,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields
        }


@dataclass 
class FormData:
    """表单数据"""
    values: Dict[str, Any]  # 字段值
    validation_status: Dict[str, bool]  # 每个字段的验证状态
    missing_required: List[str]  # 缺失的必需字段
    is_complete: bool  # 是否完整
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "validation_status": self.validation_status,
            "missing_required": self.missing_required,
            "is_complete": self.is_complete
        }


class FormEventBuilder:
    """表单事件构建器"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.event_builder = EventBuilder(session_id)
    
    def form_schema_updated(self, form_schema: FormSchema, message: str = "表单结构已更新") -> Event:
        """表单结构更新事件"""
        return Event(
            type=EventType.STATE_UPDATE,
            session_id=self.session_id,
            timestamp=datetime.now(),
            data={
                "form_event_type": FormEventType.FORM_SCHEMA_UPDATED,
                "form_schema": form_schema.to_dict(),
                "message": message,
                "ui_type": "form_schema_update"
            }
        )
    
    def form_data_updated(self, form_data: FormData, changed_fields: List[str] = None, source: str = "llm") -> Event:
        """表单数据更新事件"""
        return Event(
            type=EventType.STATE_UPDATE,
            session_id=self.session_id,
            timestamp=datetime.now(),
            data={
                "form_event_type": FormEventType.FORM_DATA_UPDATED,
                "form_data": form_data.to_dict(),
                "changed_fields": changed_fields or [],
                "source": source,  # "llm" 或 "user"
                "ui_type": "form_data_update"
            }
        )
    
    def form_field_changed(self, field_name: str, field_value: Any, source: str = "llm") -> Event:
        """单个字段更改事件"""
        return Event(
            type=EventType.STATE_UPDATE,
            session_id=self.session_id,
            timestamp=datetime.now(),
            data={
                "form_event_type": FormEventType.FORM_FIELD_CHANGED,
                "field_name": field_name,
                "field_value": field_value,
                "source": source,
                "ui_type": "form_field_update"
            }
        )
    
    def form_validation_error(self, errors: List[str], invalid_fields: List[str] = None) -> Event:
        """表单验证错误事件"""
        return Event(
            type=EventType.STATE_UPDATE,
            session_id=self.session_id,
            timestamp=datetime.now(),
            data={
                "form_event_type": FormEventType.FORM_VALIDATION_ERROR,
                "errors": errors,
                "invalid_fields": invalid_fields or [],
                "ui_type": "form_validation_error"
            }
        )
    
    def form_ready_for_execution(self, target_workflow: str, execution_params: Dict[str, Any]) -> Event:
        """表单准备执行事件"""
        return Event(
            type=EventType.STATE_UPDATE,
            session_id=self.session_id,
            timestamp=datetime.now(),
            data={
                "form_event_type": FormEventType.FORM_READY_FOR_EXECUTION,
                "target_workflow": target_workflow,
                "execution_params": execution_params,
                "message": f"表单已完成，即将执行 {target_workflow} 工作流",
                "ui_type": "form_execution_ready"
            }
        )
    
    def form_submitted(self, target_workflow: str, execution_params: Dict[str, Any]) -> Event:
        """表单提交事件"""
        return Event(
            type=EventType.STATE_UPDATE,
            session_id=self.session_id,
            timestamp=datetime.now(),
            data={
                "form_event_type": FormEventType.FORM_SUBMITTED,
                "target_workflow": target_workflow,
                "execution_params": execution_params,
                "message": f"表单已提交，正在执行 {target_workflow}",
                "ui_type": "form_submitted"
            }
        )


class FormStateManager:
    """表单状态管理器 - 在Master Agent state中管理表单数据"""
    
    def __init__(self):
        self.form_schemas: Dict[str, FormSchema] = {}  # session_id -> FormSchema
        self.form_data: Dict[str, FormData] = {}       # session_id -> FormData
    
    def update_form_schema(self, session_id: str, workflow_name: str, workflow_description: str, 
                          fields: Dict[str, Any], required_fields: List[str], 
                          optional_fields: List[str] = None) -> FormSchema:
        """更新表单结构"""
        form_schema = FormSchema(
            workflow_name=workflow_name,
            workflow_description=workflow_description,
            fields=fields,
            required_fields=required_fields,
            optional_fields=optional_fields or []
        )
        self.form_schemas[session_id] = form_schema
        return form_schema
    
    def update_form_data(self, session_id: str, field_updates: Dict[str, Any], 
                        source: str = "llm") -> FormData:
        """更新表单数据"""
        # 获取或创建表单数据
        if session_id not in self.form_data:
            self.form_data[session_id] = FormData(
                values={},
                validation_status={},
                missing_required=[],
                is_complete=False
            )
        
        form_data = self.form_data[session_id]
        
        # 更新字段值
        form_data.values.update(field_updates)
        
        # 重新验证表单
        if session_id in self.form_schemas:
            form_schema = self.form_schemas[session_id]
            
            # 检查必需字段
            missing_required = []
            for field in form_schema.required_fields:
                if field not in form_data.values or not form_data.values[field]:
                    missing_required.append(field)
            
            form_data.missing_required = missing_required
            form_data.is_complete = len(missing_required) == 0
        
        return form_data
    
    def get_form_schema(self, session_id: str) -> Optional[FormSchema]:
        """获取表单结构"""
        return self.form_schemas.get(session_id)
    
    def get_form_data(self, session_id: str) -> Optional[FormData]:
        """获取表单数据"""
        return self.form_data.get(session_id)
    
    def is_form_complete(self, session_id: str) -> bool:
        """检查表单是否完整"""
        form_data = self.get_form_data(session_id)
        return form_data.is_complete if form_data else False
    
    def clear_form(self, session_id: str):
        """清除表单数据"""
        if session_id in self.form_schemas:
            del self.form_schemas[session_id]
        if session_id in self.form_data:
            del self.form_data[session_id]


# 全局表单状态管理器
form_state_manager = FormStateManager()


__all__ = [
    'FormEventType',
    'FormSchema', 
    'FormData',
    'FormEventBuilder',
    'FormStateManager',
    'form_state_manager'
]
