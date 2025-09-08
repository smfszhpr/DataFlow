"""
事件管理器
提供高级的事件流管理和预定义的工作流模板
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from . import EventEngine, Event, EventPriority
from .adapters import AnalysisEventAgent, ExecutionEventAgent, DebugEventAgent, CompletionEventAgent, FormerEventAgent
from ..toolkits import ChatAgentRequest
from dataflow import get_logger

logger = get_logger()

class EventManager:
    """事件管理器 - 高级事件流控制"""
    
    def __init__(self):
        self.engine = EventEngine()
        self.session_id = None
        self.workflow_templates = {}
        self._setup_default_agents()
        self._setup_workflow_templates()
        self._setup_event_listeners()
    
    def _setup_default_agents(self):
        """设置默认的Agent"""
        agents = [
            FormerEventAgent(),  # 添加Former Agent作为第一个
            AnalysisEventAgent(),
            ExecutionEventAgent(), 
            DebugEventAgent(),
            CompletionEventAgent()
        ]
        
        for agent in agents:
            self.engine.register_agent(agent)
    
    def _setup_workflow_templates(self):
        """设置工作流模板"""
        
        # Former Agent对话流程（新增）
        self.workflow_templates["form_conversation"] = [
            {"name": "form_conversation", "payload_keys": ["user_query", "session_id", "conversation_history"]},
            {"name": "form_validation", "payload_keys": ["xml_form", "form_type"]},
            {"name": "completion", "payload_keys": ["final_result"]}
        ]
        
        # 需求收集到XML生成流程（新增）
        self.workflow_templates["requirement_to_xml"] = [
            {"name": "requirement_collection", "payload_keys": ["user_input"]},
            {"name": "xml_generation", "payload_keys": ["requirements", "form_type"]},
            {"name": "form_validation", "payload_keys": ["xml_content", "form_type"]},
            {"name": "completion", "payload_keys": ["final_result"]}
        ]
        
        # 标准算子创建流程
        self.workflow_templates["create_operator"] = [
            {"name": "analysis", "payload_keys": ["request"]},
            {"name": "execution", "payload_keys": ["request", "analysis_result"]},
            {"name": "debug", "payload_keys": ["request", "execution_result"]},
            {"name": "completion", "payload_keys": ["final_result"]}
        ]
        
        # Pipeline推荐流程
        self.workflow_templates["recommend_pipeline"] = [
            {"name": "conversation_router", "payload_keys": ["user_query"]},
            {"name": "data_classification", "payload_keys": ["request"]},
            {"name": "analysis", "payload_keys": ["request"]},
            {"name": "completion", "payload_keys": ["final_result"]}
        ]
        
        # 代码优化流程
        self.workflow_templates["optimize_code"] = [
            {"name": "code_validation", "payload_keys": ["code"]},
            {"name": "analysis", "payload_keys": ["request", "validation_result"]},
            {"name": "execution", "payload_keys": ["request", "analysis_result"]},
            {"name": "debug", "payload_keys": ["request", "execution_result"]},
            {"name": "completion", "payload_keys": ["final_result"]}
        ]
        
        # 错误修复流程
        self.workflow_templates["fix_errors"] = [
            {"name": "error_fixing", "payload_keys": ["errors", "code"]},
            {"name": "code_validation", "payload_keys": ["code"]},
            {"name": "debug", "payload_keys": ["validation_result"]},
            {"name": "completion", "payload_keys": ["final_result"]}
        ]
    
    def _setup_event_listeners(self):
        """设置事件监听器"""
        self.engine.on("event_started", self._on_event_started)
        self.engine.on("event_completed", self._on_event_completed)
        self.engine.on("event_failed", self._on_event_failed)
    
    def _on_event_started(self, event: Event):
        """事件开始时的回调"""
        logger.info(f"📝 事件开始: [{event.event_id}] {event.name}")
    
    def _on_event_completed(self, event: Event):
        """事件完成时的回调"""
        duration = (event.completed_at - event.started_at).total_seconds()
        logger.info(f"✅ 事件完成: [{event.event_id}] {event.name} (耗时: {duration:.2f}s)")
    
    def _on_event_failed(self, event: Event):
        """事件失败时的回调"""
        logger.error(f"❌ 事件失败: [{event.event_id}] {event.name} - {event.error}")
    
    def create_workflow_from_template(self, template_name: str, initial_payload: Dict[str, Any]) -> List[Event]:
        """从模板创建工作流"""
        if template_name not in self.workflow_templates:
            raise ValueError(f"未知的工作流模板: {template_name}")
        
        template = self.workflow_templates[template_name]
        events = []
        
        for i, event_def in enumerate(template):
            # 构建事件payload
            payload = {}
            for key in event_def["payload_keys"]:
                if key in initial_payload:
                    payload[key] = initial_payload[key]
            
            # 创建事件
            event = Event(
                name=event_def["name"],
                payload=payload,
                priority=EventPriority.NORMAL
            )
            
            # 设置依赖关系（除了第一个事件）
            if i > 0:
                event.dependencies = [events[i-1].event_id]
            
            events.append(event)
        
        logger.info(f"从模板 '{template_name}' 创建了 {len(events)} 个事件")
        return events
    
    async def execute_workflow(self, template_name: str, initial_payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行完整的工作流"""
        logger.info(f"🚀 开始执行工作流: {template_name}")
        
        # 从模板创建事件
        events = self.create_workflow_from_template(template_name, initial_payload)
        
        # 添加事件到引擎
        self.engine.add_event_chain(events)
        
        # 执行引擎
        await self.engine.run_until_empty()
        
        # 返回统计信息
        stats = self.engine.get_statistics()
        logger.info(f"🎯 工作流 '{template_name}' 执行完成: {stats}")
        
        return {
            "workflow": template_name,
            "statistics": stats,
            "events_history": [
                {
                    "id": e.event_id,
                    "name": e.name,
                    "status": e.status.value,
                    "result": e.result
                }
                for e in self.engine.event_history
            ]
        }
    
    async def execute_single_event(self, event_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个事件"""
        event = Event(name=event_name, payload=payload)
        self.engine.add_event(event)
        
        result_event = await self.engine.process_next_event()
        return {
            "event_id": result_event.event_id,
            "status": result_event.status.value,
            "result": result_event.result,
            "error": result_event.error
        }
    
    def add_custom_event(self, name: str, payload: Dict[str, Any], priority: EventPriority = EventPriority.NORMAL):
        """添加自定义事件"""
        event = Event(name=name, payload=payload, priority=priority)
        self.engine.add_event(event)
        logger.info(f"添加自定义事件: {name}")
    
    def add_urgent_intervention(self, name: str, payload: Dict[str, Any]):
        """添加紧急干预事件"""
        event = Event(name=name, payload=payload, priority=EventPriority.URGENT)
        self.engine.insert_urgent_event(event)
        logger.info(f"添加紧急干预: {name}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return self.engine.get_queue_status()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取引擎统计"""
        return self.engine.get_statistics()
    
    async def handle_user_intervention(self, intervention_type: str, data: Dict[str, Any]):
        """处理用户干预"""
        logger.info(f"🤝 用户干预: {intervention_type}")
        
        if intervention_type == "modify_code":
            # 用户修改了代码，触发重新验证
            self.add_custom_event("code_validation", {
                "code": data.get("modified_code"),
                "trigger": "user_modification"
            })
        
        elif intervention_type == "add_requirement":
            # 用户添加了新需求，重新分析
            self.add_custom_event("analysis", {
                "additional_requirements": data.get("requirements"),
                "trigger": "user_addition"
            })
        
        elif intervention_type == "fix_error":
            # 用户要求修复特定错误
            self.add_urgent_intervention("error_fixing", {
                "errors": data.get("errors"),
                "user_feedback": data.get("feedback")
            })
        
        elif intervention_type == "change_direction":
            # 用户要求改变方向，清空队列重新开始
            self.engine.event_queue.clear()
            new_workflow = data.get("new_workflow", "create_operator")
            self.create_workflow_from_template(new_workflow, data.get("payload", {}))
    
    async def interactive_session(self, initial_request: ChatAgentRequest):
        """交互式会话模式"""
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🎪 开始交互式会话: {self.session_id}")
        
        # 根据请求类型选择初始工作流
        if "optimize" in initial_request.target.lower():
            workflow = "optimize_code"
        elif "pipeline" in initial_request.target.lower():
            workflow = "recommend_pipeline" 
        else:
            workflow = "create_operator"
        
        # 执行工作流
        result = await self.execute_workflow(workflow, {"request": initial_request})
        
        return {
            "session_id": self.session_id,
            "workflow": workflow,
            "result": result
        }

class EventFlowBuilder:
    """事件流构建器 - 用于动态构建复杂事件流"""
    
    def __init__(self):
        self.events: List[Event] = []
        self.dependencies: Dict[str, List[str]] = {}
    
    def add_event(self, name: str, payload: Dict[str, Any], depends_on: List[str] = None) -> str:
        """添加事件到流"""
        event = Event(name=name, payload=payload)
        
        if depends_on:
            event.dependencies = depends_on
            
        self.events.append(event)
        return event.event_id
    
    def add_parallel_events(self, events_data: List[Dict[str, Any]]) -> List[str]:
        """添加并行事件"""
        event_ids = []
        for event_data in events_data:
            event_id = self.add_event(
                event_data["name"],
                event_data["payload"],
                event_data.get("depends_on", [])
            )
            event_ids.append(event_id)
        return event_ids
    
    def add_conditional_branch(self, condition_event: str, true_events: List[Dict], false_events: List[Dict]):
        """添加条件分支"""
        # 这是一个简化版本，实际实现需要更复杂的条件逻辑
        for event_data in true_events:
            self.add_event(
                event_data["name"],
                {**event_data["payload"], "condition": "true"},
                [condition_event]
            )
        
        for event_data in false_events:
            self.add_event(
                event_data["name"], 
                {**event_data["payload"], "condition": "false"},
                [condition_event]
            )
    
    def build(self) -> List[Event]:
        """构建最终的事件流"""
        return self.events.copy()

# 便捷函数
async def quick_execute(workflow_name: str, request: ChatAgentRequest) -> Dict[str, Any]:
    """快速执行工作流的便捷函数"""
    manager = EventManager()
    return await manager.execute_workflow(workflow_name, {"request": request})

async def handle_xml_form_execution(xml_content: str, form_type: str) -> Dict[str, Any]:
    """处理XML表单执行的便捷函数"""
    manager = EventManager()
    
    # 根据表单类型选择工作流
    workflow_map = {
        "create_operator": "create_operator",
        "optimize_operator": "optimize_code", 
        "recommend_pipeline": "recommend_pipeline",
        "knowledge_base": "create_operator"  # 知识库构建也使用算子创建流程
    }
    
    workflow = workflow_map.get(form_type, "create_operator")
    
    return await manager.execute_workflow(workflow, {
        "xml_content": xml_content,
        "form_type": form_type
    })
