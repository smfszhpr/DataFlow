#!/usr/bin/env python3
"""
参考AutoAgent的事件驱动架构，为DataFlow Agent实现真正的EventEngine
支持事件链式调用，自动触发agent执行
"""

import asyncio
import inspect
import uuid
from typing import Dict, List, Callable, Optional, Any, Literal, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class EventStatus(Enum):
    """事件状态枚举"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class EventResult:
    """事件执行结果"""
    event_id: str
    status: EventStatus
    data: Any = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: float = 0.0

@dataclass 
class BaseEvent:
    """基础事件类，参考AutoAgent的BaseEvent"""
    func: Callable
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)  # 完成后触发的事件
    
    def __post_init__(self):
        if not self.name:
            self.name = self.func.__name__
        if not self.description:
            self.description = self.func.__doc__ or f"Event: {self.name}"

class EventBroker:
    """事件代理，管理事件的传递和状态"""
    
    def __init__(self):
        self.event_store: Dict[str, EventResult] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
    
    async def publish(self, event_id: str, result: EventResult):
        """发布事件结果"""
        self.event_store[event_id] = result
        
        # 通知订阅者
        if event_id in self.subscribers:
            for callback in self.subscribers[event_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)
                    else:
                        callback(result)
                except Exception as e:
                    logger.error(f"订阅者回调失败: {e}")
    
    def subscribe(self, event_id: str, callback: Callable):
        """订阅事件"""
        if event_id not in self.subscribers:
            self.subscribers[event_id] = []
        self.subscribers[event_id].append(callback)
    
    def get_result(self, event_id: str) -> Optional[EventResult]:
        """获取事件结果"""
        return self.event_store.get(event_id)

class SmartEventEngine:
    """智能事件引擎，参考AutoAgent架构但专为DataFlow优化"""
    
    def __init__(self, name: str = "DataFlowEventEngine"):
        self.name = name
        self.events: Dict[str, BaseEvent] = {}
        self.broker = EventBroker()
        self.execution_graph: Dict[str, List[str]] = {}  # 事件依赖图
        self.running_events: Dict[str, asyncio.Task] = {}
        
    def register_event(self, func: Callable, 
                      name: str = "", 
                      description: str = "",
                      dependencies: List[str] = None,
                      triggers: List[str] = None) -> BaseEvent:
        """注册事件"""
        
        assert asyncio.iscoroutinefunction(func), f"事件函数 {func.__name__} 必须是异步函数"
        
        event = BaseEvent(
            func=func,
            name=name or func.__name__,
            description=description or func.__doc__ or "",
            dependencies=dependencies or [],
            triggers=triggers or []
        )
        
        self.events[event.event_id] = event
        self.execution_graph[event.event_id] = event.triggers
        
        logger.info(f"注册事件: {event.name} ({event.event_id})")
        return event
    
    async def emit_event(self, event_name_or_id: str, input_data: Any = None) -> EventResult:
        """触发事件执行"""
        
        # 先尝试按事件名称查找
        event = None
        for evt in self.events.values():
            if evt.name == event_name_or_id:
                event = evt
                break
        
        # 如果按名称没找到，尝试按ID查找
        if event is None and event_name_or_id in self.events:
            event = self.events[event_name_or_id]
        
        if event is None:
            available_events = [evt.name for evt in self.events.values()]
            raise ValueError(f"未找到事件: {event_name_or_id}，可用事件: {available_events}")
        
        # 检查依赖
        for dep_id in event.dependencies:
            dep_result = self.broker.get_result(dep_id)
            if not dep_result or dep_result.status != EventStatus.COMPLETED:
                logger.warning(f"事件 {event.name} 依赖 {dep_id} 未完成")
                return EventResult(
                    event_id=event.event_id,
                    status=EventStatus.FAILED,
                    error=f"依赖事件 {dep_id} 未完成"
                )
        
        # 执行事件
        start_time = datetime.now()
        
        try:
            logger.info(f"开始执行事件: {event.name}")
            
            # 获取函数签名
            sig = inspect.signature(event.func)
            
            # 准备参数
            kwargs = {}
            if 'data' in sig.parameters:
                kwargs['data'] = input_data
            if 'input_data' in sig.parameters:
                kwargs['input_data'] = input_data
            if 'event_context' in sig.parameters:
                kwargs['event_context'] = self._build_context(event.event_id)
            
            # 执行函数
            result_data = await event.func(**kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = EventResult(
                event_id=event.event_id,
                status=EventStatus.COMPLETED,
                data=result_data,
                execution_time=execution_time
            )
            
            logger.info(f"事件 {event.name} 执行成功，耗时 {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            result = EventResult(
                event_id=event.event_id,
                status=EventStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )
            logger.error(f"事件 {event.name} 执行失败: {e}")
        
        # 发布结果
        await self.broker.publish(event.event_id, result)
        
        # 触发后续事件
        if result.status == EventStatus.COMPLETED:
            await self._trigger_next_events(event.event_id, result.data)
        
        return result
    
    async def wait_for_completion(self, event_name_or_id: str, timeout: float = 30.0) -> Optional[EventResult]:
        """等待事件完成"""
        import asyncio
        
        # 找到对应的事件
        event = None
        for evt in self.events.values():
            if evt.name == event_name_or_id or evt.event_id == event_name_or_id:
                event = evt
                break
        
        if event is None:
            logger.error(f"等待完成时未找到事件: {event_name_or_id}")
            return None
        
        # 轮询检查结果
        start_time = asyncio.get_event_loop().time()
        while True:
            result = self.broker.get_result(event.event_id)
            if result and result.status in [EventStatus.COMPLETED, EventStatus.FAILED]:
                return result
            
            # 检查超时
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning(f"等待事件 {event_name_or_id} 完成超时")
                return None
            
            # 短暂等待
            await asyncio.sleep(0.1)
    
    async def _trigger_next_events(self, completed_event_id: str, result_data: Any):
        """触发后续事件"""
        
        if completed_event_id not in self.execution_graph:
            return
            
        for next_event_id in self.execution_graph[completed_event_id]:
            if next_event_id in self.events:
                logger.info(f"自动触发后续事件: {self.events[next_event_id].name}")
                # 异步触发，不等待完成
                asyncio.create_task(self.emit_event(next_event_id, result_data))
    
    def _build_context(self, event_id: str) -> Dict[str, Any]:
        """构建事件上下文"""
        event = self.events[event_id]
        context = {
            'event_id': event_id,
            'event_name': event.name,
            'dependencies_results': {},
            'broker': self.broker
        }
        
        # 添加依赖事件的结果
        for dep_id in event.dependencies:
            dep_result = self.broker.get_result(dep_id)
            if dep_result:
                context['dependencies_results'][dep_id] = dep_result.data
        
        return context
    
    def create_workflow(self, workflow_name: str, event_chain: List[str]) -> str:
        """创建工作流 - 按顺序执行一系列事件"""
        
        workflow_id = str(uuid.uuid4())
        
        # 设置依赖关系
        for i in range(1, len(event_chain)):
            current_event_id = event_chain[i]
            prev_event_id = event_chain[i-1]
            
            if current_event_id in self.events:
                self.events[current_event_id].dependencies.append(prev_event_id)
            
            if prev_event_id in self.events:
                self.events[prev_event_id].triggers.append(current_event_id)
                self.execution_graph[prev_event_id] = self.events[prev_event_id].triggers
        
        logger.info(f"创建工作流 {workflow_name}: {' -> '.join(event_chain)}")
        return workflow_id
    
    async def run_workflow(self, workflow_name: str, start_event_id: str, input_data: Any = None):
        """运行工作流"""
        logger.info(f"启动工作流: {workflow_name}")
        return await self.emit_event(start_event_id, input_data)
    
    def get_event_status(self, event_id: str) -> Optional[EventStatus]:
        """获取事件状态"""
        result = self.broker.get_result(event_id)
        return result.status if result else None
    
    def list_events(self) -> List[Dict[str, Any]]:
        """列出所有注册的事件"""
        return [
            {
                'id': event.event_id,
                'name': event.name,
                'description': event.description,
                'dependencies': event.dependencies,
                'triggers': event.triggers
            }
            for event in self.events.values()
        ]

# 全局事件引擎实例
global_event_engine = SmartEventEngine()

def event(name: str = "", 
          description: str = "",
          dependencies: List[str] = None,
          triggers: List[str] = None):
    """事件装饰器"""
    def decorator(func: Callable) -> Callable:
        # 注册事件到全局引擎
        global_event_engine.register_event(
            func, name, description, dependencies, triggers
        )
        # 返回原函数以便正常调用
        return func
    return decorator

# 导出主要接口
__all__ = [
    'SmartEventEngine', 
    'BaseEvent', 
    'EventResult', 
    'EventStatus',
    'event',
    'global_event_engine'
]
