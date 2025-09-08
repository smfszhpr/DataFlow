"""
事件驱动引擎模块
封装各个Agent，实现事件队列和动态调度机制
"""

import asyncio
import uuid
from collections import deque
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from dataflow import get_logger

logger = get_logger()

class EventStatus(Enum):
    """事件状态枚举"""
    PENDING = "pending"
    RUNNING = "running" 
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EventPriority(Enum):
    """事件优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

@dataclass
class Event:
    """事件对象"""
    name: str                           # 事件名称
    payload: Dict[str, Any]            # 事件数据/参数
    status: EventStatus = EventStatus.PENDING
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)  # 依赖的事件ID
    
    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = EventStatus(self.status)
        if isinstance(self.priority, int):
            self.priority = EventPriority(self.priority)

class AgentInterface:
    """Agent接口基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.supported_events = []
    
    async def handle(self, event: Event) -> Dict[str, Any]:
        """处理事件的核心方法"""
        raise NotImplementedError("Subclass must implement handle method")
    
    def can_handle(self, event_name: str) -> bool:
        """判断是否能处理指定事件"""
        return event_name in self.supported_events

class EventEngine:
    """事件驱动引擎"""
    
    def __init__(self):
        self.event_queue = deque()          # 事件队列
        self.agents: Dict[str, AgentInterface] = {}  # 注册的Agent
        self.event_history: List[Event] = [] # 事件历史
        self.running = False                # 引擎运行状态
        self.event_handlers: Dict[str, List[Callable]] = {}  # 事件监听器
        self.global_context: Dict[str, Any] = {}  # 全局上下文
        
    def register_agent(self, agent: AgentInterface):
        """注册Agent"""
        self.agents[agent.name] = agent
        logger.info(f"注册Agent: {agent.name}, 支持事件: {agent.supported_events}")
    
    def add_event(self, event: Event):
        """添加事件到队列"""
        # 按优先级插入事件
        inserted = False
        for i, existing_event in enumerate(self.event_queue):
            if event.priority.value > existing_event.priority.value:
                self.event_queue.insert(i, event)
                inserted = True
                break
        
        if not inserted:
            self.event_queue.append(event)
        
        logger.info(f"添加事件 [{event.event_id}] {event.name} (优先级: {event.priority.name})")
        self._trigger_event_added(event)
    
    def add_event_chain(self, events: List[Event]):
        """批量添加事件链"""
        for event in events:
            self.add_event(event)
    
    def create_event(self, name: str, payload: Dict[str, Any], **kwargs) -> Event:
        """创建事件的便捷方法"""
        event = Event(name=name, payload=payload, **kwargs)
        return event
    
    def insert_urgent_event(self, event: Event):
        """插入紧急事件到队列前端"""
        event.priority = EventPriority.URGENT
        self.event_queue.appendleft(event)
        logger.info(f"插入紧急事件 [{event.event_id}] {event.name}")
        self._trigger_event_added(event)
    
    def cancel_event(self, event_id: str) -> bool:
        """取消指定事件"""
        for event in self.event_queue:
            if event.event_id == event_id:
                event.status = EventStatus.CANCELLED
                self.event_queue.remove(event)
                logger.info(f"取消事件 [{event_id}] {event.name}")
                return True
        return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        status = {
            "total_events": len(self.event_queue),
            "pending_events": len([e for e in self.event_queue if e.status == EventStatus.PENDING]),
            "event_summary": []
        }
        
        for event in list(self.event_queue)[:10]:  # 只显示前10个
            status["event_summary"].append({
                "id": event.event_id,
                "name": event.name,
                "status": event.status.value,
                "priority": event.priority.name,
                "created_at": event.created_at.strftime("%H:%M:%S")
            })
        
        return status
    
    async def process_next_event(self) -> Optional[Event]:
        """处理下一个事件"""
        if not self.event_queue:
            return None
        
        event = self.event_queue.popleft()
        
        # 检查依赖关系
        if not self._check_dependencies(event):
            # 依赖未满足，重新放回队列末尾
            self.event_queue.append(event)
            logger.warning(f"事件 [{event.event_id}] {event.name} 依赖未满足，重新排队")
            return None
        
        return await self._execute_event(event)
    
    def _check_dependencies(self, event: Event) -> bool:
        """检查事件依赖是否满足"""
        if not event.dependencies:
            return True
        
        completed_events = {e.event_id for e in self.event_history if e.status == EventStatus.DONE}
        return all(dep_id in completed_events for dep_id in event.dependencies)
    
    async def _execute_event(self, event: Event) -> Event:
        """执行单个事件"""
        event.status = EventStatus.RUNNING
        event.started_at = datetime.now()
        
        logger.info(f"🚀 开始执行事件 [{event.event_id}] {event.name}")
        self._trigger_event_started(event)
        
        try:
            # 查找能处理此事件的Agent
            agent = self._find_agent_for_event(event.name)
            if not agent:
                raise ValueError(f"没有找到能处理事件 '{event.name}' 的Agent")
            
            # 执行事件
            result = await agent.handle(event)
            
            event.result = result
            event.status = EventStatus.DONE
            event.completed_at = datetime.now()
            
            logger.info(f"✅ 事件 [{event.event_id}] {event.name} 执行成功")
            self._trigger_event_completed(event)
            
            # 根据结果自动生成后续事件
            await self._auto_generate_next_events(event)
            
        except Exception as e:
            event.error = str(e)
            event.status = EventStatus.FAILED
            event.completed_at = datetime.now()
            
            logger.error(f"❌ 事件 [{event.event_id}] {event.name} 执行失败: {e}")
            self._trigger_event_failed(event)
            
            # 处理失败重试
            await self._handle_event_failure(event)
        
        finally:
            self.event_history.append(event)
        
        return event
    
    def _find_agent_for_event(self, event_name: str) -> Optional[AgentInterface]:
        """查找能处理指定事件的Agent"""
        for agent in self.agents.values():
            if agent.can_handle(event_name):
                return agent
        return None
    
    async def _auto_generate_next_events(self, completed_event: Event):
        """根据完成的事件自动生成后续事件"""
        event_name = completed_event.name
        result = completed_event.result or {}
        
        # 事件流规则
        if event_name == "analysis":
            # 分析完成后，进入执行阶段
            if result.get("success"):
                execution_event = self.create_event(
                    "execution", 
                    {"analysis_result": result, **completed_event.payload}
                )
                self.add_event(execution_event)
                
        elif event_name == "execution":
            # 执行完成后，进入调试阶段
            debug_event = self.create_event(
                "debug",
                {"execution_result": result, **completed_event.payload}
            )
            self.add_event(debug_event)
            
        elif event_name == "debug":
            # 调试完成后，根据结果决定下一步
            if not result.get("success") and completed_event.retry_count < completed_event.max_retries:
                # 调试失败，重新执行
                retry_event = self.create_event(
                    "execution",
                    {"debug_feedback": result, **completed_event.payload},
                    retry_count=completed_event.retry_count + 1
                )
                self.add_event(retry_event)
            else:
                # 调试成功或达到最大重试次数，结束流程
                completion_event = self.create_event(
                    "completion",
                    {"final_result": result, **completed_event.payload}
                )
                self.add_event(completion_event)
    
    async def _handle_event_failure(self, failed_event: Event):
        """处理事件失败"""
        if failed_event.retry_count < failed_event.max_retries:
            # 重试事件
            retry_event = Event(
                name=failed_event.name,
                payload=failed_event.payload,
                retry_count=failed_event.retry_count + 1,
                max_retries=failed_event.max_retries
            )
            self.add_event(retry_event)
            logger.info(f"⏳ 事件 {failed_event.name} 将进行第 {retry_event.retry_count} 次重试")
        else:
            logger.error(f"💀 事件 {failed_event.name} 达到最大重试次数，执行失败")
    
    async def run_until_empty(self):
        """运行引擎直到队列为空"""
        self.running = True
        logger.info("🎬 事件引擎开始运行")
        
        try:
            while self.running and self.event_queue:
                await self.process_next_event()
                
                # 短暂暂停，避免CPU占用过高
                await asyncio.sleep(0.1)
        
        finally:
            self.running = False
            logger.info("⏹️ 事件引擎停止运行")
    
    async def run_single_cycle(self):
        """运行单个事件处理周期"""
        return await self.process_next_event()
    
    def stop(self):
        """停止引擎"""
        self.running = False
        logger.info("🛑 收到停止信号，事件引擎将停止")
    
    # 事件监听器系统
    def on(self, event_type: str, handler: Callable):
        """注册事件监听器"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def _trigger_event_added(self, event: Event):
        """触发事件添加监听器"""
        self._trigger_handlers("event_added", event)
    
    def _trigger_event_started(self, event: Event):
        """触发事件开始监听器"""
        self._trigger_handlers("event_started", event)
    
    def _trigger_event_completed(self, event: Event):
        """触发事件完成监听器"""
        self._trigger_handlers("event_completed", event)
    
    def _trigger_event_failed(self, event: Event):
        """触发事件失败监听器"""
        self._trigger_handlers("event_failed", event)
    
    def _trigger_handlers(self, event_type: str, event: Event):
        """触发指定类型的所有监听器"""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"事件监听器执行失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        total_events = len(self.event_history)
        successful_events = len([e for e in self.event_history if e.status == EventStatus.DONE])
        failed_events = len([e for e in self.event_history if e.status == EventStatus.FAILED])
        
        return {
            "total_events": total_events,
            "successful_events": successful_events,
            "failed_events": failed_events,
            "success_rate": successful_events / total_events if total_events > 0 else 0,
            "queue_length": len(self.event_queue),
            "running": self.running,
            "registered_agents": list(self.agents.keys())
        }
