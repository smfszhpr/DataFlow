#!/usr/bin/env python3
"""
DataFlow Agent Events System - 事件驱动架构
实现事件协议、EventSink抽象和各种事件类型
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
import json
import asyncio


class EventType(str, Enum):
    """事件类型枚举"""
    # 执行生命周期事件
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_ERROR = "run_error"
    
    # 工具执行事件
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TOOL_ERROR = "tool_error"
    
    # 规划决策事件
    PLAN_STARTED = "plan_started"
    PLAN_DECISION = "plan_decision"
    PLAN_ERROR = "plan_error"
    
    # 总结事件
    SUMMARIZE_STARTED = "summarize_started"
    SUMMARIZE_FINISHED = "summarize_finished"
    
    # 状态更新事件
    STATE_UPDATE = "state_update"
    
    # 实时状态事件
    HEARTBEAT = "heartbeat"
    

class Event(BaseModel):
    """通用事件结构"""
    type: EventType = Field(..., description="事件类型")
    timestamp: datetime = Field(default_factory=datetime.now, description="事件时间戳")
    session_id: str = Field(..., description="会话ID")
    step_id: Optional[str] = Field(None, description="步骤ID，用于事件排序和去重")
    data: Dict[str, Any] = Field(default_factory=dict, description="事件数据")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def model_dump_json(self, **kwargs) -> str:
        """序列化为JSON"""
        return json.dumps(self.model_dump(), default=str, ensure_ascii=False)


class EventSink(ABC):
    """事件接收器抽象接口"""
    
    @abstractmethod
    async def emit(self, event: Event) -> None:
        """发送事件"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """关闭接收器"""
        pass


class PrintSink(EventSink):
    """控制台打印事件接收器"""
    
    def __init__(self, prefix: str = "🔔"):
        self.prefix = prefix
    
    async def emit(self, event: Event) -> None:
        """打印事件到控制台"""
        timestamp = event.timestamp.strftime("%H:%M:%S")
        print(f"{self.prefix} [{timestamp}] {event.type.value}: {event.data}")
    
    async def close(self) -> None:
        """关闭（无需操作）"""
        pass


class BufferedSink(EventSink):
    """缓冲事件接收器 - 用于测试和批处理"""
    
    def __init__(self):
        self.events: List[Event] = []
        self._lock = asyncio.Lock()
    
    async def emit(self, event: Event) -> None:
        """缓冲事件"""
        async with self._lock:
            self.events.append(event)
    
    async def close(self) -> None:
        """关闭（无需操作）"""
        pass
    
    def get_events(self) -> List[Event]:
        """获取所有事件"""
        return self.events.copy()
    
    def clear(self) -> None:
        """清空事件缓冲"""
        self.events.clear()


class CompositeSink(EventSink):
    """组合事件接收器 - 支持同时发送到多个接收器"""
    
    def __init__(self, sinks: List[EventSink]):
        self.sinks = sinks
    
    async def emit(self, event: Event) -> None:
        """发送到所有接收器"""
        tasks = [sink.emit(event) for sink in self.sinks]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def close(self) -> None:
        """关闭所有接收器"""
        tasks = [sink.close() for sink in self.sinks]
        await asyncio.gather(*tasks, return_exceptions=True)


class EventBuilder:
    """事件构建器 - 简化事件创建"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.step_counter = 0
    
    def _next_step_id(self) -> str:
        """生成下一个步骤ID"""
        self.step_counter += 1
        return f"step_{self.step_counter}"
    
    def run_started(self, user_input: str, **extra) -> Event:
        """创建执行开始事件"""
        return Event(
            type=EventType.RUN_STARTED,
            session_id=self.session_id,
            step_id=self._next_step_id(),
            data={"user_input": user_input, **extra}
        )
    
    def run_finished(self, result: Any, **extra) -> Event:
        """创建执行完成事件"""
        return Event(
            type=EventType.RUN_FINISHED,
            session_id=self.session_id,
            step_id=self._next_step_id(),
            data={"result": result, **extra}
        )
    
    def run_error(self, error: str, **extra) -> Event:
        """创建执行错误事件"""
        return Event(
            type=EventType.RUN_ERROR,
            session_id=self.session_id,
            step_id=self._next_step_id(),
            data={"error": error, **extra}
        )
    
    def tool_started(self, tool_name: str, tool_input: Dict[str, Any], **extra) -> Event:
        """创建工具开始事件"""
        return Event(
            type=EventType.TOOL_STARTED,
            session_id=self.session_id,
            step_id=self._next_step_id(),
            data={"tool_name": tool_name, "tool_input": tool_input, **extra}
        )
    
    def tool_finished(self, tool_name: str, tool_output: Any, **extra) -> Event:
        """创建工具完成事件"""
        return Event(
            type=EventType.TOOL_FINISHED,
            session_id=self.session_id,
            step_id=self._next_step_id(),
            data={"tool_name": tool_name, "tool_output": tool_output, **extra}
        )
    
    def tool_error(self, tool_name: str, error: str, **extra) -> Event:
        """创建工具错误事件"""
        return Event(
            type=EventType.TOOL_ERROR,
            session_id=self.session_id,
            step_id=self._next_step_id(),
            data={"tool_name": tool_name, "error": error, **extra}
        )
    
    def plan_started(self, **extra) -> Event:
        """创建规划开始事件"""
        return Event(
            type=EventType.PLAN_STARTED,
            session_id=self.session_id,
            step_id=self._next_step_id(),
            data={**extra}
        )
    
    def plan_decision(self, decision: Dict[str, Any], **extra) -> Event:
        """创建规划决策事件"""
        return Event(
            type=EventType.PLAN_DECISION,
            session_id=self.session_id,
            step_id=self._next_step_id(),
            data={"decision": decision, **extra}
        )
    
    def plan_error(self, error: str, **extra) -> Event:
        """创建规划错误事件"""
        return Event(
            type=EventType.PLAN_ERROR,
            session_id=self.session_id,
            step_id=self._next_step_id(),
            data={"error": error, **extra}
        )
    
    def summarize_started(self, **extra) -> Event:
        """创建总结开始事件"""
        return Event(
            type=EventType.SUMMARIZE_STARTED,
            session_id=self.session_id,
            step_id=self._next_step_id(),
            data={**extra}
        )
    
    def summarize_finished(self, summary: str, **extra) -> Event:
        """创建总结完成事件"""
        return Event(
            type=EventType.SUMMARIZE_FINISHED,
            session_id=self.session_id,
            step_id=self._next_step_id(),
            data={"summary": summary, **extra}
        )
    
    def state_update(self, state_info: Dict[str, Any], **extra) -> Event:
        """创建状态更新事件"""
        return Event(
            type=EventType.STATE_UPDATE,
            session_id=self.session_id,
            step_id=self._next_step_id(),
            data={"state_info": state_info, **extra}
        )
    
    def heartbeat(self, **extra) -> Event:
        """创建心跳事件"""
        return Event(
            type=EventType.HEARTBEAT,
            session_id=self.session_id,
            data={**extra}
        )


# 导出主要类和函数
__all__ = [
    'EventType',
    'Event',
    'EventSink',
    'PrintSink',
    'BufferedSink',
    'CompositeSink',
    'EventBuilder'
]
