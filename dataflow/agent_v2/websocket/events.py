#!/usr/bin/env python3
"""
DataFlow WebSocket Events System - WebSocket事件传输
实现WebSocket事件接收器和连接管理
"""

from typing import Dict, Optional, Any, List
import json
import asyncio
import logging
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from ..events import EventSink, Event

logger = logging.getLogger(__name__)


class WebSocketSink(EventSink):
    """WebSocket事件接收器"""
    
    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.closed = False
        self._send_lock = asyncio.Lock()
    
    async def emit(self, event: Event) -> None:
        """通过WebSocket发送事件"""
        if self.closed:
            return
        
        try:
            async with self._send_lock:
                if not self.closed:
                    # 发送事件JSON
                    event_data = event.model_dump()
                    # 确保datetime正确序列化
                    event_data['timestamp'] = event.timestamp.isoformat()
                    
                    await self.websocket.send_json(event_data)
                    logger.debug(f"📤 发送事件到 {self.session_id}: {event.type.value}")
        
        except Exception as e:
            logger.error(f"❌ WebSocket发送事件失败 {self.session_id}: {e}")
            self.closed = True
    
    async def close(self) -> None:
        """关闭WebSocket连接"""
        if not self.closed:
            self.closed = True
            try:
                await self.websocket.close()
                logger.info(f"🔌 WebSocket连接已关闭: {self.session_id}")
            except Exception as e:
                logger.error(f"❌ 关闭WebSocket连接失败: {e}")


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 存储活跃连接: session_id -> WebSocketSink
        self.active_connections: Dict[str, WebSocketSink] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, session_id: str) -> WebSocketSink:
        """建立新的WebSocket连接"""
        await websocket.accept()
        
        async with self._lock:
            # 如果已有连接，先关闭旧连接
            if session_id in self.active_connections:
                await self.active_connections[session_id].close()
            
            # 创建新的WebSocket接收器
            sink = WebSocketSink(websocket, session_id)
            self.active_connections[session_id] = sink
            
            logger.info(f"🔗 新WebSocket连接: {session_id}")
            return sink
    
    async def disconnect(self, session_id: str) -> None:
        """断开WebSocket连接"""
        async with self._lock:
            if session_id in self.active_connections:
                await self.active_connections[session_id].close()
                del self.active_connections[session_id]
                logger.info(f"🔌 WebSocket连接断开: {session_id}")
    
    async def get_connection(self, session_id: str) -> Optional[WebSocketSink]:
        """获取指定的WebSocket连接"""
        async with self._lock:
            return self.active_connections.get(session_id)
    
    async def broadcast(self, event: Event, exclude_session: Optional[str] = None) -> None:
        """广播事件到所有连接"""
        async with self._lock:
            connections = list(self.active_connections.items())
        
        # 并发发送到所有连接
        tasks = []
        for session_id, sink in connections:
            if exclude_session and session_id == exclude_session:
                continue
            tasks.append(sink.emit(event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_active_sessions(self) -> List[str]:
        """获取所有活跃会话ID"""
        return list(self.active_connections.keys())
    
    def get_connection_count(self) -> int:
        """获取活跃连接数"""
        return len(self.active_connections)


# 全局连接管理器实例
connection_manager = ConnectionManager()


class WebSocketEventRouter:
    """WebSocket事件路由器 - 处理从前端接收的事件"""
    
    def __init__(self):
        self.handlers: Dict[str, Any] = {}
    
    def register_handler(self, event_type: str, handler: Any) -> None:
        """注册事件处理器"""
        self.handlers[event_type] = handler
        logger.info(f"📝 注册事件处理器: {event_type}")
    
    async def handle_message(self, websocket: WebSocket, session_id: str, message: Dict[str, Any]) -> None:
        """处理来自WebSocket的消息"""
        try:
            msg_type = message.get("type", "")
            
            if msg_type == "ping":
                # 处理心跳
                await websocket.send_json({
                    "type": "pong", 
                    "timestamp": datetime.now().isoformat()
                })
            
            elif msg_type == "user_input":
                # 处理用户输入
                user_input = message.get("input", "")
                if user_input and "user_input_handler" in self.handlers:
                    handler = self.handlers["user_input_handler"]
                    # 异步执行处理器，避免阻塞WebSocket接收循环
                    asyncio.create_task(handler(user_input, session_id))
            
            else:
                logger.warning(f"⚠️ 未知消息类型: {msg_type}")
        
        except Exception as e:
            logger.error(f"❌ 处理WebSocket消息失败: {e}")
            await websocket.send_json({
                "type": "error",
                "error": f"处理消息失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })


# 全局事件路由器实例
event_router = WebSocketEventRouter()


# 导出主要类和实例
__all__ = [
    'WebSocketSink',
    'ConnectionManager',
    'WebSocketEventRouter',
    'connection_manager',
    'event_router'
]
