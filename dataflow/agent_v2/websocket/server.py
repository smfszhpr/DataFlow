#!/usr/bin/env python3
"""
DataFlow WebSocket Server - FastAPI WebSocket服务器
提供实时事件推送的Web API接口
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from ..events import EventBuilder, PrintSink, CompositeSink
from .events import connection_manager, event_router, WebSocketSink
from ..events import create_event_driven_master_agent, EventDrivenMasterAgentExecutor
from ..master.agent import create_master_agent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="DataFlow Agent WebSocket API",
    description="实时事件驱动的Agent执行API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局执行器
event_executor: Optional[EventDrivenMasterAgentExecutor] = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global event_executor
    
    try:
        # 创建事件驱动的Master Agent
        event_agent, event_executor = create_event_driven_master_agent()
        
        # 注册用户输入处理器
        event_router.register_handler("user_input_handler", handle_user_input)
        
        logger.info("🚀 DataFlow WebSocket服务器启动成功")
        
    except Exception as e:
        logger.error(f"❌ 服务器启动失败: {e}")
        raise


async def handle_user_input(user_input: str, session_id: str):
    """处理用户输入的异步函数"""
    global event_executor
    
    if not event_executor:
        logger.error("❌ 执行器未初始化")
        return
    
    try:
        # 获取WebSocket连接
        ws_sink = await connection_manager.get_connection(session_id)
        if not ws_sink:
            logger.warning(f"⚠️ 会话 {session_id} 的WebSocket连接不存在")
            return
        
        # 创建组合事件接收器 (WebSocket + 控制台)
        print_sink = PrintSink(f"🎭[{session_id[:8]}]")
        composite_sink = CompositeSink([ws_sink, print_sink])
        
        logger.debug(f"🎯 开始处理用户输入: {user_input} (会话: {session_id})")
        
        # 执行Agent
        result = await event_executor.run_with_events(
            user_input=user_input,
            session_id=session_id,
            sink=composite_sink
        )
        
        logger.debug(f"✅ 用户输入处理完成: {session_id}")
        
    except Exception as e:
        logger.error(f"❌ 处理用户输入失败 {session_id}: {e}")
        # 发送错误事件
        if ws_sink:
            event_builder = EventBuilder(session_id)
            await ws_sink.emit(event_builder.run_error(f"处理失败: {str(e)}"))


@app.websocket("/ws/agent/{session_id}")
async def websocket_agent_endpoint(websocket: WebSocket, session_id: str):
    """Agent WebSocket端点"""
    
    # 建立WebSocket连接
    ws_sink = await connection_manager.connect(websocket, session_id)
    
    try:
        logger.info(f"🔗 WebSocket连接建立: {session_id}")
        
        # 发送连接成功事件
        event_builder = EventBuilder(session_id)
        await ws_sink.emit(event_builder.state_update({
            "status": "connected",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "server_info": {
                "version": "1.0.0",
                "capabilities": ["real_time_events", "multi_round_planning", "tool_execution"]
            }
        }))
        
        # 消息接收循环
        while True:
            try:
                # 接收消息
                message = await websocket.receive_json()
                logger.debug(f"📨 收到消息 {session_id}: {message}")
                
                # 路由消息到处理器
                await event_router.handle_message(websocket, session_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket正常断开: {session_id}")
                break
            except Exception as e:
                logger.error(f"❌ 处理WebSocket消息失败 {session_id}: {e}")
                # 发送错误消息给客户端
                await websocket.send_json({
                    "type": "error",
                    "error": f"服务器错误: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
    
    except Exception as e:
        logger.error(f"❌ WebSocket连接错误 {session_id}: {e}")
    
    finally:
        # 清理连接
        await connection_manager.disconnect(session_id)


@app.get("/")
async def root():
    """根路径 - 返回简单的测试页面"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>DataFlow Agent WebSocket Test</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .event { margin: 5px 0; padding: 8px; background: #f5f5f5; border-radius: 4px; font-family: monospace; white-space: pre-wrap; word-wrap: break-word; }
        .event.run_started { background: #e8f5e8; border-left: 4px solid #4caf50; }
        .event.tool_started { background: #e3f2fd; border-left: 4px solid #2196f3; }
        .event.tool_finished { background: #e8f5e8; border-left: 4px solid #4caf50; }
        .event.plan_decision { background: #fff3e0; border-left: 4px solid #ff9800; }
        .event.summarize_finished { background: #f3e5f5; border-left: 4px solid #9c27b0; }
        .event.run_finished { background: #e8f5e8; border-left: 4px solid #4caf50; font-weight: bold; }
        .event.state_update { background: #f0f0f0; border-left: 4px solid #607d8b; }
        .event.error, .event.run_error { background: #ffebee; border-left: 4px solid #f44336; }
        .event.user_input { background: #e1f5fe; border-left: 4px solid #00bcd4; }
        input, button { padding: 10px; margin: 5px; }
        #input { width: 400px; }
        #events { height: 500px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 DataFlow Agent WebSocket 测试</h1>
        <p><strong>连接状态:</strong> <span id="status">未连接</span></p>
        
        <div>
            <input type="text" id="input" placeholder="输入您的问题..." />
            <button onclick="sendMessage()">发送</button>
            <button onclick="clearEvents()">清除</button>
        </div>
        
        <h3>📡 实时事件:</h3>
        <div id="events"></div>
    </div>

    <script>
        let ws = null;
        const sessionId = 'test_' + Math.random().toString(36).substr(2, 9);
        
        function connect() {
            ws = new WebSocket(`ws://localhost:8000/ws/agent/${sessionId}`);
            
            ws.onopen = function(event) {
                document.getElementById('status').textContent = '已连接';
                document.getElementById('status').style.color = 'green';
                addEvent('system', '🔗 WebSocket连接成功');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                addEvent(data.type, formatEvent(data));
            };
            
            ws.onclose = function(event) {
                document.getElementById('status').textContent = '连接断开';
                document.getElementById('status').style.color = 'red';
                addEvent('system', '🔌 WebSocket连接断开');
            };
            
            ws.onerror = function(error) {
                addEvent('error', '❌ 连接错误: ' + error);
            };
        }
        
        function sendMessage() {
            const input = document.getElementById('input');
            if (ws && input.value.trim()) {
                ws.send(JSON.stringify({
                    type: 'user_input',
                    input: input.value.trim()
                }));
                addEvent('user_input', '👤 ' + input.value);
                input.value = '';
            }
        }
        
        function formatEvent(data) {
            const time = new Date(data.timestamp).toLocaleTimeString();
            let content = `[${time}] ${data.type}`;
            
            if (data.data) {
                // 根据事件类型格式化详细信息
                if (data.type === 'tool_started') {
                    content += ` - 开始执行: ${data.data.tool_name}`;
                    if (data.data.tool_input) {
                        const input = JSON.stringify(data.data.tool_input);
                        content += ` | 输入: ${input}`;
                    }
                } else if (data.type === 'tool_finished') {
                    content += ` - 完成: ${data.data.tool_name}`;
                    if (data.data.tool_output) {
                        const output = data.data.tool_output;
                        if (output.apikey) {
                            content += ` | API密钥: ${output.apikey}`;
                        }
                        if (output.result) {
                            content += ` | 结果: ${output.result}`;
                        }
                        if (output.message) {
                            content += ` | 消息: ${output.message}`;
                        }
                    }
                } else if (data.type === 'plan_decision') {
                    content += ` - 决策: ${JSON.stringify(data.data.decision || data.data)}`;
                } else if (data.type === 'summarize_finished') {
                    content += ` - 总结完成`;
                    if (data.data.summary) {
                        content += ` | ${data.data.summary}`;
                    }
                } else if (data.type === 'run_finished') {
                    content += ' - 执行完成';
                    if (data.data.result) {
                        content += ` | 最终结果: ${data.data.result}`;
                    }
                } else if (data.type === 'state_update') {
                    content += ` - 状态: ${data.data.state_info?.phase || data.data.phase || '未知'}`;
                }
            }
            
            return content;
        }
        
        function addEvent(type, content) {
            const events = document.getElementById('events');
            const div = document.createElement('div');
            div.className = `event ${type}`;
            div.textContent = content;
            events.appendChild(div);
            events.scrollTop = events.scrollHeight;
        }
        
        function clearEvents() {
            document.getElementById('events').innerHTML = '';
        }
        
        // 自动连接
        connect();
        
        // 回车发送
        document.getElementById('input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
    """)


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connections": connection_manager.get_connection_count(),
        "active_sessions": connection_manager.get_active_sessions()
    }


@app.get("/api/stats")
async def get_stats():
    """获取服务器统计信息"""
    return {
        "active_connections": connection_manager.get_connection_count(),
        "active_sessions": connection_manager.get_active_sessions(),
        "server_uptime": datetime.now().isoformat(),
        "executor_available": event_executor is not None
    }


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """运行WebSocket服务器"""
    logger.info(f"🚀 启动DataFlow WebSocket服务器: http://{host}:{port}")
    uvicorn.run(
        "dataflow.agent_v2.websocket.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


def start_websocket_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """启动WebSocket服务器 - 别名函数以保持向后兼容"""
    return run_server(host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server(reload=True)
