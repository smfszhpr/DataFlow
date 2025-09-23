#!/usr/bin/env python3
"""
DataFlow WebSocket Server - FastAPI WebSocket服务器
提供实时事件推送的Web API接口
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from ..events import EventBuilder, PrintSink, CompositeSink, Event, EventType
from .events import connection_manager, event_router, WebSocketSink
from ..events import create_event_driven_master_agent, EventDrivenMasterAgentExecutor

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

# 全局执行器和状态同步
event_executor: Optional[EventDrivenMasterAgentExecutor] = None
form_state_sync_timer = None

# 全局状态存储 (简化版的Master Agent state同步)
global_agent_states = {}

# 🔥 新增：表单状态历史，用于检测变化
form_state_history = {}


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global event_executor, form_state_sync_timer
    
    try:
        # 创建事件驱动的Master Agent
        event_agent, event_executor = create_event_driven_master_agent()
        
        # 🎯 注册表单状态更新处理器
        async def handle_form_state_update(message: dict, session_id: str):
            """处理前端表单状态更新"""
            try:
                # 获取Master Agent状态
                agent_state = global_agent_states.get(session_id)
                if not agent_state:
                    logger.warning(f"⚠️ 会话 {session_id} 的Agent状态未找到")
                    return
                
                # 确保form_session存在
                if not hasattr(agent_state, 'form_session') or not agent_state.form_session:
                    logger.warning(f"⚠️ 会话 {session_id} 的form_session未找到")
                    return
                
                # 更新表单数据
                form_data = message.get('form_data', {})
                if 'fields' in form_data:
                    if not hasattr(agent_state.form_session, 'form_data'):
                        agent_state.form_session.form_data = {'fields': {}}
                    elif not agent_state.form_session.form_data:
                        agent_state.form_session.form_data = {'fields': {}}
                    elif 'fields' not in agent_state.form_session.form_data:
                        agent_state.form_session.form_data['fields'] = {}
                    
                    # 更新字段
                    agent_state.form_session.form_data['fields'].update(form_data['fields'])
                
                # 更新时间戳
                agent_state.form_session.updated_at = datetime.now().isoformat()
                
                logger.info(f"✅ 表单状态已更新: {session_id}")
                
            except Exception as e:
                logger.error(f"❌ 处理表单状态更新失败: {e}")
        
        # 注册处理器
        event_router.register_handler("form_state_update_handler", handle_form_state_update)
        event_router.register_handler("user_input_handler", handle_user_input)
        
        # 启动状态同步定时器
        form_state_sync_timer = asyncio.create_task(form_state_sync_loop())
        
        logger.info("🚀 DataFlow WebSocket服务器启动成功")
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 服务器启动失败: {e}"
        logger.error(error_msg)
        logger.error(f"❌ 详细错误信息:\n{traceback.format_exc()}")
        raise


async def form_state_sync_loop():
    """表单状态同步循环 - 每2秒同步一次"""
    while True:
        try:
            await asyncio.sleep(2)  # 每2秒同步一次
            await sync_form_states_to_clients()
        except Exception as e:
            logger.error(f"❌ 表单状态同步失败: {e}")


async def sync_form_states_to_clients():
    """同步表单状态到所有客户端（智能检测变化）"""
    if not global_agent_states:
        return
    
    # 获取所有活跃连接
    active_sessions = connection_manager.get_active_sessions()
    
    for session_id in active_sessions:
        if session_id in global_agent_states:
            state = global_agent_states[session_id]
            
            # 🎯 修复：处理不同的状态结构
            form_session = None
            if isinstance(state, dict):
                form_session = state.get("form_session", {})
                logger.debug(f"🔍 从dict state提取form_session: {form_session}")
                # 🔥 新增：如果AgentState中有tool_results，提取former的missing_params
                if "tool_results" in state:
                    logger.debug(f"🔍 检查tool_results，数量: {len(state['tool_results'])}")
                    for tool_result in state["tool_results"]:
                        if tool_result.get("tool") == "former" and tool_result.get("payload"):
                            payload = tool_result["payload"]
                            logger.debug(f"🔍 发现former工具结果，payload keys: {list(payload.keys())}")
                            if payload.get("missing_params"):
                                if not form_session:
                                    form_session = {}
                                form_session["missing_params"] = payload["missing_params"]
                                logger.debug(f"🔍 添加missing_params: {payload['missing_params']}")
                            if payload.get("extracted_params"):
                                if not form_session:
                                    form_session = {}
                                form_session["extracted_params"] = payload["extracted_params"]
                                logger.debug(f"🔍 添加extracted_params: {payload['extracted_params']}")
                            if payload.get("form_stage"):
                                if not form_session:
                                    form_session = {}
                                form_session["form_stage"] = payload["form_stage"]
                            logger.debug(f"🔍 更新后的form_session: {form_session}")
            elif hasattr(state, 'form_session'):
                # 如果是Agent状态对象
                form_session = {
                    "form_data": getattr(state.form_session, 'form_data', {"fields": {}}) if state.form_session else {"fields": {}},
                    "form_stage": getattr(state.form_session, 'form_stage', 'initial') if state.form_session else 'initial',
                    "updated_at": getattr(state.form_session, 'updated_at', datetime.now().isoformat()) if state.form_session else datetime.now().isoformat()
                }
                # 🔥 新增：提取missing_params
                if hasattr(state, 'tool_results'):
                    for tool_result in state.tool_results:
                        if tool_result.get("tool") == "former" and tool_result.get("payload"):
                            payload = tool_result["payload"]
                            if payload.get("missing_params"):
                                form_session["missing_params"] = payload["missing_params"]
                            if payload.get("extracted_params"):
                                form_session["extracted_params"] = payload["extracted_params"]
            
            # 🔥 新增：检测表单状态是否有实际变化
            if form_session:
                # 生成状态摘要用于比较
                current_state_summary = {
                    "missing_params": form_session.get("missing_params", []),
                    "extracted_params": form_session.get("extracted_params", {}),
                    "form_data": form_session.get("form_data", {"fields": {}}),
                    "form_stage": form_session.get("form_stage", "initial")
                }
                
                # 检查是否与历史状态相同
                if session_id in form_state_history:
                    last_state = form_state_history[session_id]
                    if current_state_summary == last_state:
                        logger.debug(f"🔍 会话 {session_id} 表单状态无变化，跳过同步")
                        continue
                
                # 更新历史状态
                form_state_history[session_id] = current_state_summary
                
                logger.debug(f"🔍 发送表单状态同步到会话: {session_id}")
                
                # 发送表单状态更新事件
                ws_sink = await connection_manager.get_connection(session_id)
                if ws_sink:
                    try:
                        await ws_sink.emit(Event(
                            type=EventType.STATE_UPDATE,
                            session_id=session_id,
                            timestamp=datetime.now(),
                            data={
                                "type": "form_state_sync",
                                "form_session": form_session
                            }
                        ))
                        logger.debug(f"🔄 表单状态已同步: {session_id}")
                    except Exception as e:
                        logger.error(f"❌ 发送表单状态失败 {session_id}: {e}")


async def handle_form_state_update(message: Dict[str, Any], session_id: str):
    """处理前端表单状态更新（简化版）"""
    try:
        # 获取表单数据更新
        form_data = message.get("form_data", {})
        
        # 更新全局状态
        if session_id not in global_agent_states:
            global_agent_states[session_id] = {}
        
        if "form_session" not in global_agent_states[session_id]:
            global_agent_states[session_id]["form_session"] = {}
        
        # 更新表单数据
        global_agent_states[session_id]["form_session"]["form_data"] = form_data
        global_agent_states[session_id]["form_session"]["updated_at"] = datetime.now().isoformat()
        
        # 🔥 新增：同时更新历史状态，防止同步循环立即覆盖
        current_form_session = global_agent_states[session_id]["form_session"]
        updated_state_summary = {
            "missing_params": current_form_session.get("missing_params", []),
            "extracted_params": current_form_session.get("extracted_params", {}),
            "form_data": form_data,
            "form_stage": current_form_session.get("form_stage", "initial")
        }
        form_state_history[session_id] = updated_state_summary
        
        logger.debug(f"✅ 表单状态更新完成: {session_id}")
        
    except Exception as e:
        logger.error(f"❌ 处理表单状态更新失败 {session_id}: {e}")


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
        
        logger.info(f"🎯 开始处理用户输入: {user_input} (会话: {session_id})")
        
        # 🔥 调试：检查全局状态
        logger.info(f"🔍 WebSocket调试 - global_agent_states中的会话: {list(global_agent_states.keys())}")
        logger.info(f"🔍 WebSocket调试 - 当前会话ID: {session_id}")
        logger.info(f"🔍 WebSocket调试 - 会话存在检查: {session_id in global_agent_states}")
        
        # 🔥 新增：构建初始状态，包含现有的表单状态
        initial_state = {
            "input": user_input,
            "session_id": session_id
        }
        
        # 检查是否有现有的agent状态（包括表单状态）
        if session_id in global_agent_states:
            existing_state = global_agent_states[session_id]
            logger.info(f"🔄 发现会话 {session_id} 的现有状态，keys: {list(existing_state.keys())}")
            
            # 传递现有的表单状态
            if "form_session" in existing_state:
                initial_state["form_session"] = existing_state["form_session"]
                form_data = existing_state["form_session"].get("form_data", {})
                logger.info(f"🔍 WebSocket调试 - form_session结构: {existing_state['form_session']}")
                if form_data:
                    if isinstance(form_data, dict) and "fields" in form_data:
                        logger.info(f"🔄 传递现有表单数据: {list(form_data['fields'].keys())}")
                        logger.info(f"🔄 表单字段值: {form_data['fields']}")
                    else:
                        logger.info(f"🔄 表单数据结构: {form_data}")
                else:
                    logger.info(f"🔄 表单会话存在但form_data为空")
            else:
                logger.info(f"🔄 现有状态中无form_session")
            
            # 传递其他相关状态
            for key in ["current_workflow_id", "tool_results"]:
                if key in existing_state:
                    initial_state[key] = existing_state[key]
        else:
            logger.info(f"🔄 会话 {session_id} 无现有状态，从空白开始")
            # 🎯 确保全局状态中有该会话的初始状态
            global_agent_states[session_id] = {
                "form_session": {
                    "form_data": {"fields": {}},
                    "form_stage": "initial",
                    "updated_at": datetime.now().isoformat()
                },
                "current_workflow_id": None,
                "session_id": session_id
            }
            logger.info(f"✅ 初始化会话状态: {session_id}")
        
        # 执行Agent（需要修改executor以支持initial_state）
        try:
            result = await event_executor.run_with_events(
                user_input=user_input,
                session_id=session_id,
                sink=composite_sink,
                initial_state=initial_state  # 🔥 传递初始状态
            )
        except TypeError:
            # 如果executor不支持initial_state，使用原始方法
            logger.warning("⚠️ Executor不支持initial_state参数，使用原始调用方式")
            result = await event_executor.run_with_events(
                user_input=user_input,
                session_id=session_id,
                sink=composite_sink
            )
        
        # 🎯 更新全局状态 - 从executor结果中提取完整的AgentState
        if result:
            logger.info(f"🔍 Agent执行结果类型: {type(result)}")
            
            # 尝试获取AgentState数据
            agent_state_data = None
            if hasattr(result, 'agent_state'):
                agent_state_data = result.agent_state
            elif hasattr(result, 'state'):
                agent_state_data = result.state
            elif isinstance(result, dict):
                agent_state_data = result
            
            logger.info(f"🔍 提取的agent_state_data类型: {type(agent_state_data)}")
            
            if agent_state_data:
                # 更新global_agent_states以包含完整的AgentState数据
                if isinstance(agent_state_data, dict):
                    # 如果是字典，直接更新
                    global_agent_states[session_id].update(agent_state_data)
                    logger.info(f"🔍 更新了global_agent_states[{session_id}]，keys: {list(global_agent_states[session_id].keys())}")
                else:
                    # 如果是对象，提取所需属性
                    if hasattr(agent_state_data, 'form_session'):
                        global_agent_states[session_id]["form_session"] = {
                            "form_data": getattr(agent_state_data.form_session, 'form_data', {"fields": {}}),
                            "form_stage": getattr(agent_state_data.form_session, 'form_stage', 'processing'),
                            "updated_at": datetime.now().isoformat()
                        }
                    
                    if hasattr(agent_state_data, 'current_workflow_id'):
                        global_agent_states[session_id]["current_workflow_id"] = agent_state_data.current_workflow_id
                    
                    if hasattr(agent_state_data, 'tool_results'):
                        global_agent_states[session_id]["tool_results"] = agent_state_data.tool_results
                        logger.info(f"🔍 保存了tool_results，数量: {len(agent_state_data.tool_results)}")
            else:
                logger.warning(f"⚠️ 无法从result中提取agent_state_data")
        
        logger.info(f"✅ 用户输入处理完成: {session_id}")
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 处理用户输入失败 {session_id}: {e}"
        logger.error(error_msg)
        logger.error(f"❌ 详细错误信息:\n{traceback.format_exc()}")
        # 发送错误事件
        ws_sink = await connection_manager.get_connection(session_id)
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
    # 读取HTML模板文件
    html_file_path = Path(__file__).parent / "templates" / "test.html"
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="""
<html>
<body>
<h1>DataFlow Agent WebSocket Server</h1>
<p>HTML模板文件未找到。请确保 templates/test.html 文件存在。</p>
<p>WebSocket端点: <code>ws://localhost:8000/ws/agent/{session_id}</code></p>
</body>
</html>
        """)


@app.get("/test")
async def test_page():
    """测试页面 - 和根路径相同"""
    return await root()


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
