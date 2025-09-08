"""
测试Former Agent和事件引擎的WebUI接口
提供前端调用测试接口
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import os

from dataflow.agent.agentrole.former import FormerAgent
from dataflow.agent.xmlforms.models import FormRequest, FormResponse
from dataflow.agent.xmlforms.form_templates import FormTemplateManager
from dataflow.agent.eventengine.manager import EventManager, quick_execute
from dataflow.agent.toolkits import ChatAgentRequest
from dataflow import get_logger

logger = get_logger()

# Pydantic模型定义
class ConversationRequest(BaseModel):
    user_query: str
    session_id: str = None
    conversation_history: List[Dict[str, str]] = []

class ConversationResponse(BaseModel):
    session_id: str
    agent_response: str
    need_more_info: bool
    xml_form: str = None
    form_type: str
    conversation_history: List[Dict[str, str]]

class EventTestRequest(BaseModel):
    workflow_name: str
    user_query: str
    session_id: str = None

class EventTestResponse(BaseModel):
    workflow: str
    status: str
    events_count: int
    result: Dict[str, Any]
    event_history: List[Dict[str, Any]]

# 创建FastAPI应用
app = FastAPI(title="Former Agent & Event Engine Test API")

# 全局变量
former_agent = None
event_manager = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化组件"""
    global former_agent, event_manager
    
    # 初始化Former Agent
    template_manager = FormTemplateManager()
    former_agent = FormerAgent(template_manager)
    
    # 初始化Event Manager
    event_manager = EventManager()
    
    logger.info("Former Agent和Event Engine初始化完成")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Former Agent & Event Engine Test API",
        "status": "running",
        "endpoints": {
            "conversation": "/conversation",
            "event_test": "/event_test",
            "event_status": "/event_status",
            "available_forms": "/available_forms",
            "available_workflows": "/available_workflows"
        }
    }

@app.post("/conversation", response_model=ConversationResponse)
async def handle_conversation(request: ConversationRequest):
    """处理与Former Agent的对话"""
    try:
        # 创建FormRequest
        form_request = FormRequest(
            user_query=request.user_query,
            session_id=request.session_id,
            conversation_history=request.conversation_history
        )
        
        # 调用Former Agent
        response = await former_agent.process_conversation(form_request)
        
        return ConversationResponse(
            session_id=form_request.session_id,
            agent_response=response.agent_response,
            need_more_info=response.need_more_info,
            xml_form=response.xml_form,
            form_type=response.form_type,
            conversation_history=response.conversation_history
        )
        
    except Exception as e:
        logger.error(f"对话处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/event_test", response_model=EventTestResponse)
async def test_event_workflow(request: EventTestRequest):
    """测试事件驱动工作流"""
    try:
        # 创建ChatAgentRequest
        chat_request = ChatAgentRequest(
            language="zh",
            target=request.user_query,
            model="deepseek-v3",
            sessionKEY=request.session_id or str(uuid.uuid4())
        )
        
        # 执行工作流
        result = await event_manager.execute_workflow(
            request.workflow_name, 
            {"request": chat_request}
        )
        
        return EventTestResponse(
            workflow=request.workflow_name,
            status="completed",
            events_count=len(result.get("events_history", [])),
            result=result,
            event_history=result.get("events_history", [])
        )
        
    except Exception as e:
        logger.error(f"事件工作流测试失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/event_status")
async def get_event_status():
    """获取事件引擎状态"""
    try:
        queue_status = event_manager.get_queue_status()
        statistics = event_manager.get_statistics()
        
        return {
            "queue_status": queue_status,
            "statistics": statistics,
            "engine_running": True
        }
        
    except Exception as e:
        logger.error(f"获取事件状态失败: {e}")
        return {"error": str(e), "engine_running": False}

@app.get("/available_forms")
async def get_available_forms():
    """获取可用的表单类型"""
    try:
        forms = former_agent.template_manager.get_available_forms()
        return {"available_forms": forms}
        
    except Exception as e:
        logger.error(f"获取可用表单失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available_workflows")
async def get_available_workflows():
    """获取可用的事件工作流模板"""
    try:
        workflows = list(event_manager.workflow_templates.keys())
        return {"available_workflows": workflows}
        
    except Exception as e:
        logger.error(f"获取可用工作流失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute_xml_form")
async def execute_xml_form(xml_content: str, form_type: str):
    """执行XML表单"""
    try:
        # 使用事件管理器的便捷函数处理XML表单
        from dataflow.agent.eventengine.manager import handle_xml_form_execution
        
        result = await handle_xml_form_execution(xml_content, form_type)
        
        return {
            "status": "executed",
            "xml_content": xml_content,
            "form_type": form_type,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"XML表单执行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/conversation")
async def get_conversation_history(session_id: str):
    """获取会话的对话历史"""
    try:
        if session_id in former_agent.conversations:
            return {
                "session_id": session_id,
                "conversation_history": former_agent.conversations[session_id],
                "session_state": former_agent.session_states.get(session_id, {})
            }
        else:
            return {
                "session_id": session_id,
                "conversation_history": [],
                "session_state": {}
            }
            
    except Exception as e:
        logger.error(f"获取对话历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """清空会话"""
    try:
        if session_id in former_agent.conversations:
            del former_agent.conversations[session_id]
        
        if session_id in former_agent.session_states:
            del former_agent.session_states[session_id]
        
        return {"message": f"会话 {session_id} 已清空"}
        
    except Exception as e:
        logger.error(f"清空会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream_conversation")
async def stream_conversation(request: ConversationRequest):
    """流式对话处理"""
    async def generate_stream():
        try:
            # 创建FormRequest
            form_request = FormRequest(
                user_query=request.user_query,
                session_id=request.session_id,
                conversation_history=request.conversation_history
            )
            
            # 模拟流式处理过程
            yield f"data: {json.dumps({'event': 'start', 'message': '开始处理用户请求'})}\n\n"
            
            # 执行对话处理
            response = await former_agent.process_conversation(form_request)
            
            yield f"data: {json.dumps({'event': 'processing', 'message': '正在分析用户需求'})}\n\n"
            
            # 返回最终结果
            result = {
                "event": "complete",
                "data": {
                    "session_id": form_request.session_id,
                    "agent_response": response.agent_response,
                    "need_more_info": response.need_more_info,
                    "xml_form": response.xml_form,
                    "form_type": response.form_type,
                    "conversation_history": response.conversation_history
                }
            }
            
            yield f"data: {json.dumps(result)}\n\n"
            yield "data: {\"event\": \"done\"}\n\n"
            
        except Exception as e:
            error_result = {
                "event": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_result)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )

@app.post("/stream_event_workflow")
async def stream_event_workflow(request: EventTestRequest):
    """流式事件工作流处理"""
    async def generate_event_stream():
        try:
            yield f"data: {json.dumps({'event': 'start', 'workflow': request.workflow_name})}\n\n"
            
            # 创建ChatAgentRequest
            chat_request = ChatAgentRequest(
                language="zh",
                target=request.user_query,
                model="deepseek-v3",
                sessionKEY=request.session_id or str(uuid.uuid4())
            )
            
            # 设置事件监听器来流式输出
            events_processed = []
            
            def on_event_progress(event_data):
                events_processed.append(event_data)
                return f"data: {json.dumps({'event': 'progress', 'data': event_data})}\n\n"
            
            # 执行工作流
            result = await event_manager.execute_workflow(
                request.workflow_name,
                {"request": chat_request}
            )
            
            # 发送最终结果
            final_result = {
                "event": "complete",
                "workflow": request.workflow_name,
                "result": result
            }
            
            yield f"data: {json.dumps(final_result)}\n\n"
            yield "data: {\"event\": \"done\"}\n\n"
            
        except Exception as e:
            error_result = {
                "event": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_result)}\n\n"
    
    return StreamingResponse(
        generate_event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
