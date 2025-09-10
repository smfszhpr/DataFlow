"""
Former Agent API测试接口
独立测试Former Agent的表单生成功能
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import asyncio

from dataflow.agent.mcp.former_mcp import FormerAgentMCP, MCPRequest, MCPResponse

app = FastAPI(title="Former Agent Test API", version="1.0.0")

class TestRequest(BaseModel):
    """测试请求格式"""
    action: str
    user_input: str
    session_id: str = "test_session"
    context: Dict[str, Any] = {}

class TestResponse(BaseModel):
    """测试响应格式"""
    status: str
    message: str
    data: Dict[str, Any]
    next_action: str = None

# 全局Former Agent实例
former_mcp = FormerAgentMCP()

@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "Former Agent Test API",
        "version": "1.0.0",
        "endpoints": [
            "/test/analyze - 分析用户需求",
            "/test/check_fields - 检查字段完整性", 
            "/test/generate_xml - 生成XML表单",
            "/test/get_form_state - 获取当前表单状态",
            "/test/reset - 重置会话状态"
        ]
    }

@app.post("/test/analyze")
async def test_analyze_requirement(request: TestRequest):
    """测试需求分析"""
    try:
        mcp_request = MCPRequest(
            action="analyze_requirement",
            user_input=request.user_input,
            context={"session_id": request.session_id, **request.context}
        )
        
        response = await former_mcp.process_mcp_request(mcp_request)
        
        return TestResponse(
            status=response.status,
            message=response.message,
            data=response.data,
            next_action=response.next_action
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/check_fields")
async def test_check_fields(request: TestRequest):
    """测试字段检查"""
    try:
        mcp_request = MCPRequest(
            action="check_fields",
            user_input=request.user_input,
            context={"session_id": request.session_id, **request.context}
        )
        
        response = await former_mcp.process_mcp_request(mcp_request)
        
        return TestResponse(
            status=response.status,
            message=response.message,
            data=response.data,
            next_action=response.next_action
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/generate_xml")
async def test_generate_xml(request: TestRequest):
    """测试XML生成"""
    try:
        mcp_request = MCPRequest(
            action="generate_xml",
            user_input=request.user_input,
            context={"session_id": request.session_id, **request.context}
        )
        
        response = await former_mcp.process_mcp_request(mcp_request)
        
        return TestResponse(
            status=response.status,
            message=response.message,
            data=response.data,
            next_action=response.next_action
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/get_form_state/{session_id}")
async def test_get_form_state(session_id: str):
    """测试获取表单状态"""
    try:
        mcp_request = MCPRequest(
            action="get_current_form",
            user_input="",
            context={"session_id": session_id}
        )
        
        response = await former_mcp.process_mcp_request(mcp_request)
        
        return TestResponse(
            status=response.status,
            message=response.message,
            data=response.data,
            next_action=response.next_action
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/test/reset/{session_id}")
async def test_reset_session(session_id: str):
    """重置会话状态"""
    try:
        if session_id in former_mcp.form_memory:
            del former_mcp.form_memory[session_id]
        
        return {"message": f"会话 {session_id} 已重置", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/sessions")
async def list_sessions():
    """列出所有会话"""
    try:
        sessions = {}
        for session_id, form_state in former_mcp.form_memory.items():
            sessions[session_id] = {
                "form_type": form_state.get('form_type'),
                "inputs_count": len(form_state.get('user_inputs', [])),
                "collected_fields": list(form_state.get('extracted_fields', {}).keys())
            }
        
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 启动Former Agent测试API...")
    print("📖 测试方法:")
    print("1. POST /test/analyze - 发送用户需求")
    print("2. POST /test/check_fields - 检查字段")
    print("3. POST /test/generate_xml - 生成XML")
    print("4. GET /test/get_form_state/session_id - 查看状态")
    print("5. DELETE /test/reset/session_id - 重置会话")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
