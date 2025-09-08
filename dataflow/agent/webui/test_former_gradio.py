#!/usr/bin/env python3
"""
Former Agent & Event Engine Gradio测试界面
基于现有的Gradio界面模式创建
"""

import os
import json
import asyncio
import uuid
import contextlib
import requests
from typing import Dict, Any, Generator, Tuple

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
import uvicorn

from dataflow.agent.agentrole.former import FormerAgent
from dataflow.agent.xmlforms.models import FormRequest, FormResponse
from dataflow.agent.xmlforms.form_templates import FormTemplateManager
from dataflow.agent.eventengine.manager import EventManager, quick_execute
from dataflow.agent.toolkits import ChatAgentRequest
from dataflow import get_logger

logger = get_logger()

# 全局变量
former_agent = None
event_manager = None

def init_agents():
    """初始化Agent"""
    global former_agent, event_manager
    
    if former_agent is None:
        former_agent = FormerAgent()
        logger.info("Former Agent初始化完成")
    
    if event_manager is None:
        event_manager = EventManager()
        logger.info("Event Manager初始化完成")

# ------------------------------------------------------------------
# Former Agent 相关函数
# ------------------------------------------------------------------
def handle_former_conversation(
    user_query: str,
    session_id: str,
    conversation_history_json: str,
) -> Tuple[str, str, str, str, str]:
    """处理Former Agent对话"""
    
    init_agents()
    
    if not user_query.strip():
        return "❌ 错误", "请输入查询内容", "", "", ""
    
    try:
        # 解析对话历史
        conversation_history = []
        if conversation_history_json.strip():
            try:
                conversation_history = json.loads(conversation_history_json)
            except json.JSONDecodeError:
                pass
        
        # 处理session_id，确保符合Pydantic要求
        processed_session_id = None
        if session_id and session_id.strip():
            processed_session_id = session_id.strip()
        
        # 创建FormRequest
        form_request = FormRequest(
            user_query=user_query,
            session_id=processed_session_id,
            conversation_history=conversation_history
        )
        
        # 同步调用异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(former_agent.process_conversation(form_request))
        finally:
            loop.close()
        
        # 格式化返回结果
        status = "✅ 成功" if not response.need_more_info else "⏳ 需要更多信息"
        
        # 更新对话历史
        history_json = json.dumps(response.conversation_history, ensure_ascii=False, indent=2)
        
        # XML表单
        xml_display = response.xml_form if response.xml_form else "尚未生成XML表单"
        
        # 会话ID
        final_session_id = getattr(response, 'session_id', form_request.session_id)
        
        return (
            status,
            response.agent_response,
            history_json,
            xml_display,
            final_session_id or ""
        )
        
    except Exception as e:
        logger.error(f"Former Agent对话处理失败: {e}")
        return "❌ 错误", f"处理失败: {str(e)}", "", "", ""

def stream_former_conversation(
    user_query: str,
    session_id: str,
    conversation_history_json: str,
) -> Generator[Tuple[str, str, str, str, str], None, None]:
    """流式处理Former Agent对话"""
    
    init_agents()
    
    if not user_query.strip():
        yield "❌ 错误", "请输入查询内容", "", "", ""
        return
    
    try:
        # 显示处理状态
        yield "⏳ 处理中", "正在分析用户需求...", "", "", ""
        
        # 解析对话历史
        conversation_history = []
        if conversation_history_json.strip():
            try:
                conversation_history = json.loads(conversation_history_json)
            except json.JSONDecodeError:
                pass
        
        # 处理session_id，确保符合Pydantic要求
        processed_session_id = None
        if session_id and session_id.strip():
            processed_session_id = session_id.strip()
        
        # 创建FormRequest
        form_request = FormRequest(
            user_query=user_query,
            session_id=processed_session_id,
            conversation_history=conversation_history
        )
        
        yield "⏳ 处理中", "正在调用Former Agent...", "", "", ""
        
        # 异步调用Former Agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(former_agent.process_conversation(form_request))
        finally:
            loop.close()
        
        yield "⏳ 处理中", "正在格式化结果...", "", "", ""
        
        # 格式化返回结果
        status = "✅ 成功" if not response.need_more_info else "⏳ 需要更多信息"
        
        # 更新对话历史
        history_json = json.dumps(response.conversation_history, ensure_ascii=False, indent=2)
        
        # XML表单
        xml_display = response.xml_form if response.xml_form else "尚未生成XML表单"
        
        # 会话ID
        final_session_id = getattr(response, 'session_id', form_request.session_id)
        
        yield (
            status,
            response.agent_response,
            history_json,
            xml_display,
            final_session_id or ""
        )
        
    except Exception as e:
        logger.error(f"Former Agent流式处理失败: {e}")
        yield "❌ 错误", f"处理失败: {str(e)}", "", "", ""

# ------------------------------------------------------------------
# Event Engine 相关函数
# ------------------------------------------------------------------
def handle_event_workflow(
    workflow_type: str,
    user_query: str,
    session_id: str,
) -> Tuple[str, str, str]:
    """处理事件工作流"""
    
    init_agents()
    
    if not user_query.strip():
        return "❌ 错误", "请输入查询内容", ""
    
    try:
        # 创建ChatAgentRequest
        chat_request = ChatAgentRequest(
            language="zh",
            target=user_query,
            model="deepseek-v3",
            sessionKEY=session_id or str(uuid.uuid4())
        )
        
        # 异步执行工作流
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                event_manager.execute_workflow(workflow_type, {"request": chat_request})
            )
        finally:
            loop.close()
        
        # 格式化结果
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        
        # 事件历史
        events_history = result.get("events_history", [])
        history_text = "\\n".join([
            f"事件 {i+1}: {event.get('name', 'unknown')} - {event.get('status', 'unknown')}"
            for i, event in enumerate(events_history)
        ])
        
        return "✅ 工作流执行成功", result_json, history_text
        
    except Exception as e:
        logger.error(f"事件工作流处理失败: {e}")
        return "❌ 错误", f"工作流执行失败: {str(e)}", ""

def stream_event_workflow(
    workflow_type: str,
    user_query: str,
    session_id: str,
) -> Generator[Tuple[str, str, str], None, None]:
    """流式处理事件工作流"""
    
    init_agents()
    
    if not user_query.strip():
        yield "❌ 错误", "请输入查询内容", ""
        return
    
    try:
        yield "⏳ 启动", "正在启动事件工作流...", ""
        
        # 创建ChatAgentRequest
        chat_request = ChatAgentRequest(
            language="zh",
            target=user_query,
            model="deepseek-v3",
            sessionKEY=session_id or str(uuid.uuid4())
        )
        
        yield "⏳ 执行中", f"正在执行 {workflow_type} 工作流...", ""
        
        # 异步执行工作流
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                event_manager.execute_workflow(workflow_type, {"request": chat_request})
            )
        finally:
            loop.close()
        
        yield "⏳ 格式化", "正在格式化结果...", ""
        
        # 格式化结果
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        
        # 事件历史
        events_history = result.get("events_history", [])
        history_text = "\\n".join([
            f"事件 {i+1}: {event.get('name', 'unknown')} - {event.get('status', 'unknown')}"
            for i, event in enumerate(events_history)
        ])
        
        yield "✅ 完成", result_json, history_text
        
    except Exception as e:
        logger.error(f"事件工作流流式处理失败: {e}")
        yield "❌ 错误", f"工作流执行失败: {str(e)}", ""

def get_system_info() -> Tuple[str, str, str]:
    """获取系统信息"""
    init_agents()
    
    try:
        # 获取可用表单
        available_forms = former_agent.template_manager.get_available_forms()
        forms_json = json.dumps(available_forms, ensure_ascii=False, indent=2)
        
        # 获取可用工作流
        available_workflows = list(event_manager.workflow_templates.keys())
        workflows_json = json.dumps(available_workflows, ensure_ascii=False, indent=2)
        
        # 获取事件引擎状态
        engine_status = event_manager.get_statistics()
        status_json = json.dumps(engine_status, ensure_ascii=False, indent=2)
        
        return forms_json, workflows_json, status_json
        
    except Exception as e:
        error_msg = f"获取系统信息失败: {str(e)}"
        return error_msg, error_msg, error_msg

def execute_xml_form(xml_content: str, form_type: str) -> str:
    """执行XML表单"""
    init_agents()
    
    if not xml_content.strip():
        return "❌ 没有提供XML内容"
    
    try:
        # 使用事件管理器执行XML表单
        from dataflow.agent.eventengine.manager import handle_xml_form_execution
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                handle_xml_form_execution(xml_content, form_type)
            )
        finally:
            loop.close()
        
        return f"✅ XML表单执行成功:\\n{json.dumps(result, ensure_ascii=False, indent=2)}"
        
    except Exception as e:
        logger.error(f"XML表单执行失败: {e}")
        return f"❌ XML表单执行失败: {str(e)}"

# ------------------------------------------------------------------
# Gradio 界面构建
# ------------------------------------------------------------------
def create_gradio_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="Former Agent & Event Engine 测试") as demo:
        gr.Markdown("# 🤖 Former Agent & Event Engine 测试界面")
        
        with gr.Tab("💬 Former Agent 对话测试"):
            gr.Markdown("### 通过对话收集需求，生成XML表单")
            
            with gr.Row():
                with gr.Column(scale=1):
                    user_query = gr.Textbox(
                        label="用户查询",
                        placeholder="请输入您的需求，例如：我需要一个情感分析的算子",
                        lines=3
                    )
                    session_id_input = gr.Textbox(
                        label="会话ID（可选）",
                        placeholder="留空自动生成"
                    )
                    conversation_history = gr.Textbox(
                        label="对话历史（JSON格式，可选）",
                        placeholder="[]",
                        lines=5
                    )
                    
                    with gr.Row():
                        normal_btn = gr.Button("普通对话", variant="primary")
                        stream_btn = gr.Button("流式对话")
                        clear_btn = gr.Button("清空", variant="secondary")
                
                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="状态", interactive=False)
                    agent_response = gr.Textbox(
                        label="Former Agent响应", 
                        lines=6, 
                        interactive=False
                    )
                    session_id_output = gr.Textbox(label="会话ID", interactive=False)
            
            with gr.Row():
                updated_history = gr.Textbox(
                    label="更新后的对话历史",
                    lines=8,
                    interactive=False
                )
                xml_form_output = gr.Textbox(
                    label="生成的XML表单",
                    lines=8,
                    interactive=False
                )
            
            with gr.Row():
                execute_xml_btn = gr.Button("执行XML表单")
                xml_execution_result = gr.Textbox(
                    label="XML执行结果",
                    lines=4,
                    interactive=False
                )
        
        with gr.Tab("⚡ Event Engine 工作流测试"):
            gr.Markdown("### 测试事件驱动工作流")
            
            with gr.Row():
                with gr.Column(scale=1):
                    workflow_type = gr.Dropdown(
                        label="工作流类型",
                        choices=[
                            "form_conversation",
                            "requirement_to_xml", 
                            "create_operator",
                            "recommend_pipeline",
                            "optimize_code",
                            "fix_errors"
                        ],
                        value="form_conversation"
                    )
                    workflow_query = gr.Textbox(
                        label="工作流查询",
                        placeholder="请输入要处理的内容",
                        lines=3
                    )
                    workflow_session_id = gr.Textbox(
                        label="工作流会话ID（可选）",
                        placeholder="留空自动生成"
                    )
                    
                    with gr.Row():
                        workflow_normal_btn = gr.Button("执行工作流", variant="primary")
                        workflow_stream_btn = gr.Button("流式执行")
                
                with gr.Column(scale=1):
                    workflow_status = gr.Textbox(label="工作流状态", interactive=False)
                    workflow_result = gr.Textbox(
                        label="工作流结果",
                        lines=10,
                        interactive=False
                    )
                    event_history = gr.Textbox(
                        label="事件执行历史",
                        lines=6,
                        interactive=False
                    )
        
        with gr.Tab("📋 系统信息"):
            gr.Markdown("### 查看系统状态和可用资源")
            
            refresh_btn = gr.Button("刷新系统信息")
            
            with gr.Row():
                available_forms = gr.Textbox(
                    label="可用表单类型",
                    lines=8,
                    interactive=False
                )
                available_workflows = gr.Textbox(
                    label="可用工作流模板",
                    lines=8,
                    interactive=False
                )
                engine_status = gr.Textbox(
                    label="事件引擎状态",
                    lines=8,
                    interactive=False
                )
        
        # 绑定事件处理函数
        normal_btn.click(
            handle_former_conversation,
            inputs=[user_query, session_id_input, conversation_history],
            outputs=[status_output, agent_response, updated_history, xml_form_output, session_id_output]
        )
        
        stream_btn.click(
            stream_former_conversation,
            inputs=[user_query, session_id_input, conversation_history],
            outputs=[status_output, agent_response, updated_history, xml_form_output, session_id_output]
        )
        
        clear_btn.click(
            lambda: ("", "", "[]", "", "", ""),
            outputs=[user_query, session_id_input, conversation_history, 
                    status_output, agent_response, updated_history]
        )
        
        execute_xml_btn.click(
            lambda xml, _: execute_xml_form(xml, "create_operator"),
            inputs=[xml_form_output, gr.State("create_operator")],
            outputs=[xml_execution_result]
        )
        
        workflow_normal_btn.click(
            handle_event_workflow,
            inputs=[workflow_type, workflow_query, workflow_session_id],
            outputs=[workflow_status, workflow_result, event_history]
        )
        
        workflow_stream_btn.click(
            stream_event_workflow,
            inputs=[workflow_type, workflow_query, workflow_session_id],
            outputs=[workflow_status, workflow_result, event_history]
        )
        
        refresh_btn.click(
            get_system_info,
            outputs=[available_forms, available_workflows, engine_status]
        )
        
        # 页面加载时自动刷新系统信息
        demo.load(
            get_system_info,
            outputs=[available_forms, available_workflows, engine_status]
        )
    
    return demo

# ------------------------------------------------------------------
# FastAPI 后端 + Gradio 前端
# ------------------------------------------------------------------
def create_backend_app():
    """创建后端API应用"""
    app = FastAPI(title="Former Agent & Event Engine Backend")
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "message": "Former Agent & Event Engine Backend"}
    
    return app

# ------------------------------------------------------------------
# 主应用
# ------------------------------------------------------------------
# 创建应用实例
backend_app = create_backend_app()
demo = create_gradio_interface()

# FastAPI 组合
root = FastAPI()
root.mount("/api", backend_app)
gr.mount_gradio_app(root, demo, path="/ui")

@root.get("/", include_in_schema=False)
async def _to_ui():
    return RedirectResponse("/ui")

@root.get("/manifest.json", include_in_schema=False)
async def _manifest():
    return JSONResponse({
        "name": "Former-Agent-Event-Engine-Test", 
        "start_url": ".", 
        "display": "standalone"
    })

# ------------------------------------------------------------------
# 运行
# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(root, host="0.0.0.0", port=7863)
