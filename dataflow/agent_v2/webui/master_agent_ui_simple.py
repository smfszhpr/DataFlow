#!/usr/bin/env python3
"""
DataFlow Master Agent Web UI - 简化版本
基于 Gradio ChatInterface 的智能代理聊天界面（兼容版本）
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目路径到 Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import gradio as gr

try:
    # 导入 Agent V2 架构
    from dataflow.agent_v2.base.core import SubAgent, BaseTool, GraphBuilder
    from dataflow.agent_v2.master.agent import MasterAgent
    print("✅ 成功导入 Agent V2 架构")
except ImportError as e:
    print(f"❌ 导入 Agent V2 架构失败: {e}")
    sys.exit(1)


class MasterAgentWebUI:
    """Master Agent Web UI 控制器"""
    
    def __init__(self):
        """初始化 Web UI"""
        try:
            self.master_agent = MasterAgent()
            self.session_id = f"session_{int(time.time())}"
            self.chat_history = []
            print(f"✅ Master Agent Web UI 初始化成功，会话ID: {self.session_id}")
        except Exception as e:
            print(f"❌ Master Agent Web UI 初始化失败: {e}")
            self.master_agent = None
    
    async def chat_function(self, message: str, history: List[List[str]]) -> str:
        """Gradio ChatInterface 聊天处理函数"""
        if not self.master_agent:
            return "❌ Master Agent 未正确初始化，无法处理请求"
        
        try:
            print(f"\n🎯 [Master Agent] 收到用户消息: {message}")
            
            # 执行 Master Agent
            result = await self.master_agent.execute(message)
            
            # 格式化响应
            response = self.format_response(result)
            
            print(f"✅ [Master Agent] 响应完成")
            return response
            
        except Exception as e:
            error_msg = f"❌ 处理请求时发生错误: {str(e)}"
            print(error_msg)
            return error_msg
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """格式化 Master Agent 响应"""
        try:
            if not result:
                return "❌ Master Agent 没有返回结果"
            
            # 基础响应格式
            response_parts = []
            
            # 添加状态信息
            if result.get('status'):
                status_icon = "✅" if result['status'] == 'completed' else "⚠️"
                response_parts.append(f"{status_icon} **状态**: {result['status']}")
            
            # 添加执行的工具信息
            if result.get('executed_tools'):
                tools_str = ", ".join(result['executed_tools'])
                response_parts.append(f"🔧 **使用的工具**: {tools_str}")
            
            # 添加主要结果
            if result.get('final_result'):
                response_parts.append(f"📋 **结果**:\n{result['final_result']}")
            
            # 添加详细信息
            if result.get('details'):
                response_parts.append(f"ℹ️ **详情**: {result['details']}")
            
            # 如果有错误信息
            if result.get('error'):
                response_parts.append(f"❌ **错误**: {result['error']}")
            
            # 如果结果为空，使用默认格式
            if not response_parts:
                response_parts.append(f"✅ **Master Agent 处理完成**\n\n结果: {str(result)}")
            
            return "\n\n".join(response_parts)
            
        except Exception as e:
            return f"❌ 格式化响应时发生错误: {str(e)}"


def create_master_agent_ui():
    """创建 Master Agent Web UI"""
    
    # 初始化 UI 控制器
    ui = MasterAgentWebUI()
    
    # 创建 Gradio 界面
    with gr.Blocks(
        title="DataFlow Master Agent",
        theme=gr.themes.Soft(),
        css="""
        .chat-container {
            border-radius: 10px !important;
        }
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        """
    ) as demo:
        
        # 标题和描述
        gr.HTML("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
            <h1>🤖 DataFlow Master Agent</h1>
            <h3>🏛️ 基于 MyScaleKB-Agent 的架构设计</h3>
            <p>本系统采用了与 MyScaleKB-Agent 相同的设计模式：Master Agent + SubAgent + LangGraph 状态机 + 流式处理</p>
            <p>实现了真正的事件驱动、工具选择和 SubAgent 路由机制</p>
        </div>
        """)
        
        # 主界面
        with gr.Row():
            with gr.Column(scale=3):
                # 使用简化的 ChatInterface 参数
                chat = gr.ChatInterface(
                    fn=ui.chat_function,
                    title="Master Agent 聊天",
                    description="输入您的需求，Master Agent 会自动选择合适的工具处理",
                    examples=[
                        "我想创建一个用户注册表单",
                        "分析这个数据集的销售趋势",
                        "生成一个数据处理流水线",
                        "创建一个包含验证功能的登录表单"
                    ]
                )
            
            with gr.Column(scale=1):
                # 系统状态面板
                with gr.Group():
                    gr.HTML("<h3>📊 系统状态</h3>")
                    
                    status_display = gr.HTML(f"""
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
                        <p><strong>🔄 运行状态</strong>: 正常运行</p>
                        <p><strong>🕒 启动时间</strong>: {datetime.now().strftime('%H:%M:%S')}</p>
                        <p><strong>🆔 会话ID</strong>: {ui.session_id}</p>
                        <p><strong>🤖 Master Agent</strong>: {"✅ 就绪" if ui.master_agent else "❌ 未就绪"}</p>
                    </div>
                    """)
                
                # 工具状态面板
                with gr.Group():
                    gr.HTML("<h3>🔧 可用工具</h3>")
                    
                    tools_display = gr.HTML("""
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
                        <p>🤖 <strong>Former Agent</strong> - 表单生成</p>
                        <p>🏗️ <strong>Pipeline Builder</strong> - 管道构建</p>
                        <p>📊 <strong>Data Analyzer</strong> - 数据分析</p>
                        <p>💻 <strong>Code Generator</strong> - 代码生成</p>
                    </div>
                    """)
                
                # 架构信息面板
                with gr.Group():
                    gr.HTML("<h3>🏛️ 架构信息</h3>")
                    
                    arch_display = gr.HTML("""
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; font-size: 12px;'>
                        <p><strong>🎯 设计模式</strong>: Master + SubAgent</p>
                        <p><strong>📊 状态管理</strong>: LangGraph</p>
                        <p><strong>🔧 工具选择</strong>: 智能路由</p>
                        <p><strong>💬 交互模式</strong>: 流式处理</p>
                        <p><strong>🔄 执行引擎</strong>: AsyncIO</p>
                    </div>
                    """)
        
        # 底部信息
        gr.HTML("""
        <div style='text-align: center; padding: 20px; margin-top: 20px; background: #f8f9fa; border-radius: 10px;'>
            <h4>💡 使用说明</h4>
            <p>• 直接描述您的需求，Master Agent 会自动选择合适的工具</p>
            <p>• 支持多轮对话，可以深入细化需求</p>
            <p>• 系统会显示执行过程和工具使用情况</p>
            <p>• 基于 MyScaleKB-Agent 架构，提供企业级的可靠性</p>
        </div>
        """)
    
    return demo


def main():
    """主函数"""
    print("🚀 启动 DataFlow Master Agent Web UI...")
    
    try:
        # 创建界面
        demo = create_master_agent_ui()
        
        # 启动服务
        print("🌐 正在启动 Web 服务...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_api=False,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
