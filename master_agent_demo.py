#!/usr/bin/env python3
"""
DataFlow Master Agent Web UI - 支持真实Master Agent
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import gradio as gr

# 导入真实的Master Agent
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    from dataflow.agent_v2.master.agent import create_master_agent
    print("✅ 成功导入真实Master Agent")
except ImportError as e:
    print(f"❌ 导入Master Agent失败: {e}")
    raise Exception(f"无法导入Master Agent: {e}")


class MasterAgentWebUI:
    """Master Agent Web UI 控制器"""
    
    def __init__(self):
        """初始化 Web UI - 只使用真实Master Agent"""
        try:
            self.master_agent, self.master_executor = create_master_agent()
            self.agent_type = "真实"
            print("✅ 使用真实Master Agent")
        except Exception as e:
            print(f"❌ 真实Master Agent初始化失败: {e}")
            raise Exception(f"Master Agent初始化失败: {e}")
        
        self.session_id = f"session_{int(time.time())}"
        self.chat_history = []
        print(f"✅ Master Agent Web UI 初始化成功，会话ID: {self.session_id}")
    
    async def chat_function(self, message: str, history: List[List[str]]) -> str:
        """Gradio ChatInterface 聊天处理函数"""
        try:
            print(f"\n🎯 [Master Agent] 收到用户消息: {message}")
            
            # 使用真实的Master Agent执行器
            result = await self.master_executor.execute(message, self.session_id)
            
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
            # 直接返回output内容，这是Master Agent已经格式化好的结果
            if result.get('success') and result.get('output'):
                return result['output']
            
            # 如果失败，返回错误信息
            elif not result.get('success'):
                error = result.get('output', '未知错误')
                return f"❌ {error}"
            
            # 兼容旧格式
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
                response_parts.append(result['final_result'])
            
            return "\n\n".join(response_parts) if response_parts else str(result)
            
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
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        """
    ) as demo:
        
        # 标题和描述
        gr.HTML(f"""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
            <h1>🤖 DataFlow {ui.agent_type} Master Agent</h1>
            <h3>🏛️ 基于 MyScaleKB-Agent 的架构设计</h3>
            <p>本系统采用了与 MyScaleKB-Agent 相同的设计模式：Master Agent + SubAgent + LangGraph 状态机</p>
            <p>实现了真正的事件驱动、工具选择和 SubAgent 路由机制</p>
            <p><strong>🔑 APIKey 测试密钥: 123121323132</strong></p>
        </div>
        """)
        
        # 主界面
        with gr.Row():
            with gr.Column(scale=3):
                # 使用最简单的 ChatInterface
                chat = gr.ChatInterface(
                    fn=ui.chat_function,
                    examples=[
                        "我需要今天的API密钥",      # 测试APIKey路由
                        "给我秘密的apikey",        # 测试APIKey路由  
                        "我想创建一个用户表单",     # 测试Former Agent路由
                        "分析这个数据集",          # 测试Data Analysis路由
                        "生成一段处理代码",        # 测试Code Generator路由
                        "今天天气怎么样？"         # 测试无法识别的请求
                    ]
                )
            
            with gr.Column(scale=1):
                
                # 工具状态面板
                with gr.Group():
                    gr.HTML("<h3>🔧 可用工具</h3>")
                    
                    tools_display = gr.HTML("""
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
                        <p>🤖 <strong>Former Agent</strong> - 表单生成</p>
                        <p>🏗️ <strong>Pipeline Builder</strong> - 管道构建</p>
                        <p>📊 <strong>Data Analyzer</strong> - 数据分析</p>
                        <p>💻 <strong>Code Generator</strong> - 代码生成</p>
                        <p>🔑 <strong>API Key Agent</strong> - 密钥获取</p>
                    </div>
                    """)
                
        # 底部信息
        gr.HTML(f"""
        <div style='text-align: center; padding: 20px; margin-top: 20px; background: #f8f9fa; border-radius: 10px;'>
            <h4>💡 使用说明</h4>
            <p>• 直接描述您的需求，Master Agent 会自动选择合适的工具</p>
            <p>• 支持多轮对话，可以深入细化需求</p>
            <p>• 系统会显示执行过程和工具使用情况</p>
            <p>• 基于 MyScaleKB-Agent 架构，提供企业级的可靠性</p>
            <br>
            <p style='font-size: 14px; color: #666;'>
                <strong>🚀 当前模式</strong>：{ui.agent_type}版本 - 真正的决策功能测试
            </p>
        </div>
        """)
    
    return demo


def main():
    """主函数"""
    print(f"🚀 启动 DataFlow Master Agent Web UI...")
    
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
