#!/usr/bin/env python3
"""
DataFlow Master Agent 真实Web UI
真正调用Master Agent的决策功能，不是演示版本
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目路径到 Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import gradio as gr

try:
    # 导入真实的Master Agent
    from dataflow.agent_v2.master.agent import MasterAgent
    print("✅ 成功导入真实Master Agent")
    REAL_MASTER_AGENT = True
except ImportError as e:
    print(f"❌ 导入Master Agent失败: {e}")
    print("🔄 使用模拟版本进行演示")
    REAL_MASTER_AGENT = False


class RealMasterAgentWebUI:
    """真实的Master Agent Web UI控制器"""
    
    def __init__(self):
        """初始化Web UI"""
        if REAL_MASTER_AGENT:
            try:
                self.master_agent = MasterAgent()
                self.session_id = f"session_{int(time.time())}"
                self.chat_history = []
                print(f"✅ 真实Master Agent初始化成功，会话ID: {self.session_id}")
            except Exception as e:
                print(f"❌ Master Agent初始化失败: {e}")
                self.master_agent = None
                REAL_MASTER_AGENT = False
        
        if not REAL_MASTER_AGENT:
            # 使用简化的模拟版本
            from zyd.test_master_decision import SimpleMasterAgent
            self.master_agent = SimpleMasterAgent()
            self.session_id = f"mock_session_{int(time.time())}"
            print(f"🔄 使用模拟Master Agent，会话ID: {self.session_id}")
    
    async def chat_function(self, message: str, history: List[List[str]]) -> str:
        """Gradio ChatInterface 聊天处理函数"""
        if not self.master_agent:
            return "❌ Master Agent 未正确初始化，无法处理请求"
        
        try:
            print(f"\n🎯 [Master Agent] 收到用户消息: {message}")
            
            if REAL_MASTER_AGENT:
                # 使用真实的Master Agent
                result = await self.master_agent.execute(message)
            else:
                # 使用模拟的Master Agent
                result = await self.master_agent.execute_request(message)
            
            # 格式化响应
            response = self.format_response(result)
            
            print(f"✅ [Master Agent] 响应完成")
            return response
            
        except Exception as e:
            error_msg = f"❌ 处理请求时发生错误: {str(e)}"
            print(error_msg)
            return error_msg
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """格式化Master Agent响应"""
        try:
            if REAL_MASTER_AGENT:
                return self.format_real_response(result)
            else:
                return self.format_mock_response(result)
        except Exception as e:
            return f"❌ 格式化响应时发生错误: {str(e)}"
    
    def format_real_response(self, result: Dict[str, Any]) -> str:
        """格式化真实Master Agent响应"""
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
        
        return "\n\n".join(response_parts) if response_parts else str(result)
    
    def format_mock_response(self, result: Dict[str, Any]) -> str:
        """格式化模拟Master Agent响应"""
        response_parts = []
        
        if result.get('success'):
            response_parts.append(f"✅ **决策成功**: {result.get('message', '')}")
            response_parts.append(f"🔧 **选择的工具**: {result.get('tool_used', 'unknown')}")
            
            # 如果是APIKey工具的结果
            if result.get('tool_used') == "APIKey获取工具":
                api_result = result.get('result', {})
                if api_result.get('access_granted'):
                    response_parts.append(f"🔑 **获取的API密钥**: `{api_result.get('apikey', 'N/A')}`")
                    response_parts.append(f"📋 **SubAgent消息**: {api_result.get('message', 'N/A')}")
                else:
                    response_parts.append(f"❌ **访问被拒绝**: {api_result.get('message', 'N/A')}")
                    if api_result.get('hint'):
                        response_parts.append(f"💡 **提示**: {api_result.get('hint')}")
            else:
                response_parts.append(f"📋 **处理结果**: {result.get('result', 'N/A')}")
        else:
            response_parts.append(f"❌ **决策失败**: {result.get('message', '未知错误')}")
        
        return "\n\n".join(response_parts)


def create_real_master_agent_ui():
    """创建真实的Master Agent Web UI"""
    
    # 初始化UI控制器
    ui = RealMasterAgentWebUI()
    
    # 创建Gradio界面
    with gr.Blocks(
        title="DataFlow Real Master Agent",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        """
    ) as demo:
        
        # 标题和描述
        agent_type = "真实" if REAL_MASTER_AGENT else "模拟"
        gr.HTML(f"""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
            <h1>🤖 DataFlow {agent_type} Master Agent</h1>
            <h3>🎯 真正的决策功能测试</h3>
            <p>这里使用的是{agent_type}的Master Agent，会真正进行意图识别和工具路由</p>
            <p>APIKey SubAgent硬编码密钥: <code>DFlow2024Secret</code></p>
        </div>
        """)
        
        # 主界面
        with gr.Row():
            with gr.Column(scale=3):
                # 聊天界面
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
                # 系统状态面板
                with gr.Group():
                    gr.HTML("<h3>📊 系统状态</h3>")
                    
                    status_display = gr.HTML(f"""
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
                        <p><strong>🔄 运行模式</strong>: {agent_type}模式</p>
                        <p><strong>🕒 启动时间</strong>: {datetime.now().strftime('%H:%M:%S')}</p>
                        <p><strong>🆔 会话ID</strong>: {ui.session_id}</p>
                        <p><strong>🤖 Master Agent</strong>: {"✅ 真实" if REAL_MASTER_AGENT else "🔄 模拟"}</p>
                    </div>
                    """)
                
                # 测试指南
                with gr.Group():
                    gr.HTML("<h3>🧪 测试指南</h3>")
                    
                    test_guide = gr.HTML("""
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; font-size: 13px;'>
                        <p><strong>🔑 API密钥测试</strong>:</p>
                        <p>• "我需要API密钥"</p>
                        <p>• "给我秘密的apikey"</p>
                        <p>期望: 返回 <code>DFlow2024Secret</code></p>
                        <br>
                        <p><strong>❌ 失败测试</strong>:</p>
                        <p>• "今天天气怎么样？"</p>
                        <p>期望: 拒绝访问或无法识别</p>
                        <br>
                        <p><strong>🎯 其他工具测试</strong>:</p>
                        <p>• 表单: "创建表单"</p>
                        <p>• 分析: "分析数据"</p>
                        <p>• 代码: "生成代码"</p>
                    </div>
                    """)
                
                # 当前密钥显示
                with gr.Group():
                    gr.HTML("<h3>🔐 秘密信息</h3>")
                    
                    secret_display = gr.HTML("""
                    <div style='background: #fff3cd; padding: 15px; border-radius: 8px; border: 1px solid #ffeaa7;'>
                        <p><strong>🔑 硬编码密钥</strong>:</p>
                        <p><code style='background: #000; color: #0f0; padding: 5px; border-radius: 3px;'>DFlow2024Secret</code></p>
                        <p style='font-size: 12px; color: #856404;'>
                            只有通过正确的请求关键词才能获取到此密钥
                        </p>
                    </div>
                    """)
        
        # 底部说明
        gr.HTML(f"""
        <div style='text-align: center; padding: 20px; margin-top: 20px; background: #f8f9fa; border-radius: 10px;'>
            <h4>🎯 测试说明</h4>
            <p>当前运行的是<strong>{agent_type}</strong>Master Agent，会真正进行决策和工具路由</p>
            <p>• 输入包含"API密钥"相关词汇，应该路由到APIKey SubAgent</p>
            <p>• 成功获取密钥说明决策功能正常工作</p>
            <p>• 拒绝访问说明SubAgent的验证逻辑正常</p>
        </div>
        """)
    
    return demo


def main():
    """主函数"""
    print("🚀 启动DataFlow真实Master Agent Web UI...")
    
    try:
        # 创建界面
        demo = create_real_master_agent_ui()
        
        # 启动服务
        print("🌐 正在启动Web服务...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,  # 使用不同的端口避免冲突
            share=False,
            show_api=False,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
