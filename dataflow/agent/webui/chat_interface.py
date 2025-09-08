#!/usr/bin/env python3
"""
基于Gradio ChatInterface的Former Agent聊天界面
类似ChatGPT风格的对话界面，支持：
- 自动管理对话历史
- XML代码块显示
- 流式响应
- 美观的聊天界面
"""

import gradio as gr
import asyncio
from typing import List, Dict, Any
import uuid
from datetime import datetime
import sys
import os

# 添加项目根路径到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

try:
    from dataflow.agent.agentrole.former import FormerAgent
    from dataflow.agent.xmlforms.models import FormRequest
    from dataflow.logger import get_logger
except ImportError as e:
    print(f"导入错误: {e}")
    # 尝试相对导入
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from agentrole.former import FormerAgent
    from xmlforms.models import FormRequest
    # 简单的日志记录器
    import logging
    get_logger = lambda: logging.getLogger(__name__)

logger = get_logger()

class ChatInterface:
    """ChatGPT风格的聊天界面"""
    
    def __init__(self):
        self.former_agent = None
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        
    def initialize_agent(self):
        """初始化Former Agent"""
        if self.former_agent is None:
            try:
                self.former_agent = FormerAgent()
                logger.info(f"Former Agent初始化成功，会话ID: {self.session_id}")
                return True
            except Exception as e:
                logger.error(f"Former Agent初始化失败: {e}")
                return False
        return True
    
    def format_xml_response(self, response_text: str, xml_form: str = None) -> str:
        """格式化响应，如果有XML则添加代码块"""
        formatted_response = response_text
        
        if xml_form:
            formatted_response += "\n\n生成的XML表单配置：\n\n"
            formatted_response += f"```xml\n{xml_form}\n```"
            
        return formatted_response
    
    async def process_message(self, message: str, history, current_agent=None, current_event=None, event_history=None):
        """处理用户消息"""
        try:
            # 添加用户消息到历史记录（使用新的messages格式）
            history.append({"role": "user", "content": message})
            
            # 初始化状态
            agent_status = "🤖 DataFlow Agent"
            event_status = "开始处理..."
            history_status = event_history or ""
            
            # 首先尝试使用事件驱动系统
            try:
                # 处理用户请求
                from dataflow.agent.eventengine.agent_events import process_user_request, global_event_engine
                
                # 调试：检查事件注册情况
                events = global_event_engine.list_events()
                logger.info(f"当前已注册事件: {[e['name'] for e in events]}")
                
                # 更新状态：开始事件驱动处理
                agent_status = "🔄 Event-Driven System"
                event_status = "form_generation"
                history_status += "\n✅ 启动事件驱动系统"
                
                result_dict = await process_user_request(
                    user_query=message,
                    conversation_history=history
                )
                
                # 从字典中提取响应内容
                if isinstance(result_dict, dict):
                    result = result_dict.get('response', str(result_dict))
                    
                    # 检查是否有XML表单生成
                    if '<?xml' in result or '<form' in result:
                        agent_status = "📝 Former Agent"
                        event_status = "XML表单已生成"
                        history_status += "\n✅ Former Agent - XML表单生成完成"
                        
                        # 如果生成了XML表单，准备跳转到执行阶段
                        result += "\n\n🎯 **下一步**: 表单将被传递到代码执行阶段进行测试"
                    else:
                        agent_status = "💬 Former Agent"
                        event_status = "需求分析中"
                        history_status += "\n🔄 Former Agent - 与用户交流中"
                else:
                    result = str(result_dict)
                
                # 添加助手回复到历史记录
                history.append({"role": "assistant", "content": result})
                
            except Exception as e:
                logger.warning(f"事件驱动系统失败，使用备用方案: {e}")
                import traceback
                logger.debug(f"详细错误信息: {traceback.format_exc()}")
                
                # 备用方案：使用Former Agent
                from dataflow.agent.agentrole.former import FormerAgent, FormRequest
                
                agent_status = "🔧 Former Agent (备用)"
                event_status = "直接交流模式"
                history_status += "\n⚠️ 切换到Former Agent备用模式"
                
                former_agent = FormerAgent()
                request = FormRequest(
                    user_query=message,
                    conversation_history=[]
                )
                form_response = await former_agent.process_conversation(request)
                former_response = form_response.agent_response
                
                # 检查是否需要更多信息
                if form_response.need_more_info:
                    former_response += "\n\n💡 **提示**: 我需要更多信息来为您生成准确的表单配置"
                    event_status = "等待更多信息"
                    history_status += "\n❓ Former Agent - 需要更多信息"
                elif form_response.xml_form:
                    former_response += f"\n\n📋 **生成的XML表单**:\n```xml\n{form_response.xml_form}\n```"
                    event_status = "XML表单已生成"
                    history_status += "\n✅ Former Agent - XML表单生成完成"
                
                # 添加助手回复到历史记录
                history.append({"role": "assistant", "content": former_response})
                
            return "", history, agent_status, event_status, history_status
            
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            import traceback
            traceback.print_exc()
            # 添加错误消息到历史记录
            history.append({"role": "assistant", "content": f"抱歉，处理您的请求时出现错误：{str(e)}"})
            return "", history, "❌ 错误状态", "处理失败", "❌ 系统错误"
    
    def create_interface(self) -> gr.Blocks:
        """创建Gradio聊天界面"""
        
        # 自定义CSS样式，让界面更像ChatGPT
        custom_css = """
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        .header-info {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .status-info {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        /* 聊天气泡样式 */
        .message {
            margin: 10px 0;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 70%;
            word-wrap: break-word;
        }
        
        .user-message {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .bot-message {
            background: #f1f3f5;
            color: #333;
            margin-right: auto;
        }
        """
        
        with gr.Blocks(
            title="DataFlow Agent - 智能表单生成助手",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:
            
            # 页面标题和说明
            with gr.Row(elem_classes="chat-container"):
                with gr.Column():
                    gr.HTML("""
                    <div class="header-info">
                        <h1>🤖 DataFlow Agent</h1>
                        <p>智能表单生成助手 - 类似ChatGPT的对话体验</p>
                        <p>支持创建算子、优化算子、推荐Pipeline、构建知识库等多种表单</p>
                    </div>
                    """)
            
            # 状态信息显示
            with gr.Row(elem_classes="chat-container"):
                with gr.Column():
                    status_display = gr.HTML("""
                    <div class="status-info">
                        <h3>🔧 系统状态</h3>
                        <p><strong>会话ID:</strong> 正在初始化...</p>
                        <p><strong>Agent状态:</strong> 准备中</p>
                        <p><strong>配置状态:</strong> 检查中</p>
                        <p><strong>支持的表单类型:</strong> 创建算子、优化算子、推荐Pipeline、知识库构建</p>
                    </div>
                    """)
            
            # 主要聊天界面
            with gr.Row(elem_classes="chat-container"):
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        type="messages",
                        label="💬 对话历史",
                        height=600,
                        show_label=True,
                        placeholder="我是DataFlow Agent，专门帮助您生成各种XML表单配置。\\n\\n"
                                  "您可以说：\\n"
                                  "• '我想创建一个文本处理算子'\\n" 
                                  "• '帮我优化现有的数据清洗代码'\\n"
                                  "• '推荐一个数据治理pipeline'\\n"
                                  "• '我需要构建一个知识库'",
                        elem_classes=["chatbot"]
                    )
                
                with gr.Column(scale=1, min_width=200):
                    # 事件状态显示面板
                    gr.Markdown("### 🔄 系统状态")
                    current_agent = gr.Textbox(
                        label="当前Agent",
                        value="等待中...",
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                    current_event = gr.Textbox(
                        label="当前事件",
                        value="未开始",
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                    event_history = gr.Textbox(
                        label="事件历史",
                        value="",
                        lines=4,
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                    upcoming_events = gr.Textbox(
                        label="预期事件",
                        value="form_generation → code_analysis → code_execution → workflow_completion",
                        lines=2,
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                    
            msg = gr.Textbox(
                        label="💭 输入您的需求",
                        placeholder="例如：我想创建一个情感分析算子，输入是文本，输出是情感分类结果...",
                        lines=2,
                        max_lines=5,
                        show_label=True
                    )
                    
            with gr.Row():
                send_btn = gr.Button("📤 发送", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", scale=1)
            
            # 示例问题
            with gr.Row(elem_classes="chat-container"):
                with gr.Column():
                    gr.HTML("""
                    <h3>💡 示例问题</h3>
                    <p>点击下面的示例快速开始：</p>
                    """)
                    
                    with gr.Row():
                        example1 = gr.Button("📝 创建文本分类算子", size="sm")
                        example2 = gr.Button("🔧 优化数据处理代码", size="sm") 
                        example3 = gr.Button("🚀 推荐数据治理Pipeline", size="sm")
                        example4 = gr.Button("📚 构建知识库处理流程", size="sm")
            
            # 初始化状态更新
            def update_status():
                chat_interface = ChatInterface()
                success = chat_interface.initialize_agent()
                
                if success:
                    status_html = f"""
                    <div class="status-info">
                        <h3>✅ 系统状态</h3>
                        <p><strong>会话ID:</strong> {chat_interface.session_id}</p>
                        <p><strong>Agent状态:</strong> 运行正常</p>
                        <p><strong>配置状态:</strong> 已加载</p>
                        <p><strong>支持的表单类型:</strong> 创建算子、优化算子、推荐Pipeline、知识库构建</p>
                    </div>
                    """
                else:
                    status_html = """
                    <div class="status-info">
                        <h3>❌ 系统状态</h3>
                        <p><strong>Agent状态:</strong> 初始化失败</p>
                        <p><strong>配置状态:</strong> 检查配置文件</p>
                        <p>请检查EventEngine配置是否正确</p>
                    </div>
                    """
                return status_html
            
            # 创建全局聊天接口实例
            chat_interface = ChatInterface()
            
            # 处理消息的异步包装函数
            def handle_message(message, history, current_agent, current_event, event_history):
                if not message.strip():
                    return history, "", current_agent, current_event, event_history
                
                try:
                    # 运行异步函数
                    _, updated_history, agent_status, event_status, history_status = asyncio.run(
                        chat_interface.process_message(message, history, current_agent, current_event, event_history)
                    )
                    return updated_history, "", agent_status, event_status, history_status
                except Exception as e:
                    logger.error(f"处理消息失败: {e}")
                    error_response = f"❌ 处理失败: {str(e)}"
                    # 使用messages格式
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": error_response})
                    return history, "", "❌ 错误状态", "处理失败", "❌ 系统错误"
            
            # 清空对话
            def clear_conversation():
                chat_interface.session_id = str(uuid.uuid4())
                return [], "等待中...", "未开始", "", "form_generation → code_analysis → code_execution → workflow_completion"
            
            # 示例问题处理
            def set_example_text(example_text):
                return example_text
            
            # 事件绑定
            send_btn.click(
                handle_message,
                inputs=[msg, chatbot, current_agent, current_event, event_history],
                outputs=[chatbot, msg, current_agent, current_event, event_history]
            )
            
            msg.submit(
                handle_message,
                inputs=[msg, chatbot, current_agent, current_event, event_history], 
                outputs=[chatbot, msg, current_agent, current_event, event_history]
            )
            
            clear_btn.click(
                clear_conversation,
                outputs=[chatbot, current_agent, current_event, event_history, upcoming_events]
            )
            
            # 示例按钮事件
            example1.click(
                set_example_text,
                inputs=[gr.State("我想创建一个文本分类算子，输入是文本内容，输出是分类标签")],
                outputs=[msg]
            )
            
            example2.click(
                set_example_text,
                inputs=[gr.State("我有一段数据清洗的代码需要优化性能，代码处理的是用户评论数据")],
                outputs=[msg]
            )
            
            example3.click(
                set_example_text,
                inputs=[gr.State("我需要一个数据治理pipeline来处理电商用户行为数据，目标是提升数据质量")],
                outputs=[msg]
            )
            
            example4.click(
                set_example_text,
                inputs=[gr.State("我要构建一个知识库，处理PDF文档和网页内容，需要清洗和向量化")],
                outputs=[msg]
            )
            
            # 页面加载时更新状态
            interface.load(update_status, outputs=[status_display])
        
        return interface

def create_chat_interface() -> gr.Blocks:
    """创建聊天界面的工厂函数"""
    chat = ChatInterface()
    return chat.create_interface()

def main():
    """启动聊天界面"""
    print("🚀 启动DataFlow Agent聊天界面...")
    
    interface = create_chat_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7863,  # 使用不同端口避免冲突
        share=False,
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    main()
