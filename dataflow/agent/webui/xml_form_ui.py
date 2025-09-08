"""
XML表单Web UI
基于Gradio的XML表单对话界面
"""

import gradio as gr
import asyncio
import json
from typing import List, Tuple

from ..agentrole.former import FormerAgent
from ..xmlforms.models import FormRequest, FormResponse
from ..xmlforms.form_templates import FormTemplateManager
from ..xmlforms.worker_agent import WorkerAgent


class XMLFormWebUI:
    """XML表单Web界面"""
    
    def __init__(self):
        self.template_manager = FormTemplateManager()
        self.former_agent = FormerAgent()
        self.worker_agent = WorkerAgent(self.template_manager)
        
    def _process_conversation(self, message: str, history: List[List[str]], session_id: str) -> Tuple[List[List[str]], str]:
        """处理对话"""
        try:
            # 准备请求
            conversation_history = []
            for msg_pair in history:
                if len(msg_pair) >= 2:
                    conversation_history.extend([
                        {"role": "user", "content": msg_pair[0]},
                        {"role": "assistant", "content": msg_pair[1]}
                    ])
            
            request = FormRequest(
                user_query=message,
                conversation_history=conversation_history,
                session_id=session_id
            )
            
            # 调用异步方法
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self.former_agent.process_conversation(request))
            
            # 更新历史
            history.append([message, response.agent_response])
            
            # 构建状态信息
            status_info = f"表单类型: {response.form_type}\\n"
            if response.xml_form:
                status_info += "✅ XML表单已生成"
            else:
                status_info += "⏳ 继续收集信息中..."
            
            return history, status_info
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            history.append([message, error_msg])
            return history, f"❌ 错误: {str(e)}"
    
    def _execute_xml_form(self, history: List[List[str]], session_id: str) -> str:
        """执行XML表单"""
        try:
            # 从对话历史中提取最新的XML表单
            xml_form = None
            for msg_pair in reversed(history):
                if len(msg_pair) >= 2 and "```xml" in msg_pair[1]:
                    response_text = msg_pair[1]
                    xml_start = response_text.find("```xml") + 6
                    xml_end = response_text.find("```", xml_start)
                    if xml_end > xml_start:
                        xml_form = response_text[xml_start:xml_end].strip()
                        break
            
            if not xml_form:
                return "❌ 未找到有效的XML表单"
            
            # 执行XML表单
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.worker_agent.execute_xml_form(xml_form))
            
            # 处理结果
            if hasattr(result, 'success') and result.success:
                output_text = f"✅ 执行成功!\\n\\n{result.result}"
                if hasattr(result, 'generated_code') and result.generated_code:
                    output_text += f"\\n\\n生成的代码:\\n```python\\n{result.generated_code}\\n```"
                return output_text
            else:
                # ChatResponse对象的处理
                if hasattr(result, 'content'):
                    return f"✅ 执行完成!\\n\\n{result.content}"
                else:
                    return f"✅ 执行完成!\\n\\n{str(result)}"
                
        except Exception as e:
            return f"❌ 执行异常: {str(e)}"
    
    def _show_available_forms(self) -> str:
        """显示可用表单类型"""
        forms = self.template_manager.get_available_forms()
        output = "## 可用的表单类型:\\n\\n"
        
        for form_type, description in forms.items():
            output += f"### {form_type}\\n{description}\\n\\n"
        
        return output
    
    def _get_session_info(self, session_id: str) -> str:
        """获取会话信息"""
        if session_id in self.former_agent.session_states:
            state = self.former_agent.session_states[session_id]
            conv_count = len(self.former_agent.conversations.get(session_id, []))
            return f"会话ID: {session_id}\\n表单类型: {state.get('form_type', '未知')}\\n对话轮数: {conv_count // 2}"
        return f"会话ID: {session_id}\\n状态: 新会话"
    
    def create_interface(self) -> gr.Interface:
        """创建Gradio界面"""
        
        with gr.Blocks(title="DataFlow XML表单系统", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🔧 DataFlow XML表单系统")
            gr.Markdown("通过对话方式创建和执行XML配置表单")
            
            with gr.Tab("💬 对话生成"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="对话历史",
                            height=400,
                            show_label=True
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="输入您的需求",
                                placeholder="例如：我想创建一个文本预处理算子",
                                lines=2,
                                scale=4
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1)
                        
                        with gr.Row():
                            execute_btn = gr.Button("执行XML表单", variant="secondary")
                            clear_btn = gr.Button("清空对话", variant="stop")
                    
                    with gr.Column(scale=1):
                        session_id_input = gr.Textbox(
                            label="会话ID（可选）",
                            placeholder="留空自动生成",
                            value=""
                        )
                        
                        status_display = gr.Textbox(
                            label="状态信息",
                            interactive=False,
                            lines=3
                        )
                        
                        execution_output = gr.Textbox(
                            label="执行结果",
                            interactive=False,
                            lines=10
                        )
            
            with gr.Tab("📋 表单类型"):
                forms_info = gr.Markdown()
                
                # 显示可用表单
                interface.load(
                    fn=self._show_available_forms,
                    outputs=forms_info
                )
            
            with gr.Tab("ℹ️ 会话信息"):
                session_info_display = gr.Textbox(
                    label="当前会话信息",
                    interactive=False,
                    lines=5
                )
                
                refresh_session_btn = gr.Button("刷新会话信息")
                
                refresh_session_btn.click(
                    fn=self._get_session_info,
                    inputs=session_id_input,
                    outputs=session_info_display
                )
            
            # 事件绑定
            def handle_message(message, history, session_id):
                return self._process_conversation(message, history, session_id)
            
            # 发送消息
            send_btn.click(
                fn=handle_message,
                inputs=[msg_input, chatbot, session_id_input],
                outputs=[chatbot, status_display]
            ).then(
                lambda: "",
                outputs=msg_input
            )
            
            # 回车发送
            msg_input.submit(
                fn=handle_message,
                inputs=[msg_input, chatbot, session_id_input],
                outputs=[chatbot, status_display]
            ).then(
                lambda: "",
                outputs=msg_input
            )
            
            # 执行XML表单
            execute_btn.click(
                fn=self._execute_xml_form,
                inputs=[chatbot, session_id_input],
                outputs=execution_output
            )
            
            # 清空对话
            clear_btn.click(
                lambda: ([], "会话已清空"),
                outputs=[chatbot, status_display]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """启动Web界面"""
        interface = self.create_interface()
        return interface.launch(**kwargs)


def main():
    """主函数"""
    ui = XMLFormWebUI()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
