"""
Former Agent V2 集成示例
展示如何将新的 SubAgent 模式的 Former Agent 集成到现有系统中
"""
import gradio as gr
import json
from datetime import datetime

from dataflow.agent_v2.former.mcp_adapter import FormerAgentMCPV2


class FormerAgentV2Integration:
    """Former Agent V2 集成类"""
    
    def __init__(self):
        self.mcp_agent = FormerAgentMCPV2()
        self.current_session = None
        
    def create_gradio_interface(self):
        """创建 Gradio 界面"""
        
        def start_new_session():
            """开始新会话"""
            self.current_session = self.mcp_agent.create_session()
            return f"✓ 新会话已创建: {self.current_session}", ""
        
        def analyze_requirement(user_input):
            """分析用户需求"""
            if not self.current_session:
                self.current_session = self.mcp_agent.create_session()
            
            result = self.mcp_agent.analyze_requirement(
                session_id=self.current_session,
                user_requirement=user_input,
                context=""
            )
            
            if result['status'] == 'success':
                data = result['data']
                response = f"""
🎯 **需求分析完成**

**表单类型**: {data['form_type']}  
**置信度**: {data['confidence']*100:.1f}%  
**推理过程**: {data['reasoning']}  
**建议模板**: {data['suggested_template']}  

请继续下一步字段检查...
"""
                return response, "analyze_complete"
            else:
                return f"❌ 分析失败: {result['message']}", "error"
        
        def check_fields():
            """检查字段完整性"""
            if not self.current_session:
                return "❌ 请先进行需求分析", "error"
            
            result = self.mcp_agent.check_fields(session_id=self.current_session)
            
            if result['status'] == 'success':
                data = result['data']
                
                if data['is_complete']:
                    response = f"""
✅ **字段检查完成**

**表单类型**: {data['form_type']}  
**完整性**: 完整 ✓  
**验证字段**: {len(data['validated_fields'])} 个  

字段详情:
"""
                    for field, value in data['validated_fields'].items():
                        response += f"• **{field}**: {value}\n"
                    
                    response += "\n可以进行XML生成..."
                    return response, "fields_complete"
                else:
                    response = f"""
⚠️ **字段检查结果**

**表单类型**: {data['form_type']}  
**完整性**: 不完整  
**缺失字段**: {', '.join(data['missing_fields'])}  

建议补充信息:
"""
                    for suggestion in data['suggestions']:
                        response += f"• {suggestion}\n"
                    
                    return response, "fields_incomplete"
            else:
                return f"❌ 字段检查失败: {result['message']}", "error"
        
        def generate_xml():
            """生成XML表单"""
            if not self.current_session:
                return "❌ 请先进行需求分析", "error"
            
            result = self.mcp_agent.generate_xml(session_id=self.current_session)
            
            if result['status'] == 'success':
                data = result['data']
                response = f"""
🎉 **XML生成成功**

**表单类型**: {data['form_type']}  
**字段数量**: {data['field_count']}  
**使用模板**: {data['template_used']}  
**置信度**: {data['confidence']*100:.1f}%  

**生成的XML**:
```xml
{data['xml_content']}
```

生成时间: {data['generation_timestamp']}
"""
                return response, "xml_complete"
            else:
                return f"❌ XML生成失败: {result['message']}", "error"
        
        def get_session_status():
            """获取当前会话状态"""
            if not self.current_session:
                return "❌ 当前无活动会话"
            
            result = self.mcp_agent.get_current_form(session_id=self.current_session)
            
            if result['status'] == 'success':
                data = result['data']
                status = f"""
📊 **会话状态**

**会话ID**: {data['session_id']}  
**创建时间**: {data['created_at']}  
**历史记录**: {data['history_count']} 条  
**最后活动**: {data['last_activity']}  

**分析结果**: {'✓' if data.get('analysis_result') else '✗'}  
**字段检查**: {'✓' if data.get('field_check_result') else '✗'}  
**XML生成**: {'✓' if data.get('xml_result') else '✗'}  
"""
                return status
            else:
                return f"❌ 获取状态失败: {result['message']}"
        
        # 创建界面
        with gr.Blocks(title="Former Agent V2 集成示例", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 Former Agent V2 - SubAgent 模式演示")
            gr.Markdown("基于 myscalekb-agent 架构重构的表单生成代理")
            
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("## 📝 用户输入")
                    
                    user_input = gr.Textbox(
                        label="描述您的需求",
                        placeholder="例如：我需要分析销售数据，生成月度报表...",
                        lines=3
                    )
                    
                    with gr.Row():
                        new_session_btn = gr.Button("🆕 新建会话", variant="secondary")
                        analyze_btn = gr.Button("🎯 分析需求", variant="primary")
                        check_btn = gr.Button("✅ 检查字段", variant="secondary")
                        generate_btn = gr.Button("⚡ 生成XML", variant="primary")
                    
                    status_btn = gr.Button("📊 查看状态", variant="secondary")
                
                with gr.Column(scale=2):
                    gr.Markdown("## 📊 处理状态")
                    
                    status_display = gr.Textbox(
                        label="当前状态",
                        lines=3,
                        interactive=False
                    )
                    
                    step_indicator = gr.Textbox(
                        label="进度指示",
                        value="等待开始...",
                        interactive=False
                    )
            
            with gr.Row():
                result_display = gr.Markdown("### 🔄 等待处理...")
            
            # 隐藏状态变量
            session_state = gr.State("")
            process_state = gr.State("")
            
            # 事件绑定
            new_session_btn.click(
                fn=start_new_session,
                outputs=[status_display, process_state]
            )
            
            analyze_btn.click(
                fn=analyze_requirement,
                inputs=[user_input],
                outputs=[result_display, process_state]
            )
            
            check_btn.click(
                fn=check_fields,
                outputs=[result_display, process_state]
            )
            
            generate_btn.click(
                fn=generate_xml,
                outputs=[result_display, process_state]
            )
            
            status_btn.click(
                fn=get_session_status,
                outputs=[status_display]
            )
            
            # 状态更新
            def update_step_indicator(state):
                steps = {
                    "": "🔘 等待开始",
                    "analyze_complete": "🎯 需求分析完成 → 🔘 字段检查",
                    "fields_complete": "✅ 字段检查完成 → 🔘 XML生成",
                    "fields_incomplete": "⚠️ 字段不完整 → 🔘 补充信息",
                    "xml_complete": "🎉 XML生成完成 ✓",
                    "error": "❌ 处理出错"
                }
                return steps.get(state, "🔄 处理中...")
            
            process_state.change(
                fn=update_step_indicator,
                inputs=[process_state],
                outputs=[step_indicator]
            )
        
        return interface
    
    def launch_demo(self):
        """启动演示"""
        interface = self.create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7862,
            share=False,
            debug=True
        )


if __name__ == "__main__":
    print("=== Former Agent V2 集成演示 ===")
    print("启动 Gradio 界面...")
    
    integration = FormerAgentV2Integration()
    integration.launch_demo()
