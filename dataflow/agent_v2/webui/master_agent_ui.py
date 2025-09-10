"""
DataFlow Master Agent Web UI
基于 Gradio ChatInterface 的智能代理聊天界面
"""
import gradio as gr
import asyncio
import sys
import os
from typing import List, Dict, Any
import uuid
from datetime import datetime

# 添加路径
sys.path.insert(0, '/Users/zyd/DataFlow')

try:
    from dataflow.agent_v2.master.agent import create_master_agent, MasterAgentExecutor
    from dataflow.agent_v2.subagents.pipeline_builder import create_pipeline_builder
    AGENT_V2_AVAILABLE = True
    print("✅ 成功导入 Agent V2 架构")
except ImportError as e:
    print(f"⚠️ 无法导入 Agent V2: {e}")
    AGENT_V2_AVAILABLE = False


class MasterAgentWebUI:
    """Master Agent Web UI 控制器"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_count = 0
        self.start_time = datetime.now()
        
        if AGENT_V2_AVAILABLE:
            try:
                self.master_agent, self.executor = create_master_agent()
                # 注册 SubAgent
                pipeline_builder = create_pipeline_builder()
                self.master_agent.sub_agents["pipeline_builder"] = pipeline_builder
                self.agent_type = "Agent V2 Architecture"
                self.status = "✅ 完全加载"
            except Exception as e:
                print(f"❌ Agent V2 初始化失败: {e}")
                self.agent_type = "Agent V2 (简化模式)"
                self.status = "⚠️ 简化加载"
                self.executor = self._create_mock_executor()
        else:
            self.agent_type = "Mock Agent"
            self.status = "⚠️ 演示模式"
            self.executor = self._create_mock_executor()
    
    def _create_mock_executor(self):
        """创建模拟执行器用于演示"""
        class MockExecutor:
            async def execute(self, user_input: str, session_id: str = None):
                await asyncio.sleep(1)  # 模拟处理时间
                return {
                    "success": True,
                    "output": f"🤖 Mock Response: 收到您的请求 '{user_input}'。\n\n"
                             f"在完整版本中，我会调用相应的工具来处理这个请求。\n\n"
                             f"📋 识别到的意图类型:\n"
                             f"• 表单生成: {'✅' if any(k in user_input.lower() for k in ['表单', '创建', '算子']) else '❌'}\n"
                             f"• 管道构建: {'✅' if any(k in user_input.lower() for k in ['管道', '流程', '构建']) else '❌'}\n"
                             f"• 代码生成: {'✅' if any(k in user_input.lower() for k in ['代码', '生成', '编程']) else '❌'}\n"
                             f"• 数据分析: {'✅' if any(k in user_input.lower() for k in ['分析', '数据', '洞察']) else '❌'}"
                }
        return MockExecutor()
    
    async def process_message(self, message: str, history: List[Dict[str, str]]) -> str:
        """处理用户消息"""
        try:
            result = await self.executor.execute(message, self.session_id)
            self.conversation_count += 1
            
            if result.get("success"):
                return result["output"]
            else:
                return f"❌ 处理失败: {result.get('output', '未知错误')}"
        
        except Exception as e:
            return f"❌ 系统错误: {str(e)}"
    
    def get_status_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        uptime = datetime.now() - self.start_time
        return {
            "agent_type": self.agent_type,
            "status": self.status,
            "session_id": self.session_id[:8] + "...",
            "conversations": self.conversation_count,
            "uptime": str(uptime).split('.')[0],
            "architecture": "myscalekb-agent inspired",
            "available": AGENT_V2_AVAILABLE
        }


# 全局实例
webui = MasterAgentWebUI()


def chat_function(message: str, history: List[Dict[str, str]]) -> str:
    """Gradio ChatInterface 聊天处理函数"""
    if not message.strip():
        return "请输入您的需求..."
    
    try:
        response = asyncio.run(webui.process_message(message, history))
        return response
    except Exception as e:
        return f"❌ 处理过程中出现错误: {str(e)}"


def get_system_status() -> str:
    """获取系统状态 HTML"""
    status = webui.get_status_info()
    
    status_color = "#28a745" if status['available'] else "#ffc107"
    status_icon = "✅" if status['available'] else "⚠️"
    
    return f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 10px; 
                margin: 10px 0;">
        <h2>🤖 DataFlow Master Agent</h2>
        <p><strong>架构:</strong> {status['agent_type']}</p>
        <p><strong>状态:</strong> <span style="color: {status_color};">{status_icon} {status['status']}</span></p>
        <p><strong>会话ID:</strong> {status['session_id']}</p>
        <p><strong>对话次数:</strong> {status['conversations']} 次</p>
        <p><strong>运行时间:</strong> {status['uptime']}</p>
        <p><strong>基于架构:</strong> {status['architecture']}</p>
    </div>
    
    <div style="background: #f8f9fa; 
                border: 1px solid #e9ecef; 
                padding: 15px; 
                border-radius: 8px; 
                margin: 10px 0;">
        <h3>🏗️ MyScaleKB-Agent 架构特性</h3>
        <ul>
            <li>🔧 <strong>Master Agent</strong> - LLM 驱动的工具选择</li>
            <li>🤖 <strong>SubAgent 系统</strong> - 可插拔的专业代理</li>
            <li>📊 <strong>状态机管理</strong> - LangGraph 工作流引擎</li>
            <li>🔄 <strong>流式处理</strong> - SSE 实时反馈</li>
        </ul>
        
        <h3>💡 支持的工具和 SubAgent</h3>
        <ul>
            <li>🤖 <strong>Former Agent</strong> - XML 表单生成</li>
            <li>🏗️ <strong>Pipeline Builder</strong> - 管道构建专家</li>
            <li>💻 <strong>Code Generator</strong> - 算子代码生成</li>
            <li>📊 <strong>Data Analyzer</strong> - 数据分析洞察</li>
        </ul>
        
        <h3>🎯 使用示例</h3>
        <ul>
            <li>"创建情感分析算子" → Former Agent 处理</li>
            <li>"构建数据处理管道" → Pipeline Builder SubAgent</li>
            <li>"生成预处理代码" → Code Generator 工具</li>
            <li>"分析数据特征" → Data Analyzer 工具</li>
        </ul>
    </div>
    """


def refresh_status():
    """刷新状态"""
    return get_system_status()


def reset_session():
    """重置会话"""
    global webui
    webui = MasterAgentWebUI()
    return [], get_system_status()


# 创建 Gradio 界面
with gr.Blocks(
    title="DataFlow Master Agent - MyScaleKB-Agent Architecture",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="gray"
    ),
    css="""
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
    }
    
    .chat-container {
        height: 650px;
    }
    
    .status-panel {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
    }
    
    .architecture-info {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    """
) as demo:
    
    # 页面标题
    gr.HTML(get_system_status(), elem_id="status-header")
    
    # 架构说明
    with gr.Row():
        gr.HTML("""
        <div class="architecture-info">
            <h3>🏛️ 基于 MyScaleKB-Agent 的架构设计</h3>
            <p>本系统采用了与 MyScaleKB-Agent 相同的设计模式：Master Agent + SubAgent + LangGraph 状态机 + 流式处理</p>
            <p>实现了真正的事件驱动、工具选择和 SubAgent 路由机制</p>
        </div>
        """)
    
    # 主界面
    with gr.Row():
        with gr.Column(scale=3):
            # ChatInterface
            chat = gr.ChatInterface(
                fn=chat_function,
                type="messages",
                chatbot=gr.Chatbot(
                    height=650,
                    type="messages",
                    placeholder="""
<div style='text-align: center; padding: 50px;'>
    <h2>🤖 DataFlow Master Agent</h2>
    <h3 style='color: #667eea;'>基于 MyScaleKB-Agent 架构</h3>
    <p style='color: #666; font-size: 16px;'>
        采用 Master + SubAgent 模式的智能数据处理助手
    </p>
    
    <div style='text-align: left; margin: 30px auto; max-width: 500px; background: #f8f9fa; padding: 20px; border-radius: 10px;'>
        <h4>🔧 架构组件:</h4>
        <p>🎯 <strong>Master Agent</strong> - 智能工具选择和任务路由</p>
        <p>🤖 <strong>Former Agent</strong> - 需求收集和表单生成</p>
        <p>🏗️ <strong>Pipeline Builder</strong> - 专业管道构建 SubAgent</p>
        <p>💻 <strong>Code Generator</strong> - 算子代码生成工具</p>
        <p>📊 <strong>Data Analyzer</strong> - 数据分析工具</p>
    </div>
    
    <div style='text-align: left; margin: 20px auto; max-width: 500px;'>
        <h4>💡 使用方式:</h4>
        <p>• 直接描述需求，Master Agent 会自动选择合适的工具</p>
        <p>• 支持多轮对话，深入细化需求</p>
        <p>• 可以查看执行过程和状态变化</p>
    </div>
    
    <p style='color: #888;'>请在下方输入您的需求开始体验...</p>
</div>
                    """,
                    show_copy_button=True,
                    elem_classes=["chat-container"]
                ),
                textbox=gr.Textbox(
                    placeholder="💭 描述您的需求，Master Agent 会自动选择合适的工具或 SubAgent 来处理...",
                    container=False,
                    scale=7,
                    lines=2,
                    max_lines=4
                ),
                submit_btn=gr.Button("🚀 提交给 Master Agent", variant="primary"),
                retry_btn=gr.Button("🔄 重新处理", variant="secondary"),
                undo_btn=gr.Button("↩️ 撤销上次", variant="secondary"),
                clear_btn=gr.Button("🗑️ 清空会话", variant="secondary"),
                examples=[
                    "我想创建一个情感分析算子，输入文本输出情感分类",
                    "构建一个包含数据清洗、转换、验证的完整管道",
                    "生成一个中文文本预处理算子的代码", 
                    "分析电商用户行为数据，提供业务洞察",
                    "我需要优化现有算子的性能，代码如下..."
                ],
                cache_examples=False
            )
        
        with gr.Column(scale=1, min_width=350):
            # 系统状态面板
            with gr.Group():
                gr.Markdown("### 🔧 系统状态")
                
                status_display = gr.HTML(
                    get_system_status(),
                    elem_classes=["status-panel"]
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新", size="sm")
                    reset_btn = gr.Button("🆕 新会话", size="sm", variant="secondary")
            
            # 架构说明
            with gr.Group():
                gr.Markdown("### 🏛️ 架构说明")
                gr.Markdown("""
                **Master Agent 工作流程:**
                1. **Bootstrap** - 分析用户意图
                2. **Tool Selection** - LLM 驱动选择工具
                3. **Execute** - 执行工具或转发 SubAgent
                4. **Summarize** - 汇总结果并响应
                
                **核心特性:**
                - 🎯 LLM 驱动的工具选择
                - 🔄 状态机工作流管理
                - 🤖 可插拔 SubAgent 架构
                - 📊 结构化状态传递
                """)
            
            # 快捷示例
            with gr.Group():
                gr.Markdown("### ⚡ 快捷示例")
                
                with gr.Column():
                    former_btn = gr.Button(
                        "🤖 Former Agent\n(表单生成)",
                        size="sm"
                    )
                    
                    pipeline_btn = gr.Button(
                        "🏗️ Pipeline Builder\n(管道构建)",
                        size="sm"
                    )
                    
                    code_btn = gr.Button(
                        "💻 Code Generator\n(代码生成)",
                        size="sm"
                    )
                    
                    analysis_btn = gr.Button(
                        "📊 Data Analyzer\n(数据分析)",
                        size="sm"
                    )
            
            # 技术细节
            with gr.Group():
                gr.Markdown("### 🔬 技术细节")
                gr.Markdown("""
                **基于技术栈:**
                - LangChain + LangGraph
                - OpenAI Tools Agent
                - Async/Await 模式
                - Pydantic 状态管理
                
                **参考架构:**
                - MyScaleKB-Agent
                - Multi-Agent System
                - Event-Driven Pattern
                """)
    
    # 事件处理
    refresh_btn.click(refresh_status, outputs=[status_display])
    reset_btn.click(reset_session, outputs=[chat.chatbot, status_display])
    
    # 示例按钮
    former_btn.click(
        lambda: "我想创建一个文本分类算子，能够将客服对话分为询问、投诉、建议等类别，请生成相应的XML配置",
        outputs=[chat.textbox]
    )
    
    pipeline_btn.click(
        lambda: "我需要构建一个完整的数据处理管道，包含数据加载、质量检查、清洗转换、特征提取、验证和导出等步骤",
        outputs=[chat.textbox]
    )
    
    code_btn.click(
        lambda: "生成一个中文文本预处理算子的完整代码，包括分词、去停用词、词性标注、命名实体识别等功能",
        outputs=[chat.textbox]
    )
    
    analysis_btn.click(
        lambda: "对电商平台的用户购买行为数据进行深度分析，包括用户画像、购买模式、季节性趋势和推荐策略",
        outputs=[chat.textbox]
    )


def main():
    """启动 Web UI"""
    print("🚀 启动 DataFlow Master Agent Web UI")
    print(f"🏛️ 架构类型: {webui.agent_type}")
    print(f"📊 状态: {webui.status}")
    print(f"🆔 会话ID: {webui.session_id}")
    print(f"⚡ Agent V2 可用: {AGENT_V2_AVAILABLE}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False,
        show_error=True,
        debug=True,
        show_api=False
    )


if __name__ == "__main__":
    main()
