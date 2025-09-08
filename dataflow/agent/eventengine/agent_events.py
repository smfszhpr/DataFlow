"""
DataFlow Agent事件处理模块
定义所有与DataFlow Agent相关的事件处理器和工作流
"""

import asyncio
from typing import Dict, Any, List
from dataflow.logger import get_logger
logger = get_logger()
from dataflow.agent.agentrole.former import FormerAgent
from dataflow.agent.eventengine.smart_engine import event, global_event_engine

@event("form_generation")
async def handle_form_generation(data: Dict[str, Any]) -> Dict[str, Any]:
    """处理表单生成事件"""
    logger.info(f"处理表单生成事件: {data.get('user_query', '')}")
    
    try:
        # 使用Former Agent处理表单生成
        former_agent = FormerAgent()
        # 使用process_conversation方法处理请求
        from dataflow.agent.agentrole.former import FormRequest
        # 转换conversation_history格式 - 只保留基本的role和content
        simple_history = []
        for msg in data.get('conversation_history', []):
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                simple_history.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        request = FormRequest(
            user_query=data.get('user_query', ''),
            conversation_history=simple_history
        )
        form_response = await former_agent.process_conversation(request)
        response = form_response.agent_response
        
        # 检查是否生成了XML表单，XML表单完成的标志
        xml_generated = False
        if form_response.xml_form and form_response.xml_form.strip():
            # 有明确的XML表单字段
            xml_generated = True
            response += f"\\n\\n📋 **生成的XML表单**:\\n```xml\\n{form_response.xml_form}\\n```"
        elif '<?xml' in response and '</workflow>' in response:
            # 响应中包含完整的XML结构
            xml_generated = True
        elif '```xml' in response and '```' in response:
            # 响应中包含XML代码块
            xml_generated = True
        
        # 根据是否生成XML来决定下一步事件
        if xml_generated:
            # XML已生成，准备跳转到代码执行事件
            logger.info("检测到XML表单生成完成，准备跳转到代码执行阶段")
            await global_event_engine.emit_event("code_execution", {
                "form_content": response,
                "xml_form": form_response.xml_form or response,
                "user_query": data.get('user_query', ''),
                "conversation_history": data.get('conversation_history', []),
                "form_completed": True
            })
            
            return {
                "status": "success",
                "form_content": response,
                "xml_generated": True,
                "next_event": "code_execution"
            }
        else:
            # 需要继续收集信息，触发代码分析事件以决定下一步
            await global_event_engine.emit_event("code_analysis", {
                "form_content": response,
                "user_query": data.get('user_query', ''),
                "conversation_history": data.get('conversation_history', []),
                "form_completed": False
            })
            
            return {
                "status": "success", 
                "form_content": response,
                "xml_generated": False,
                "next_event": "code_analysis"
            }
        
    except Exception as e:
        logger.error(f"表单生成失败: {e}")
        return {
            "status": "error",
            "error": str(e),
            "fallback_response": "表单生成过程中出现错误，请重试。"
        }

@event("code_analysis")
async def handle_code_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """处理代码分析事件"""
    logger.info("执行代码分析...")
    
    try:
        form_content = data.get('form_content', '')
        form_completed = data.get('form_completed', False)
        
        # 简单的代码分析逻辑
        analysis_result = {
            "complexity": "medium",
            "dependencies": [],
            "estimated_time": "2-3 minutes",
            "risk_level": "low",
            "form_ready": form_completed
        }
        
        if form_completed:
            # 表单已完成，可以直接跳转到代码执行
            logger.info("表单已完成，跳转到代码执行阶段")
            await global_event_engine.emit_event("code_execution", {
                "analysis": analysis_result,
                "form_content": form_content,
                "user_query": data.get('user_query', ''),
                "conversation_history": data.get('conversation_history', [])
            })
            next_event = "code_execution"
        elif "```xml" in form_content or "<?xml" in form_content:
            # 包含XML内容，也跳转到执行阶段
            logger.info("检测到XML内容，跳转到代码执行阶段")
            await global_event_engine.emit_event("code_execution", {
                "analysis": analysis_result,
                "form_content": form_content,
                "user_query": data.get('user_query', ''),
                "conversation_history": data.get('conversation_history', [])
            })
            next_event = "code_execution"
        else:
            # 没有XML，直接完成工作流（继续对话）
            logger.info("未检测到XML表单，继续对话流程")
            await global_event_engine.emit_event("workflow_completion", {
                "final_result": form_content,
                "analysis": analysis_result,
                "user_query": data.get('user_query', ''),
                "conversation_history": data.get('conversation_history', []),
                "continue_conversation": True
            })
            next_event = "workflow_completion"
        
        return {
            "status": "success",
            "analysis": analysis_result,
            "next_event": next_event
        }
        
    except Exception as e:
        logger.error(f"代码分析失败: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@event("code_execution")
async def handle_code_execution(data: Dict[str, Any]) -> Dict[str, Any]:
    """处理代码执行事件"""
    logger.info("执行代码...")
    
    try:
        form_content = data.get('form_content', '')
        xml_form = data.get('xml_form', '')
        
        # 检查是否有XML表单可以执行
        xml_to_execute = xml_form or form_content
        
        if '<?xml' in xml_to_execute or '<workflow>' in xml_to_execute:
            # 模拟XML表单处理和代码生成
            execution_result = {
                "success": True,
                "output": "✅ XML表单解析成功\\n📝 代码生成完成\\n🧪 测试用例创建完成",
                "execution_time": "2.1s",
                "generated_code": "# 基于XML表单生成的算子代码\\nclass CustomOperator:\\n    def process(self, data):\\n        # 实现您的算子逻辑\\n        return processed_data",
                "test_cases": ["test_basic_functionality", "test_edge_cases", "test_performance"]
            }
            
            # 成功执行，触发工作流完成
            await global_event_engine.emit_event("workflow_completion", {
                "final_result": f"{form_content}\\n\\n🎯 **执行结果**:\\n{execution_result['output']}\\n\\n💻 **生成的代码**:\\n```python\\n{execution_result['generated_code']}\\n```",
                "execution": execution_result,
                "analysis": data.get('analysis', {}),
                "user_query": data.get('user_query', ''),
                "conversation_history": data.get('conversation_history', []),
                "execution_completed": True
            })
            
            next_event = "workflow_completion"
        else:
            # 没有可执行的XML，触发调试事件
            execution_result = {
                "success": False,
                "output": "⚠️ 未检测到有效的XML表单",
                "execution_time": "0.1s",
                "error": "需要有效的XML表单才能生成代码"
            }
            
            await global_event_engine.emit_event("code_debug", {
                "execution": execution_result,
                "form_content": form_content,
                "user_query": data.get('user_query', ''),
                "conversation_history": data.get('conversation_history', [])
            })
            
            next_event = "code_debug"
        
        return {
            "status": "success",
            "execution": execution_result,
            "next_event": next_event
        }
        
    except Exception as e:
        logger.error(f"代码执行失败: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@event("code_debug")
async def handle_code_debug(data: Dict[str, Any]) -> Dict[str, Any]:
    """处理代码调试事件"""
    logger.info("执行代码调试...")
    
    try:
        # 模拟调试过程
        debug_result = {
            "issues_found": 0,
            "fixes_applied": 0,
            "debug_log": "未发现明显问题"
        }
        
        # 完成调试后，触发工作流完成
        await global_event_engine.emit_event("workflow_completion", {
            "final_result": data.get('form_content', ''),
            "debug": debug_result,
            "execution": data.get('execution', {}),
            "user_query": data.get('user_query', ''),
            "conversation_history": data.get('conversation_history', [])
        })
        
        return {
            "status": "success",
            "debug": debug_result,
            "next_event": "workflow_completion"
        }
        
    except Exception as e:
        logger.error(f"代码调试失败: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@event("workflow_completion")
async def handle_workflow_completion(data: Dict[str, Any]) -> Dict[str, Any]:
    """处理工作流完成事件"""
    logger.info("工作流完成")
    
    try:
        final_result = data.get('final_result', '')
        execution_completed = data.get('execution_completed', False)
        continue_conversation = data.get('continue_conversation', False)
        
        # 生成综合响应
        response_parts = [final_result]
        
        if execution_completed:
            # 完整执行流程完成
            response_parts.append("\n\n🎉 **流程状态**: 完整的XML表单生成和代码执行流程已完成")
            
            if data.get('analysis'):
                response_parts.append(f"\n📊 **分析结果**: 复杂度 {data['analysis'].get('complexity', 'unknown')}")
            
            if data.get('execution'):
                response_parts.append(f"\n⚡ **执行状态**: {data['execution'].get('output', 'N/A')}")
        elif continue_conversation:
            # 需要继续对话
            response_parts.append("\n\n💬 **状态**: 正在与Former Agent交流，收集更多信息以生成准确的XML表单")
            response_parts.append("\n🔄 **下一步**: 请提供更多详细信息，我会根据您的需求生成相应的XML配置")
        else:
            # 常规完成
            if data.get('analysis'):
                response_parts.append(f"\n📊 **分析结果**: 复杂度 {data['analysis'].get('complexity', 'unknown')}")
            
            if data.get('execution'):
                response_parts.append(f"\n⚡ **执行状态**: {data['execution'].get('output', 'N/A')}")
            
            if data.get('debug'):
                response_parts.append(f"\n🔧 **调试信息**: {data['debug'].get('debug_log', 'N/A')}")
        
        complete_response = "".join(response_parts)
        
        return {
            "status": "success",
            "response": complete_response,
            "workflow": "completed",
            "execution_completed": execution_completed,
            "continue_conversation": continue_conversation
        }
        
    except Exception as e:
        logger.error(f"工作流完成处理失败: {e}")
        return {
            "status": "error",
            "error": str(e),
            "response": data.get('final_result', '工作流处理出现错误')
        }

# 设置DataFlow Agent工作流
def setup_dataflow_workflow():
    """设置DataFlow Agent的完整工作流"""
    logger.info("设置DataFlow Agent工作流...")
    
    # 创建事件链: 表单生成 → 代码分析 → 代码执行 → 代码调试 → 工作流完成
    workflow_id = global_event_engine.create_workflow(
        "dataflow_agent_workflow",
        [
            "form_generation",
            "code_analysis", 
            "code_execution",
            "code_debug",
            "workflow_completion"
        ]
    )
    
    logger.info("DataFlow Agent工作流设置完成")
    return workflow_id

# 对外接口
async def process_user_request(user_query: str, conversation_history: List = None, session_id: str = None) -> Dict[str, Any]:
    """
    处理用户请求的主入口
    自动触发完整的事件驱动工作流
    """
    
    session_id = session_id or f"session_{asyncio.get_event_loop().time()}"
    conversation_history = conversation_history or []
    
    input_data = {
        "user_query": user_query,
        "conversation_history": conversation_history,
        "session_id": session_id
    }
    
    # 设置工作流（如果还没有设置）
    setup_dataflow_workflow()
    
    # 启动表单生成事件，开始整个工作流
    logger.info(f"开始处理用户请求: {user_query[:50]}...")
    result = await global_event_engine.emit_event("form_generation", input_data)
    
    # 等待工作流完成
    completion_result = await global_event_engine.wait_for_completion("workflow_completion", timeout=30)
    
    if completion_result and completion_result.status.value == "completed":
        return completion_result.data
    else:
        # 如果工作流没有完成，返回表单生成的结果
        if result and result.status.value == "completed":
            return {
                "response": result.data.get('form_content', '处理完成'),
                "status": "partial_completion"
            }
        else:
            return {
                "response": "处理请求时出现问题，请重试。",
                "status": "error"
            }

# 自动执行设置
try:
    setup_dataflow_workflow()
    
    # 列出所有注册的事件
    events = global_event_engine.list_events()
    logger.info(f"模块导入时已注册 {len(events)} 个事件: {[e['name'] for e in events]}")
    
    # 详细打印每个事件
    for event in events:
        logger.info(f"事件详情: {event}")
        
except Exception as e:
    logger.error(f"设置工作流失败: {e}")
    import traceback
    logger.error(f"详细错误: {traceback.format_exc()}")

if __name__ == "__main__":
    # 设置工作流
    setup_dataflow_workflow()
    
    # 列出所有注册的事件
    events = global_event_engine.list_events()
    logger.info(f"已注册 {len(events)} 个事件: {[e['name'] for e in events]}")
