"""
Agent适配器模块
将现有的DataFlow Agent封装为事件驱动的处理器
"""

import asyncio
from typing import Dict, Any

from . import AgentInterface, Event, EventStatus
from ..agentrole.analyst import AnalystAgent
from ..agentrole.executioner import ExecutionAgent
from ..agentrole.debugger import DebugAgent
from ..agentrole.former import FormerAgent
from ..xmlforms.models import FormRequest, FormResponse
from ..xmlforms.form_templates import FormTemplateManager
from ..servicemanager import AnalysisService, Memory
from ..promptstemplates.prompt_template import PromptsTemplateGenerator
from ..toolkits import ChatAgentRequest
from dataflow import get_logger

logger = get_logger()

class AnalysisEventAgent(AgentInterface):
    """分析事件处理Agent"""
    
    def __init__(self):
        super().__init__("analysis")
        self.supported_events = ["analysis", "data_classification", "conversation_router"]
        self.memory = Memory()
        
    async def handle(self, event: Event) -> Dict[str, Any]:
        """处理分析类事件"""
        event_name = event.name
        payload = event.payload
        
        logger.info(f"🔍 开始处理分析事件: {event_name}")
        
        try:
            if event_name == "analysis":
                return await self._handle_analysis(payload)
            elif event_name == "data_classification":
                return await self._handle_data_classification(payload)
            elif event_name == "conversation_router":
                return await self._handle_conversation_router(payload)
            else:
                raise ValueError(f"不支持的分析事件类型: {event_name}")
                
        except Exception as e:
            logger.error(f"分析事件处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理通用分析任务"""
        try:
            # 从payload中提取请求信息
            request = payload.get("request")
            if not request:
                raise ValueError("缺少请求信息")
            
            # 创建分析服务
            prompts = PromptsTemplateGenerator(request.language)
            analyst = AnalystAgent(
                request=request,
                memory_entity=self.memory,
                prompt_template=prompts
            )
            
            # 执行分析
            result = await analyst.process()
            
            return {
                "success": True,
                "analysis_result": result,
                "request": request,
                "agent": "AnalysisEventAgent"
            }
            
        except Exception as e:
            logger.error(f"分析处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_data_classification(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据分类任务"""
        # 这里可以实现具体的数据分类逻辑
        logger.info("执行数据分类分析")
        return {
            "success": True,
            "classification": "text_processing",
            "confidence": 0.95
        }
    
    async def _handle_conversation_router(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理对话路由任务"""
        logger.info("执行对话路由分析")
        user_query = payload.get("user_query", "")
        
        # 简单的意图分类逻辑
        if "算子" in user_query or "operator" in user_query.lower():
            intent = "create_operator"
        elif "pipeline" in user_query.lower() or "流水线" in user_query:
            intent = "recommend_pipeline"
        elif "优化" in user_query or "optimize" in user_query.lower():
            intent = "optimize_operator"
        else:
            intent = "general_query"
        
        return {
            "success": True,
            "intent": intent,
            "confidence": 0.8,
            "next_action": f"route_to_{intent}"
        }

class ExecutionEventAgent(AgentInterface):
    """执行事件处理Agent"""
    
    def __init__(self):
        super().__init__("execution")
        self.supported_events = ["execution", "code_generation", "operator_creation"]
        self.memory = Memory()
        
    async def handle(self, event: Event) -> Dict[str, Any]:
        """处理执行类事件"""
        event_name = event.name
        payload = event.payload
        
        logger.info(f"⚡ 开始处理执行事件: {event_name}")
        
        try:
            if event_name == "execution":
                return await self._handle_execution(payload)
            elif event_name == "code_generation":
                return await self._handle_code_generation(payload)
            elif event_name == "operator_creation":
                return await self._handle_operator_creation(payload)
            else:
                raise ValueError(f"不支持的执行事件类型: {event_name}")
                
        except Exception as e:
            logger.error(f"执行事件处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_execution(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理通用执行任务"""
        try:
            request = payload.get("request")
            analysis_result = payload.get("analysis_result")
            
            if not request:
                raise ValueError("缺少请求信息")
            
            # 创建执行Agent
            prompts = PromptsTemplateGenerator(request.language)
            executor = ExecutionAgent(
                request=request,
                memory_entity=self.memory,
                prompt_template=prompts
            )
            
            # 执行任务
            result = await executor.execute()
            
            return {
                "success": True,
                "execution_result": result,
                "generated_code": result.get("code"),
                "output_file": result.get("file_path"),
                "agent": "ExecutionEventAgent"
            }
            
        except Exception as e:
            logger.error(f"执行处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_code_generation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理代码生成任务"""
        logger.info("执行代码生成")
        
        requirements = payload.get("requirements", "")
        template = payload.get("template", "basic")
        
        # 模拟代码生成
        generated_code = f'''
def generated_function():
    """
    根据需求生成的函数: {requirements}
    """
    pass
'''
        
        return {
            "success": True,
            "generated_code": generated_code,
            "template_used": template
        }
    
    async def _handle_operator_creation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理算子创建任务"""
        logger.info("执行算子创建")
        
        operator_type = payload.get("operator_type", "unknown")
        specifications = payload.get("specifications", {})
        
        return {
            "success": True,
            "operator_created": True,
            "operator_type": operator_type,
            "specifications": specifications
        }

class DebugEventAgent(AgentInterface):
    """调试事件处理Agent"""
    
    def __init__(self):
        super().__init__("debug")
        self.supported_events = ["debug", "code_validation", "error_fixing"]
        self.memory = Memory()
        
    async def handle(self, event: Event) -> Dict[str, Any]:
        """处理调试类事件"""
        event_name = event.name
        payload = event.payload
        
        logger.info(f"🐛 开始处理调试事件: {event_name}")
        
        try:
            if event_name == "debug":
                return await self._handle_debug(payload)
            elif event_name == "code_validation":
                return await self._handle_code_validation(payload)
            elif event_name == "error_fixing":
                return await self._handle_error_fixing(payload)
            else:
                raise ValueError(f"不支持的调试事件类型: {event_name}")
                
        except Exception as e:
            logger.error(f"调试事件处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_debug(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理通用调试任务"""
        try:
            request = payload.get("request")
            execution_result = payload.get("execution_result")
            
            if not request:
                raise ValueError("缺少请求信息")
            
            # 创建调试Agent
            prompts = PromptsTemplateGenerator(request.language)
            debugger = DebugAgent(
                tasks=[],  # 任务链由事件系统管理
                memory_entity=self.memory,
                request=request
            )
            
            # 执行调试
            result = await debugger.debug()
            
            return {
                "success": result.get("success", True),
                "debug_result": result,
                "errors_found": result.get("errors", []),
                "fixes_applied": result.get("fixes", []),
                "agent": "DebugEventAgent"
            }
            
        except Exception as e:
            logger.error(f"调试处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_code_validation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理代码验证任务"""
        logger.info("执行代码验证")
        
        code = payload.get("code", "")
        if not code:
            return {"success": False, "error": "没有提供要验证的代码"}
        
        # 模拟代码验证
        validation_result = {
            "syntax_valid": True,
            "logic_valid": True,
            "warnings": [],
            "suggestions": ["添加错误处理", "优化性能"]
        }
        
        return {
            "success": True,
            "validation_result": validation_result
        }
    
    async def _handle_error_fixing(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理错误修复任务"""
        logger.info("执行错误修复")
        
        errors = payload.get("errors", [])
        code = payload.get("code", "")
        
        return {
            "success": True,
            "fixes_applied": len(errors),
            "fixed_code": code,  # 这里应该是修复后的代码
            "remaining_errors": []
        }

class CompletionEventAgent(AgentInterface):
    """完成事件处理Agent"""
    
    def __init__(self):
        super().__init__("completion")
        self.supported_events = ["completion", "result_formatting", "report_generation"]
        
    async def handle(self, event: Event) -> Dict[str, Any]:
        """处理完成类事件"""
        event_name = event.name
        payload = event.payload
        
        logger.info(f"🎯 开始处理完成事件: {event_name}")
        
        try:
            if event_name == "completion":
                return await self._handle_completion(payload)
            elif event_name == "result_formatting":
                return await self._handle_result_formatting(payload)
            elif event_name == "report_generation":
                return await self._handle_report_generation(payload)
            else:
                raise ValueError(f"不支持的完成事件类型: {event_name}")
                
        except Exception as e:
            logger.error(f"完成事件处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理任务完成"""
        logger.info("🎉 任务流程完成")
        
        final_result = payload.get("final_result", {})
        
        # 汇总整个流程的结果
        summary = {
            "status": "completed",
            "final_result": final_result,
            "completion_time": payload.get("completion_time"),
            "total_events_processed": payload.get("total_events", 0)
        }
        
        return {
            "success": True,
            "completion_summary": summary,
            "agent": "CompletionEventAgent"
        }
    
    async def _handle_result_formatting(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理结果格式化"""
        logger.info("格式化结果")
        
        raw_result = payload.get("raw_result", {})
        format_type = payload.get("format", "json")
        
        return {
            "success": True,
            "formatted_result": raw_result,  # 这里应该根据format_type格式化
            "format": format_type
        }
    
    async def _handle_report_generation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理报告生成"""
        logger.info("生成执行报告")
        
        events_history = payload.get("events_history", [])
        
        report = {
            "total_events": len(events_history),
            "successful_events": len([e for e in events_history if e.get("success")]),
            "execution_summary": "任务执行完成"
        }
        
        return {
            "success": True,
            "report": report
        }

class FormerEventAgent(AgentInterface):
    """Former Agent事件处理器 - 处理XML表单相关事件"""
    
    def __init__(self):
        super().__init__("former")
        self.supported_events = [
            "form_conversation", 
            "xml_generation", 
            "form_validation",
            "requirement_collection"
        ]
        self.former_agent = FormerAgent()  # FormerAgent不需要参数，它会自己初始化FormTemplateManager
        
    async def handle(self, event: Event) -> Dict[str, Any]:
        """处理Former Agent相关事件"""
        event_name = event.name
        payload = event.payload
        
        logger.info(f"📝 开始处理Former事件: {event_name}")
        
        try:
            if event_name == "form_conversation":
                return await self._handle_form_conversation(payload)
            elif event_name == "xml_generation":
                return await self._handle_xml_generation(payload)
            elif event_name == "form_validation":
                return await self._handle_form_validation(payload)
            elif event_name == "requirement_collection":
                return await self._handle_requirement_collection(payload)
            else:
                raise ValueError(f"不支持的Former事件类型: {event_name}")
                
        except Exception as e:
            logger.error(f"Former事件处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_form_conversation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理表单对话"""
        try:
            user_query = payload.get("user_query", "")
            session_id = payload.get("session_id")
            conversation_history = payload.get("conversation_history", [])
            
            if not user_query:
                raise ValueError("缺少用户查询内容")
            
            # 创建FormRequest
            form_request = FormRequest(
                user_query=user_query,
                session_id=session_id,
                conversation_history=conversation_history
            )
            
            # 调用Former Agent处理对话
            response = await self.former_agent.process_conversation(form_request)
            
            return {
                "success": True,
                "session_id": response.session_id if hasattr(response, 'session_id') else form_request.session_id,
                "agent_response": response.agent_response,
                "need_more_info": response.need_more_info,
                "xml_form": response.xml_form,
                "form_type": response.form_type,
                "conversation_history": response.conversation_history,
                "agent": "FormerEventAgent"
            }
            
        except Exception as e:
            logger.error(f"表单对话处理失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_xml_generation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理XML生成"""
        logger.info("生成XML表单")
        
        requirements = payload.get("requirements", {})
        form_type = payload.get("form_type", "create_operator")
        
        # 这里可以调用Former Agent的XML生成逻辑
        # 或者直接使用模板管理器生成
        template_manager = self.former_agent.template_manager
        xml_schema = template_manager.get_xml_schema(form_type)
        
        # 根据需求生成XML
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<workflow>
    <form_type>{form_type}</form_type>
    <requirements>{requirements}</requirements>
    <generated_by>FormerEventAgent</generated_by>
</workflow>"""
        
        return {
            "success": True,
            "xml_content": xml_content,
            "form_type": form_type,
            "schema": xml_schema
        }
    
    async def _handle_form_validation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理表单验证"""
        logger.info("验证XML表单")
        
        xml_content = payload.get("xml_content", "")
        form_type = payload.get("form_type", "")
        
        if not xml_content:
            return {"success": False, "error": "没有提供XML内容"}
        
        try:
            # 使用Former Agent的模板管理器验证
            template_manager = self.former_agent.template_manager
            is_valid = template_manager.validate_xml(xml_content, form_type)
            
            return {
                "success": True,
                "is_valid": is_valid,
                "xml_content": xml_content,
                "form_type": form_type
            }
            
        except Exception as e:
            return {
                "success": False,
                "is_valid": False,
                "error": str(e)
            }
    
    async def _handle_requirement_collection(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理需求收集"""
        logger.info("收集用户需求")
        
        user_input = payload.get("user_input", "")
        current_requirements = payload.get("current_requirements", {})
        
        # 简单的需求提取逻辑
        extracted_requirements = {
            **current_requirements,
            "user_input": user_input,
            "extracted_keywords": user_input.split(),
            "timestamp": payload.get("timestamp")
        }
        
        return {
            "success": True,
            "requirements": extracted_requirements,
            "extraction_complete": len(extracted_requirements) > 3
        }
