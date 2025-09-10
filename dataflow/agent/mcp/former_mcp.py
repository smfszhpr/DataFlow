"""
Former Agent MCP协议接口
专注于XML表单生成，不直接与用户对话
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from dataflow.logger import get_logger
from dataflow.agent.agentrole.former import FormerAgent

logger = get_logger()

class MCPRequest(BaseModel):
    """MCP统一请求格式"""
    action: str  # 操作类型：analyze_requirement, check_fields, generate_xml, get_current_form
    user_input: str  # 用户输入
    context: Dict[str, Any] = {}  # 上下文信息
    form_state: Dict[str, Any] = {}  # 当前表单状态

class MCPResponse(BaseModel):
    """MCP统一响应格式"""
    status: str  # success, need_more_info, error
    message: str  # 给主LLM的消息
    data: Dict[str, Any] = {}  # 结构化数据
    next_action: Optional[str] = None  # 建议的下一步操作

class FormerAgentMCP:
    """Former Agent MCP接口实现"""
    
    def __init__(self):
        self.former_agent = FormerAgent()
        self.form_memory = {}  # 存储用户的表单状态
        
    async def process_mcp_request(self, request: MCPRequest) -> MCPResponse:
        """处理MCP请求"""
        try:
            if request.action == "analyze_requirement":
                return await self._analyze_requirement(request)
            elif request.action == "check_fields":
                return await self._check_fields(request)
            elif request.action == "generate_xml":
                return await self._generate_xml(request)
            elif request.action == "get_current_form":
                return await self._get_current_form(request)
            else:
                return MCPResponse(
                    status="error",
                    message=f"未知操作: {request.action}",
                    data={"error": "invalid_action"}
                )
        except Exception as e:
            logger.error(f"Former Agent MCP处理失败: {e}")
            return MCPResponse(
                status="error",
                message=f"处理失败: {str(e)}",
                data={"error": str(e)}
            )
    
    async def _analyze_requirement(self, request: MCPRequest) -> MCPResponse:
        """分析用户需求，判断是否足够生成表单"""
        user_input = request.user_input
        session_id = request.context.get('session_id', 'default')
        
        # 将用户输入添加到记忆中
        if session_id not in self.form_memory:
            self.form_memory[session_id] = {
                'user_inputs': [],
                'form_type': None,
                'extracted_fields': {},
                'required_fields': [],
                'optional_fields': []
            }
        
        self.form_memory[session_id]['user_inputs'].append(user_input)
        
        # 使用Former Agent分析需求
        form_type = await self.former_agent._detect_form_type_with_llm(user_input)
        self.form_memory[session_id]['form_type'] = form_type
        
        # 获取该表单类型需要的字段
        xml_schema = self.former_agent.template_manager.get_xml_schema(form_type)
        required_fields = xml_schema.get('required_fields', [])
        optional_fields = xml_schema.get('optional_fields', [])
        
        self.form_memory[session_id]['required_fields'] = required_fields
        self.form_memory[session_id]['optional_fields'] = optional_fields
        
        # 分析已收集的字段
        collected_fields = self._extract_fields_from_inputs(
            self.form_memory[session_id]['user_inputs'], 
            required_fields + optional_fields
        )
        self.form_memory[session_id]['extracted_fields'] = collected_fields
        
        # 检查是否有足够信息
        missing_required = [f for f in required_fields if f not in collected_fields or not collected_fields[f]]
        
        if missing_required:
            return MCPResponse(
                status="need_more_info",
                message=f"需要了解更多信息才能生成{form_type}表单",
                data={
                    "form_type": form_type,
                    "missing_fields": missing_required,
                    "collected_fields": collected_fields,
                    "questions_for_user": self._generate_questions_for_fields(missing_required)
                },
                next_action="collect_more_info"
            )
        else:
            return MCPResponse(
                status="success",
                message=f"需求分析完成，可以生成{form_type}表单",
                data={
                    "form_type": form_type,
                    "collected_fields": collected_fields,
                    "ready_for_xml": True
                },
                next_action="generate_xml"
            )
    
    async def _check_fields(self, request: MCPRequest) -> MCPResponse:
        """检查特定字段是否完整"""
        session_id = request.context.get('session_id', 'default')
        
        if session_id not in self.form_memory:
            return MCPResponse(
                status="error",
                message="未找到表单状态，请先分析需求",
                data={"error": "no_form_state"}
            )
        
        form_state = self.form_memory[session_id]
        required_fields = form_state['required_fields']
        collected_fields = form_state['extracted_fields']
        
        missing_fields = [f for f in required_fields if f not in collected_fields or not collected_fields[f]]
        
        if missing_fields:
            return MCPResponse(
                status="need_more_info",
                message="还需要补充一些信息",
                data={
                    "missing_fields": missing_fields,
                    "collected_fields": collected_fields,
                    "questions": self._generate_questions_for_fields(missing_fields)
                },
                next_action="collect_more_info"
            )
        else:
            return MCPResponse(
                status="success",
                message="所有必要字段已收集完成",
                data={
                    "all_fields_complete": True,
                    "collected_fields": collected_fields
                },
                next_action="generate_xml"
            )
    
    async def _generate_xml(self, request: MCPRequest) -> MCPResponse:
        """生成完整的XML表单"""
        session_id = request.context.get('session_id', 'default')
        
        if session_id not in self.form_memory:
            return MCPResponse(
                status="error",
                message="未找到表单状态",
                data={"error": "no_form_state"}
            )
        
        form_state = self.form_memory[session_id]
        form_type = form_state['form_type']
        collected_fields = form_state['extracted_fields']
        
        # 生成XML
        xml_content = self._build_xml_from_fields(form_type, collected_fields)
        
        return MCPResponse(
            status="success",
            message="XML表单生成完成",
            data={
                "xml_content": xml_content,
                "form_type": form_type,
                "fields_used": collected_fields
            },
            next_action="xml_ready"
        )
    
    async def _get_current_form(self, request: MCPRequest) -> MCPResponse:
        """获取当前表单状态"""
        session_id = request.context.get('session_id', 'default')
        
        if session_id not in self.form_memory:
            return MCPResponse(
                status="error",
                message="未找到表单状态",
                data={"error": "no_form_state"}
            )
        
        form_state = self.form_memory[session_id]
        
        return MCPResponse(
            status="success",
            message="当前表单状态",
            data={
                "form_type": form_state.get('form_type'),
                "collected_fields": form_state.get('extracted_fields', {}),
                "required_fields": form_state.get('required_fields', []),
                "user_inputs": form_state.get('user_inputs', [])
            },
            next_action="show_form_status"
        )
    
    def _extract_fields_from_inputs(self, inputs: List[str], field_names: List[str]) -> Dict[str, str]:
        """从用户输入中提取字段信息（简化版本）"""
        # 这里应该使用NLP或者LLM来智能提取，暂时用简化逻辑
        collected = {}
        combined_input = " ".join(inputs).lower()
        
        for field in field_names:
            if field == "operator_name" and ("算子" in combined_input or "operator" in combined_input):
                # 提取算子名称的简单逻辑
                collected[field] = "custom_operator"
            elif field == "input_format" and ("输入" in combined_input or "input" in combined_input):
                collected[field] = "文本数据"
            elif field == "output_format" and ("输出" in combined_input or "output" in combined_input):
                collected[field] = "分类结果"
            # 更多字段提取逻辑...
        
        return collected
    
    def _generate_questions_for_fields(self, missing_fields: List[str]) -> List[str]:
        """为缺失字段生成问题"""
        questions = []
        for field in missing_fields:
            if field == "operator_name":
                questions.append("请问您希望给这个算子起什么名称？")
            elif field == "input_format":
                questions.append("请描述一下输入数据的格式和结构？")
            elif field == "output_format":
                questions.append("您期望的输出格式是什么样的？")
            elif field == "algorithm_type":
                questions.append("您希望使用哪种算法或处理方式？")
            else:
                questions.append(f"请提供{field}的相关信息")
        return questions
    
    def _build_xml_from_fields(self, form_type: str, fields: Dict[str, str]) -> str:
        """根据字段构建XML"""
        # 这里应该根据实际的XML模板来生成
        xml_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<workflow>
    <form_type>{form_type}</form_type>
    <operator_name>{fields.get('operator_name', 'CustomOperator')}</operator_name>
    <input_format>{fields.get('input_format', '文本数据')}</input_format>
    <output_format>{fields.get('output_format', '处理结果')}</output_format>
    <algorithm_type>{fields.get('algorithm_type', 'custom')}</algorithm_type>
    <description>基于用户需求自动生成的算子配置</description>
</workflow>"""
        return xml_template

# 全局实例
former_agent_mcp = FormerAgentMCP()
