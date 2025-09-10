"""
Former Agent V2 实现
基于 myscalekb-agent SubAgent 模式的重构版本
"""
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field
from enum import Enum
import json

from ..base.core import SubAgent, node, entry, conditional_edge, NodeType, GraphBuilder
from .tools import RequirementAnalysis, FieldValidation, XMLGeneration
from .prompt import FormerAgentPrompt


class ProcessingStatus(Enum):
    """处理状态枚举"""
    ANALYZING = "analyzing"
    VALIDATING = "validating"
    GENERATING = "generating"
    COMPLETED = "completed"
    ERROR = "error"


class FormerAgentState(BaseModel):
    """Former Agent 状态定义，模仿 TypedDict 模式"""
    
    # 输入信息
    user_requirement: str = Field(default="", description="用户需求")
    user_input: str = Field(default="", description="用户输入")
    
    # 分析结果
    form_type: Optional[str] = Field(default=None, description="表单类型")
    confidence: float = Field(default=0.0, description="置信度")
    reasoning: str = Field(default="", description="分析推理")
    template_name: Optional[str] = Field(default=None, description="模板名称")
    
    # 字段信息
    extracted_fields: Dict[str, Any] = Field(default_factory=dict, description="提取的字段")
    validated_fields: Dict[str, Any] = Field(default_factory=dict, description="验证的字段")
    missing_fields: List[str] = Field(default_factory=list, description="缺失字段")
    
    # 生成结果
    xml_content: str = Field(default="", description="XML内容")
    is_complete: bool = Field(default=False, description="是否完成")
    
    # 状态管理
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.ANALYZING, description="处理状态")
    error_message: str = Field(default="", description="错误信息")
    
    # 历史记录
    processing_history: List[Dict[str, Any]] = Field(default_factory=list, description="处理历史")


class FormerAgentV2(SubAgent):
    """Former Agent V2 - 基于 SubAgent 模式的重构版本"""
    
    def __init__(self):
        super().__init__()
        self.prompt = FormerAgentPrompt()
        
        # 添加工具
        self.add_tool(RequirementAnalysis())
        self.add_tool(FieldValidation())
        self.add_tool(XMLGeneration())
    
    def state_definition(self) -> Type[BaseModel]:
        """状态定义"""
        return FormerAgentState
    
    def _setup_graph(self):
        """设置图结构，定义工作流"""
        # 添加节点
        self.graph_builder.add_node("entry", self.entry_node, NodeType.ENTRY)
        self.graph_builder.add_node("analyze_requirement", self.analyze_requirement_node, NodeType.PROCESSOR)
        self.graph_builder.add_node("validate_fields", self.validate_fields_node, NodeType.PROCESSOR)
        self.graph_builder.add_node("generate_xml", self.generate_xml_node, NodeType.PROCESSOR)
        self.graph_builder.add_node("check_completion", self.check_completion_node, NodeType.CONDITIONAL)
        
        # 设置入口点
        self.graph_builder.set_entry_point("entry")
        
        # 添加边
        self.graph_builder.add_edge("entry", "analyze_requirement")
        self.graph_builder.add_edge("analyze_requirement", "validate_fields")
        self.graph_builder.add_edge("validate_fields", "check_completion")
        
        # 添加条件边
        self.graph_builder.add_conditional_edge(
            "check_completion",
            self._should_generate_xml,
            {
                "generate": "generate_xml",
                "incomplete": GraphBuilder.END,
                "error": GraphBuilder.END
            }
        )
        
        self.graph_builder.add_edge("generate_xml", GraphBuilder.END)
    
    @entry
    def entry_node(self, state: FormerAgentState) -> FormerAgentState:
        """入口节点"""
        # 记录开始处理
        state.processing_history.append({
            "step": "entry",
            "timestamp": self._get_timestamp(),
            "status": "started"
        })
        
        state.processing_status = ProcessingStatus.ANALYZING
        return state
    
    @node(NodeType.PROCESSOR)
    def analyze_requirement_node(self, state: FormerAgentState) -> FormerAgentState:
        """需求分析节点"""
        try:
            # 使用需求分析工具
            analysis_tool = RequirementAnalysis()
            result = analysis_tool.execute(
                user_requirement=state.user_requirement,
                context=state.user_input
            )
            
            if result["success"]:
                state.form_type = result["form_type"]
                state.confidence = result["confidence"]
                state.reasoning = result["reasoning"]
                state.template_name = result["suggested_template"]
                
                # 记录分析结果
                state.processing_history.append({
                    "step": "analyze_requirement",
                    "timestamp": self._get_timestamp(),
                    "result": result,
                    "status": "completed"
                })
            else:
                state.processing_status = ProcessingStatus.ERROR
                state.error_message = "需求分析失败"
        
        except Exception as e:
            state.processing_status = ProcessingStatus.ERROR
            state.error_message = f"需求分析异常: {str(e)}"
        
        return state
    
    @node(NodeType.PROCESSOR)
    def validate_fields_node(self, state: FormerAgentState) -> FormerAgentState:
        """字段验证节点"""
        try:
            state.processing_status = ProcessingStatus.VALIDATING
            
            # 使用字段验证工具
            validation_tool = FieldValidation()
            result = validation_tool.execute(
                form_type=state.form_type,
                extracted_fields=state.extracted_fields,
                user_input=state.user_input
            )
            
            if result["success"]:
                state.is_complete = result["is_complete"]
                state.missing_fields = result["missing_fields"]
                state.validated_fields = result["validated_fields"]
                
                # 记录验证结果
                state.processing_history.append({
                    "step": "validate_fields",
                    "timestamp": self._get_timestamp(),
                    "result": result,
                    "status": "completed"
                })
            else:
                state.processing_status = ProcessingStatus.ERROR
                state.error_message = "字段验证失败"
        
        except Exception as e:
            state.processing_status = ProcessingStatus.ERROR
            state.error_message = f"字段验证异常: {str(e)}"
        
        return state
    
    @node(NodeType.PROCESSOR)
    def generate_xml_node(self, state: FormerAgentState) -> FormerAgentState:
        """XML生成节点"""
        try:
            state.processing_status = ProcessingStatus.GENERATING
            
            # 使用XML生成工具
            generation_tool = XMLGeneration()
            result = generation_tool.execute(
                form_type=state.form_type,
                validated_fields=state.validated_fields,
                template_name=state.template_name
            )
            
            if result["success"]:
                state.xml_content = result["xml_content"]
                state.processing_status = ProcessingStatus.COMPLETED
                
                # 记录生成结果
                state.processing_history.append({
                    "step": "generate_xml",
                    "timestamp": self._get_timestamp(),
                    "result": result,
                    "status": "completed"
                })
            else:
                state.processing_status = ProcessingStatus.ERROR
                state.error_message = "XML生成失败"
        
        except Exception as e:
            state.processing_status = ProcessingStatus.ERROR
            state.error_message = f"XML生成异常: {str(e)}"
        
        return state
    
    @node(NodeType.CONDITIONAL)
    def check_completion_node(self, state: FormerAgentState) -> FormerAgentState:
        """检查完成状态节点"""
        # 这个节点主要用于条件判断，不修改状态
        return state
    
    def _should_generate_xml(self, state: FormerAgentState) -> str:
        """条件判断：是否应该生成XML"""
        if state.processing_status == ProcessingStatus.ERROR:
            return "error"
        elif not state.is_complete or state.missing_fields:
            return "incomplete"
        else:
            return "generate"
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def process_request(self, user_requirement: str, user_input: str = "") -> Dict[str, Any]:
        """处理请求的主入口"""
        # 初始化状态
        state = FormerAgentState(
            user_requirement=user_requirement,
            user_input=user_input
        )
        
        # 执行工作流
        current_node = self.graph_builder.entry_point
        
        while current_node != GraphBuilder.END:
            # 获取节点函数
            node_info = self.graph_builder.nodes[current_node]
            node_func = node_info['func']
            
            # 执行节点
            state = node_func(state)
            
            # 获取下一个节点
            if current_node in self.graph_builder.edges:
                edge_info = self.graph_builder.edges[current_node]
                
                if isinstance(edge_info, dict) and edge_info.get('type') == 'conditional':
                    # 条件边
                    condition_result = edge_info['condition'](state)
                    current_node = edge_info['map'].get(condition_result, GraphBuilder.END)
                elif isinstance(edge_info, list):
                    # 普通边
                    current_node = edge_info[0] if edge_info else GraphBuilder.END
                else:
                    current_node = GraphBuilder.END
            else:
                current_node = GraphBuilder.END
        
        # 返回结果
        return {
            "success": state.processing_status == ProcessingStatus.COMPLETED,
            "form_type": state.form_type,
            "xml_content": state.xml_content,
            "is_complete": state.is_complete,
            "missing_fields": state.missing_fields,
            "error_message": state.error_message,
            "processing_history": state.processing_history,
            "confidence": state.confidence
        }
