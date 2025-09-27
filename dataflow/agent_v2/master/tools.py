"""
tools.py
职责：统一工具注册、LangChain 工具转换、ToolExecutor 初始化、工作流配置管理
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import StructuredTool, BaseTool as LangChainBaseTool
from langgraph.prebuilt import ToolExecutor

# 导入所有工具
from dataflow.agent_v2.subagents.apikey_agent import APIKeyTool
from dataflow.agent_v2.subagents.former_tool import FormerTool  
from dataflow.agent_v2.subagents.code_workflow_tool import CodeWorkflowTool, CodeWorkflowToolParams
from dataflow.agent_v2.subagents.pipeline_workflow_tool import PipelineWorkflowTool, PipelineWorkflowToolParams
from dataflow.agent_v2.subagents.mock_tools import SleepTool
from dataflow.agent_v2.subagents.csvtools import CSVProfileTool, CSVDetectTimeColumnsTool, CSVVegaSpecTool, ASTStaticCheckTool, UnitTestStubTool, LocalIndexBuildTool, LocalIndexQueryTool

from dataflow.agent_v2.base.core import BaseTool

logger = logging.getLogger(__name__)


def to_langchain_tool(tool: BaseTool) -> StructuredTool:
    """将自定义工具对象转换为LangChain的StructuredTool"""
    ArgsSchema = tool.params()

    return StructuredTool.from_function(
        coroutine=tool.execute,  # 直接使用工具的异步execute方法
        name=tool.name(),
        description=tool.description(),
        args_schema=ArgsSchema,
        return_direct=False,
    )


class WorkflowRegistry:
    """工作流配置管理器"""
    
    def __init__(self):
        self.workflows = self._discover_available_workflows()
    
    def _discover_available_workflows(self) -> Dict[str, Dict[str, Any]]:
        """动态发现可用工作流及其真实参数定义"""
        workflows = {}
        
        try:
            # 通过反射获取真实的参数定义
            code_workflow_params = self._extract_params_from_pydantic_model(CodeWorkflowToolParams)
            pipeline_workflow_params = self._extract_params_from_pydantic_model(PipelineWorkflowToolParams)
            
            workflows["code_workflow_agent"] = {
                "description": "代码生成、测试、调试循环工具",
                "params_schema": code_workflow_params,
                "tool_class": "CodeWorkflowTool"
            }
            
            workflows["pipeline_workflow_agent"] = {
                "description": "数据处理流水线推荐工具", 
                "params_schema": pipeline_workflow_params,
                "tool_class": "PipelineWorkflowTool"
            }
            
            logger.info(f"发现 {len(workflows)} 个工作流")
            
        except Exception as e:
            logger.error(f"工作流发现失败: {e}")
            # 回退到基础定义
            workflows["code_workflow_agent"] = {
                "description": "代码生成工具",
                "params_schema": {
                    "requirement": {"required": True, "type": "str", "description": "用户代码需求"}
                },
                "tool_class": "CodeWorkflowTool"
            }
            workflows["pipeline_workflow_agent"] = {
                "description": "数据处理流水线推荐工具",
                "params_schema": {
                    "json_file": {"required": True, "type": "str", "description": "数据文件路径"},
                    "target": {"required": True, "type": "str", "description": "处理目标"}
                },
                "tool_class": "PipelineWorkflowTool"
            }
        
        return workflows
    
    def _extract_params_from_pydantic_model(self, model_class) -> Dict[str, Any]:
        """从Pydantic模型提取参数定义"""
        try:
            schema = model_class.model_json_schema()
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            params = {}
            for field_name, field_info in properties.items():
                params[field_name] = {
                    "type": field_info.get("type", "str"),
                    "description": field_info.get("description", f"{field_name}参数"),
                    "required": field_name in required,
                    "default": field_info.get("default")
                }
            
            return params
        except Exception as e:
            logger.error(f"提取参数定义失败: {e}")
            return {}
    
    def get_workflow(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """获取指定工作流配置"""
        return self.workflows.get(workflow_name)
    
    def get_all_workflows(self) -> Dict[str, Dict[str, Any]]:
        """获取所有工作流配置"""
        return self.workflows


class ToolsMixin:
    """工具管理混入类，为MasterAgent提供工具注册和管理功能"""
    
    def __init__(self, *args, **kwargs):
        # 不调用 super().__init__，只初始化工具相关属性
        # 初始化工具相关属性
        self.tools: List[BaseTool] = []
        self.lc_tools: List[StructuredTool] = []
        self.tool_executor: Optional[ToolExecutor] = None
        self.workflow_registry = WorkflowRegistry()
        
        # 执行工具注册
        self._register_tools()
    
    def _register_tools(self):
        """注册工具"""
        try:
            self.tools = [
                APIKeyTool(),
                FormerTool(),
                CodeWorkflowTool(),
                PipelineWorkflowTool(),
                SleepTool(),
                CSVProfileTool(), 
                CSVDetectTimeColumnsTool(), 
                CSVVegaSpecTool(), 
                ASTStaticCheckTool(), 
                UnitTestStubTool(), 
                LocalIndexBuildTool(), 
                LocalIndexQueryTool()
            ]
            
            logger.info(f"已注册 {len(self.tools)} 个可直接调用的工具")
            
        except Exception as e:
            logger.error(f"工具注册失败: {e}")
            self.tools = []
        
        # 确保 lc_tools 总是被设置
        self.lc_tools = [to_langchain_tool(t) for t in (self.tools or [])]
        
        # 初始化工具执行器
        try:
            self.tool_executor = ToolExecutor(self.lc_tools)
        except Exception as e:
            logger.error(f"ToolExecutor初始化失败: {e}")
            self.tool_executor = None
    
    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """根据名称获取工具实例"""
        for tool in self.tools:
            if tool.name() == tool_name:
                return tool
        return None
    
    def get_available_tool_names(self) -> List[str]:
        """获取所有可用工具名称"""
        return [tool.name() for tool in self.tools]
    
    def get_workflow_config(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """获取工作流配置"""
        return self.workflow_registry.get_workflow(workflow_name)
    
    def get_all_workflow_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有工作流配置"""
        return self.workflow_registry.get_all_workflows()
