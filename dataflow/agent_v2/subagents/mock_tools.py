"""
Mock工具集 - 用于测试多轮编排
"""
import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dataflow.agent_v2.base.core import BaseTool


class SleepTool(BaseTool):
    """模拟耗时操作工具"""
    
    @classmethod
    def name(cls) -> str: 
        return "sleep_tool"
    
    @classmethod
    def description(cls) -> str: 
        return "等待N秒以模拟耗时操作"
    
    def params(self) -> type[BaseModel]:
        class SleepParams(BaseModel):
            seconds: float = 1.0
            label: str = "sleep"
        return SleepParams
    
    async def execute(self, seconds: float = 1.0, label: str = "sleep") -> Dict[str, Any]:
        await asyncio.sleep(seconds)
        print(f"⏰ [SleepTool] 完成等待 {seconds} 秒，标签: {label}")
        return {"success": True, "label": label, "slept": seconds}


class MockSearchTool(BaseTool):
    """模拟搜索工具"""
    
    @classmethod
    def name(cls) -> str: 
        return "mock_search"
    
    @classmethod
    def description(cls) -> str: 
        return "返回假搜索结果"
    
    def params(self) -> type[BaseModel]:
        class SearchParams(BaseModel):
            query: Optional[str] = Field(default="", description="搜索查询关键词")
            # 兼容LLM常用参数格式
            user_message: Optional[str] = Field(default="", description="用户消息，包含搜索需求")
        return SearchParams
    
    async def execute(self, query: Optional[str] = "", user_message: Optional[str] = "") -> Dict[str, Any]:
        # 参数兼容处理：如果没有query但有user_message，使用user_message作为query
        if not query and user_message:
            query = user_message
        
        if not query:
            return {"success": False, "error": "缺少搜索查询内容"}
        
        await asyncio.sleep(0.5)
        return {"success": True, "docs": [f"{query} result {i}" for i in range(3)]}


class MockFormerTool(BaseTool):
    """模拟表单生成工具"""
    
    @classmethod
    def name(cls) -> str: 
        return "former_agent_mock"
    
    @classmethod
    def description(cls) -> str: 
        return "模拟表单生成，need_more_info 开关"
    
    def params(self) -> type[BaseModel]:
        class FormerParams(BaseModel):
            user_query: Optional[str] = Field(default="", description="用户查询内容")
            need_more: bool = Field(default=False, description="是否需要更多信息")
            # 兼容LLM常用参数格式
            user_message: Optional[str] = Field(default="", description="用户消息")
        return FormerParams
    
    async def execute(self, user_query: Optional[str] = "", need_more: bool = False, user_message: Optional[str] = "") -> Dict[str, Any]:
        # 参数兼容处理：如果没有user_query但有user_message，使用user_message作为user_query
        if not user_query and user_message:
            user_query = user_message
        
        if not user_query:
            return {"success": False, "error": "缺少用户查询内容"}
        
        await asyncio.sleep(0.3)
        return {
            "success": True, 
            "need_more_info": need_more, 
            "agent_response": f"表单草稿 for: {user_query}"
        }


class MockCodeGenTool(BaseTool):
    """模拟代码生成工具"""
    
    @classmethod
    def name(cls) -> str: 
        return "code_generator_mock"
    
    @classmethod
    def description(cls) -> str: 
        return "模拟代码生成，需要提供需求描述"
    
    def params(self) -> type[BaseModel]:
        class CodeGenParams(BaseModel):
            requirements: Optional[str] = Field(default="", description="代码生成需求描述")
            # 兼容LLM常用参数格式
            user_message: Optional[str] = Field(default="", description="用户消息，包含代码生成需求")
        return CodeGenParams
    
    async def execute(self, requirements: Optional[str] = "", user_message: Optional[str] = "") -> Dict[str, Any]:
        # 参数兼容处理：如果没有requirements但有user_message，使用user_message作为requirements
        if not requirements and user_message:
            requirements = user_message
        
        if not requirements:
            return {"success": False, "error": "缺少代码生成需求描述"}
        
        await asyncio.sleep(0.4)
        return {
            "success": True, 
            "generated_code": f"# code for: {requirements}\nprint('Generated code for: {requirements}')",
            "requirements": requirements
        }
