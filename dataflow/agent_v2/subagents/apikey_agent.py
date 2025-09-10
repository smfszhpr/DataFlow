#!/usr/bin/env python3
"""
API Key S    def __init__(self):
        # 硬编码的固定API密钥 - 方便测试验证
        self.secret_apikey = "DFlow2024Secret"
        
        # 调用父类初始化
        super().__init__()- 用于测试Master Agent决策功能
提供硬编码的"今天的API密钥"，只有正确调用才能获取
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel

# 导入基础架构
from ..base.core import SubAgent, BaseTool, NodeType, node, entry, conditional_edge


class APIKeyState(BaseModel):
    """API密钥SubAgent状态定义"""
    user_message: str = ""
    subagent: str = ""
    status: str = "initialized"
    request_time: str = ""
    challenge_passed: bool = False
    validation_result: str = ""
    result: str = ""
    apikey: str = ""
    access_granted: bool = False
    message: str = ""
    hint: str = ""


class APIKeyAgent(SubAgent):
    """API密钥获取SubAgent
    
    硬编码今天的秘密API密钥，用于测试Master Agent的决策和路由功能
    """
    
    def __init__(self):
        # 硬编码的固定API密钥 - 方便测试验证
        self.secret_apikey = "123121323132"
        
        print(f"🔐 [APIKeyAgent] 初始化完成，秘密密钥: {self.secret_apikey}")
        
        # 调用父类初始化
        super().__init__()
    
    def state_definition(self) -> type[BaseModel]:
        """返回状态定义"""
        return APIKeyState
    
    def _setup_graph(self):
        """设置SubAgent的执行图"""
        # 添加节点
        self.graph_builder.add_node("bootstrap", self.bootstrap, NodeType.ENTRY)
        self.graph_builder.add_node("validate_request", self.validate_request, NodeType.PROCESSOR)
        self.graph_builder.add_node("provide_apikey", self.provide_apikey, NodeType.PROCESSOR)
        self.graph_builder.add_node("deny_access", self.deny_access, NodeType.END)
        
        # 设置入口点
        self.graph_builder.set_entry_point("bootstrap")
        
        # 设置边
        self.graph_builder.add_edge("bootstrap", "validate_request")
        self.graph_builder.add_conditional_edge(
            "validate_request",
            self.should_provide_key,
            {
                "provide_apikey": "provide_apikey",
                "deny_access": "deny_access"
            }
        )
    
    async def execute(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """执行SubAgent"""
        # 转换为状态对象
        state = APIKeyState(**initial_state)
        
        # 执行bootstrap
        state_dict = await self.bootstrap(state.dict())
        
        # 执行validate_request
        state_dict = await self.validate_request(state_dict)
        
        # 根据验证结果决定下一步
        next_action = await self.should_provide_key(state_dict)
        
        if next_action == "provide_apikey":
            state_dict = await self.provide_apikey(state_dict)
        else:
            state_dict = await self.deny_access(state_dict)
        
        return state_dict
    
    @entry
    @node()
    async def bootstrap(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """初始化阶段"""
        
        user_message = state.get("user_message", "")
        
        state.update({
            "status": "analyzing",
            "request_time": datetime.now().isoformat(),
            "subagent": "APIKeyAgent"
        })
        return state
    
    @node()
    async def validate_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """验证请求是否有效"""
        user_message = state.get("user_message", "").lower()
        
        # 检查是否包含正确的请求关键词
        valid_keywords = ["apikey", "api key", "密钥", "秘密", "今天", "获取"]
        
        has_valid_keyword = any(keyword in user_message for keyword in valid_keywords)
        
        if has_valid_keyword:
            state["challenge_passed"] = True
            state["validation_result"] = "✅ 请求验证通过"
        else:
            state["challenge_passed"] = False
            state["validation_result"] = "❌ 请求验证失败，缺少必要关键词"
        
        return state
    
    async def should_provide_key(self, state: Dict[str, Any]) -> str:
        """决定是否提供API密钥的条件函数"""
        challenge_passed = state.get("challenge_passed", False)
        return "provide_apikey" if challenge_passed else "deny_access"
    
    @node()
    async def provide_apikey(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """提供API密钥"""
        print(f"🔐 [APIKeyAgent] 提供API密钥: {self.secret_apikey}")
        
        state.update({
            "status": "completed",
            "result": f"🔑 秘密API密钥: {self.secret_apikey}",
            "apikey": self.secret_apikey,
            "access_granted": True,
            "message": f"✅ 成功获取秘密API密钥"
        })
        
        return state
    
    @node()
    async def deny_access(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """拒绝访问"""
        print(f"🔐 [APIKeyAgent] 拒绝访问，请求不符合要求")
        
        state.update({
            "status": "denied", 
            "result": "❌ 访问被拒绝",
            "access_granted": False,
            "message": "❌ 无法获取API密钥，请使用正确的请求方式",
            "hint": "💡 提示：请明确表达您要获取秘密API密钥"
        })
        
        return state
    
    def build_graph(self):
        """构建SubAgent的执行图"""
        
        # 设置节点连接
        self.graph.add_edge("bootstrap", "validate_request")
        self.graph.add_conditional_edges(
            "validate_request",
            self.should_provide_key,
            {
                "provide_apikey": "provide_apikey",
                "deny_access": "deny_access"
            }
        )

        return self.graph


class APIKeyTool(BaseTool):
    """API密钥工具包装器，供Master Agent调用"""
    
    def __init__(self):
        self.agent = APIKeyAgent()
    
    @classmethod
    def name(cls) -> str:
        return "APIKey获取工具"
    
    @classmethod
    def description(cls) -> str:
        return "获取今天的秘密API密钥，用于系统认证。适用于需要获取秘密密钥或API Key的场景。"
    
    def params(self) -> type[BaseModel]:
        class APIKeyParams(BaseModel):
            user_message: str
        return APIKeyParams
    
    async def execute(self, user_message: str) -> Dict[str, Any]:
        """执行API密钥获取"""
        try:
            
            # 调用APIKeyAgent执行
            initial_state = {
                "user_message": user_message,
                "subagent": "APIKeyAgent",
                "status": "initialized", 
                "request_time": datetime.now().isoformat()
            }
            
            result = await self.agent.execute(initial_state)

            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": "获取API密钥失败",
                "status": "error"
            }
