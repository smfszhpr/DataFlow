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
