"""
统一的工具结果规范 - 避免对特定工具的特判
"""
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class ToolStatus(str, Enum):
    """工具执行状态"""
    SUCCESS = "success"           # 成功完成
    NEED_USER_INPUT = "need_user_input"  # 需要用户输入
    FAILED = "failed"            # 执行失败
    WAITING = "waiting"          # 等待中（如表单收集）
    PARTIAL = "partial"          # 部分完成


class ToolArtifact(BaseModel):
    """工具生成的产物"""
    type: str  # "code", "form", "data", etc.
    content: Any
    metadata: Dict[str, Any] = {}


class ToolFollowup(BaseModel):
    """后续建议"""
    needs_followup: bool = False
    suggested_tool: Optional[str] = None
    reason: str = ""
    parameters: Dict[str, Any] = {}


class ToolResult(BaseModel):
    """统一的工具结果格式"""
    tool_name: str
    status: ToolStatus
    
    # 核心结果数据
    payload: Dict[str, Any] = {}
    
    # 用户消息（前端显示）
    message: str = ""
    
    # 生成的产物（代码、表单等）
    artifacts: List[ToolArtifact] = []
    
    # 后续建议
    followup: Optional[ToolFollowup] = None
    
    # 状态同步数据
    sync_data: Dict[str, Any] = {}
    
    # 原始返回（调试用）
    raw_result: Optional[Dict[str, Any]] = None
    
    @classmethod
    def success(cls, tool_name: str, message: str = "", **kwargs) -> "ToolResult":
        """创建成功结果"""
        return cls(
            tool_name=tool_name,
            status=ToolStatus.SUCCESS,
            message=message,
            **kwargs
        )
    
    @classmethod
    def need_user_input(cls, tool_name: str, message: str = "", **kwargs) -> "ToolResult":
        """创建需要用户输入的结果"""
        return cls(
            tool_name=tool_name,
            status=ToolStatus.NEED_USER_INPUT,
            message=message,
            **kwargs
        )
    
    @classmethod
    def failed(cls, tool_name: str, message: str = "", **kwargs) -> "ToolResult":
        """创建失败结果"""
        return cls(
            tool_name=tool_name,
            status=ToolStatus.FAILED,
            message=message,
            **kwargs
        )


class ToolResultAdapter:
    """工具结果适配器 - 将现有工具的返回格式转换为统一格式"""
    
    @staticmethod
    def adapt_former_result(raw_result: Dict[str, Any]) -> ToolResult:
        """适配 Former 工具的结果"""
        
        # 判断状态
        if raw_result.get("requires_user_input"):
            status = ToolStatus.NEED_USER_INPUT
        elif raw_result.get("success", True):
            status = ToolStatus.SUCCESS
        else:
            status = ToolStatus.FAILED
        
        # 提取消息（处理None情况）
        message = raw_result.get("message") or ""
        if not isinstance(message, str):
            message = str(message) if message is not None else ""
        
        # 提取同步数据
        sync_data = {}
        if "session_id" in raw_result:
            sync_data["session_id"] = raw_result["session_id"]
        if "form_session" in raw_result:
            sync_data["form_session"] = raw_result["form_session"]
        if "form_data" in raw_result:
            sync_data["form_data"] = raw_result["form_data"]
        
        # 提取后续建议
        followup = None
        if raw_result.get("target_workflow"):
            followup = ToolFollowup(
                needs_followup=True,
                suggested_tool=raw_result["target_workflow"],
                reason="表单收集完成，执行目标工作流",
                parameters=raw_result.get("form_data", {})
            )
        
        # 创建表单产物
        artifacts = []
        if raw_result.get("form_data"):
            artifacts.append(ToolArtifact(
                type="form",
                content=raw_result["form_data"],
                metadata={
                    "form_stage": raw_result.get("form_stage"),
                    "form_complete": raw_result.get("form_complete", False)
                }
            ))
        
        return ToolResult(
            tool_name="former",
            status=status,
            message=message,
            payload=raw_result,
            sync_data=sync_data,
            followup=followup,
            artifacts=artifacts,
            raw_result=raw_result
        )
    
    @staticmethod
    def adapt_pipeline_result(raw_result: Dict[str, Any]) -> ToolResult:
        """适配 Pipeline Workflow 工具的结果"""
        
        # 判断状态
        status = ToolStatus.SUCCESS if raw_result.get("success", False) else ToolStatus.FAILED
        
        # 提取消息
        message = raw_result.get("output", "")
        
        # 创建代码产物
        artifacts = []
        if raw_result.get("generated_pipeline_code"):
            artifacts.append(ToolArtifact(
                type="code",
                content=raw_result["generated_pipeline_code"],
                metadata={
                    "language": "python",
                    "file_path": raw_result.get("file_path"),
                    "classification_result": raw_result.get("classification_result"),
                    "recommendation_result": raw_result.get("recommendation_result")
                }
            ))
        
        return ToolResult(
            tool_name="pipeline_workflow_agent",
            status=status,
            message=message,
            payload=raw_result,
            artifacts=artifacts,
            raw_result=raw_result
        )
    
    @staticmethod
    def adapt_generic_result(tool_name: str, raw_result: Any) -> ToolResult:
        """适配通用工具结果"""
        
        if isinstance(raw_result, dict):
            status = ToolStatus.SUCCESS if raw_result.get("success", True) else ToolStatus.FAILED
            message = raw_result.get("output", raw_result.get("message", str(raw_result)))
            return ToolResult(
                tool_name=tool_name,
                status=status,
                message=message,
                payload=raw_result if isinstance(raw_result, dict) else {"result": raw_result},
                raw_result=raw_result
            )
        else:
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.SUCCESS,
                message=str(raw_result),
                payload={"result": raw_result},
                raw_result=raw_result
            )


def adapt_tool_result(tool_name: str, raw_result: Any) -> ToolResult:
    """根据工具名称自动适配结果"""
    
    if tool_name == "former":
        return ToolResultAdapter.adapt_former_result(raw_result)
    elif tool_name in ["pipeline_workflow_agent", "pipeline_workflow_tool"]:
        return ToolResultAdapter.adapt_pipeline_result(raw_result)
    else:
        return ToolResultAdapter.adapt_generic_result(tool_name, raw_result)
