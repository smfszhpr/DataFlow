#!/usr/bin/env python3
"""
CodeWorkflow SubAgent - 代码生成、测试、调试的循环工作流
基于 myscalekb_agent_base 架构实现，集成 ExecutorSubAgent 和 DebuggerSubAgent
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field

# 使用 myscalekb_agent_base 库的组件
from myscalekb_agent_base.sub_agent import SubAgent
from myscalekb_agent_base.graph_builder import GraphBuilder, node, edge, conditional_entry
from myscalekb_agent_base.schemas.agent_metadata import AgentMetadata

from dataflow.agent_v2.subagents.executor_subagent import ExecutorSubAgent
from dataflow.agent_v2.subagents.debugger_subagent import DebuggerSubAgent

logger = logging.getLogger(__name__)


class CodeWorkflowParams(BaseModel):
    """CodeWorkflow 参数模型"""
    requirement: str = Field(description="用户代码需求")
    max_iterations: Optional[int] = Field(default=5, description="最大迭代次数")
    timeout_seconds: Optional[int] = Field(default=300, description="总超时时间（秒）")


class CodeWorkflowState(TypedDict, total=False):
    """CodeWorkflow 状态定义 - 兼容 myscalekb_agent_base"""
    # myscalekb_agent_base 标准字段
    input: Any
    query: str
    chat_history: List[Any]
    agent_metadata: AgentMetadata
    agent_outcome: Any
    intermediate_steps: List[Any]
    trace_id: Optional[str]
    
    # CodeWorkflow 特定字段
    requirement: str
    max_iterations: int
    timeout_seconds: int
    current_iteration: int
    current_code: str
    execution_successful: bool
    execution_output: str
    execution_error: str
    error_traceback: str
    debug_history: List[Dict[str, Any]]
    workflow_result: Optional[str]


class CodeWorkflowSubAgent(SubAgent):
    """代码生成-测试-调试循环工作流 SubAgent"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ExecutorSubAgent()
        self.debugger = DebuggerSubAgent()
    
    @classmethod
    def name(cls) -> str:
        return "code_workflow_sub_agent"
    
    @classmethod
    def description(cls) -> str:
        return "Automated code generation, testing and debugging workflow"
    
    @classmethod
    def state_definition(cls) -> type:
        return CodeWorkflowState
    
    @classmethod
    def forward_schema(cls) -> type:
        return CodeWorkflowParams
    
    @node
    async def entry_point(self, data):
        """入口点：初始化工作流"""
        logger.info("🚀 启动代码工作流")
        
        requirement = data.get("requirement", "")
        max_iterations = data.get("max_iterations", 5)
        timeout_seconds = data.get("timeout_seconds", 300)
        
        if not requirement:
            data["agent_outcome"] = self._make_agent_finish(
                "❌ 错误：缺少代码需求"
            )
            return data
        
        data.update({
            "requirement": requirement,
            "max_iterations": max_iterations,
            "timeout_seconds": timeout_seconds,
            "current_iteration": 0,
            "execution_successful": False,
            "debug_history": []
        })
        
        logger.info(f"📋 需求：{requirement}")
        logger.info(f"🔄 最大迭代次数：{max_iterations}")
        
        return data
    
    @node
    @edge(target_node="execute_and_test")
    async def generate_initial_code(self, data):
        """生成初始代码"""
        try:
            logger.info("🛠️ 生成初始代码...")
            
            requirement = data.get("requirement", "")
            
            # 调用 ExecutorSubAgent 生成代码
            executor_data = {
                "requirement": requirement,
                "additional_info": "请生成完整可执行的代码，包含必要的错误处理"
            }
            
            # 模拟 ExecutorSubAgent 的执行
            executor_result = await self._call_executor_generate(executor_data)
            
            if not executor_result.get("code"):
                data["agent_outcome"] = self._make_agent_finish(
                    "❌ 代码生成失败：无法生成有效代码"
                )
                return data
            
            data["current_code"] = executor_result["code"]
            data["current_iteration"] = 1
            
            logger.info("✅ 初始代码生成完成")
            return data
            
        except Exception as e:
            logger.error(f"代码生成失败: {str(e)}")
            data["agent_outcome"] = self._make_agent_finish(
                f"❌ 代码生成异常: {str(e)}"
            )
            return data
    
    @node
    async def execute_and_test(self, data):
        """执行代码并测试"""
        try:
            logger.info(f"🧪 执行代码测试 (第 {data.get('current_iteration', 0)} 次迭代)...")
            
            current_code = data.get("current_code", "")
            
            # 调用 ExecutorSubAgent 执行代码
            executor_data = {
                "code": current_code,
                "requirement": data.get("requirement", "")
            }
            
            execution_result = await self._call_executor_execute(executor_data)
            
            logger.info(f"执行结果调试: {execution_result}")
            
            data["execution_output"] = execution_result.get("output", "")
            data["execution_error"] = execution_result.get("error", "")
            data["error_traceback"] = execution_result.get("traceback", "")
            data["execution_successful"] = execution_result.get("success", False)
            
            logger.info(f"执行成功: {data['execution_successful']}")
            
            # 判断下一步
            if data["execution_successful"]:
                data["workflow_condition"] = "success"
                logger.info("✅ 代码执行成功!")
            elif data["current_iteration"] >= data["max_iterations"]:
                data["workflow_condition"] = "max_iterations"
                logger.warning(f"⚠️ 达到最大迭代次数: {data['max_iterations']}")
            else:
                data["workflow_condition"] = "error"
                logger.info(f"❌ 代码执行失败，准备调试...")
            
            return data
            
        except Exception as e:
            logger.error(f"代码执行测试失败: {str(e)}")
            data["workflow_condition"] = "error"
            return data
    
    @node
    @edge(target_node="execute_and_test")
    async def debug_and_fix(self, data):
        """调试并修复代码"""
        try:
            logger.info("🔧 开始调试和修复代码...")
            
            # 记录调试历史
            debug_entry = {
                "iteration": data.get("current_iteration", 0),
                "original_code": data.get("current_code", ""),
                "error": data.get("execution_error", ""),
                "traceback": data.get("error_traceback", "")
            }
            
            # 调用 DebuggerSubAgent 分析和修复
            debugger_data = {
                "original_code": data.get("current_code", ""),
                "error_info": data.get("execution_error", ""),
                "stderr": data.get("execution_error", ""),
                "traceback": data.get("error_traceback", ""),
                "requirement": data.get("requirement", "")
            }
            
            debug_result = await self._call_debugger_fix(debugger_data)
            
            if not debug_result.get("fixed_code"):
                data["agent_outcome"] = self._make_agent_finish(
                    f"❌ 调试失败：无法生成修复代码 (第 {data.get('current_iteration', 0)} 次迭代)"
                )
                return data
            
            # 更新代码和状态
            debug_entry["fixed_code"] = debug_result["fixed_code"]
            debug_entry["fix_explanation"] = debug_result.get("fix_explanation", "")
            
            debug_history = data.get("debug_history", [])
            debug_history.append(debug_entry)
            
            data["debug_history"] = debug_history
            data["current_code"] = debug_result["fixed_code"]
            data["current_iteration"] = data.get("current_iteration", 0) + 1
            
            logger.info(f"🔧 代码修复完成 (第 {data['current_iteration']} 次迭代)")
            
            # 修复后重新设置为执行状态，让它重新执行测试
            if "workflow_condition" in data:
                del data["workflow_condition"]
            if "execution_successful" in data:
                del data["execution_successful"]
            if "execution_error" in data:
                del data["execution_error"]
                
            return data
            
        except Exception as e:
            logger.error(f"代码调试修复失败: {str(e)}")
            data["agent_outcome"] = self._make_agent_finish(
                f"❌ 调试修复异常: {str(e)}"
            )
            return data
    
    @node
    @edge(target_node="__end__")
    async def check_success(self, data):
        """检查成功并生成最终结果"""
        current_iteration = data.get("current_iteration", 0)
        current_code = data.get("current_code", "")
        execution_output = data.get("execution_output", "")
        debug_history = data.get("debug_history", [])
        
        result = f"""🎉 代码工作流执行成功！

📊 执行统计：
- 总迭代次数：{current_iteration}
- 调试修复次数：{len(debug_history)}

✅ 最终代码：
```python
{current_code}
```

📋 执行结果：
{execution_output}

🔧 调试历史：
{self._format_debug_history(debug_history)}

💡 工作流程：
1. 根据需求生成初始代码
2. 执行代码并测试结果
3. 如有错误，自动调试修复
4. 重复步骤2-3直到成功或达到最大迭代次数

✨ 代码已通过自动化测试验证，可以直接使用！"""
        
        data["workflow_result"] = result
        data["agent_outcome"] = self._make_agent_finish(result)
        return data
    
    @node
    @edge(target_node="__end__")
    async def max_iterations_reached(self, data):
        """达到最大迭代次数"""
        current_iteration = data.get("current_iteration", 0)
        max_iterations = data.get("max_iterations", 5)
        current_code = data.get("current_code", "")
        execution_error = data.get("execution_error", "")
        debug_history = data.get("debug_history", [])
        
        result = f"""⚠️ 代码工作流达到最大迭代次数

📊 执行统计：
- 迭代次数：{current_iteration}/{max_iterations}
- 调试修复次数：{len(debug_history)}

❌ 最后的错误：
{execution_error}

🔧 当前代码：
```python
{current_code}
```

🔧 调试历史：
{self._format_debug_history(debug_history)}

💭 建议：
1. 检查需求描述是否过于复杂
2. 手动检查最后的代码和错误信息
3. 考虑简化需求或增加迭代次数
4. 联系开发者进行人工干预

虽然未能完全自动修复，但提供了详细的调试过程和最新代码。"""
        
        data["workflow_result"] = result
        data["agent_outcome"] = self._make_agent_finish(result)
        return data
    
    async def _call_executor_generate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """调用真实的 ExecutorSubAgent 生成代码"""
        try:
            # 构建 ExecutorSubAgent 的输入状态
            executor_data = {
                "requirement": data.get("requirement", ""),
                "additional_info": data.get("additional_info", "")
            }
            
            # 调用真实的 ExecutorSubAgent
            result_data = await self.executor.entry_point(executor_data.copy())
            result_data = await self.executor.generate_code(result_data)
            
            if "agent_outcome" in result_data:
                # 如果有最终结果，说明生成失败
                return {"code": "", "success": False, "error": "代码生成失败"}
            
            # 提取生成的代码
            generated_code = result_data.get("generated_code", "")
            
            return {
                "code": generated_code,
                "success": True if generated_code else False,
                "error": "" if generated_code else "未生成有效代码"
            }
            
        except Exception as e:
            logger.error(f"ExecutorSubAgent generate failed: {e}")
            return {"code": "", "success": False, "error": str(e)}
    
    async def _call_executor_execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """调用真实的 ExecutorSubAgent 执行代码"""
        try:
            # 构建 ExecutorSubAgent 的执行输入
            executor_data = {
                "generated_code": data.get("code", ""),
                "requirement": data.get("requirement", "")
            }
            
            # 调用真实的 ExecutorSubAgent 执行代码
            result_data = await self.executor.execute_code(executor_data.copy())
            
            # 提取执行结果 - 使用正确的字段名
            execution_result = result_data.get("execution_result", {})
            execution_successful = execution_result.get("success", False)
            execution_output = execution_result.get("stdout", "")
            execution_error = execution_result.get("error", "")
            error_traceback = execution_result.get("stderr", "")
            
            return {
                "success": execution_successful,
                "output": execution_output,
                "error": execution_error,
                "traceback": error_traceback
            }
            
        except Exception as e:
            logger.error(f"ExecutorSubAgent execute failed: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "traceback": f"Exception in execution: {e}"
            }
    
    async def _call_debugger_fix(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """调用真实的 DebuggerSubAgent 修复代码"""
        try:
            # 构建 DebuggerSubAgent 的输入状态
            debugger_data = {
                "original_code": data.get("original_code", ""),
                "error_info": data.get("error_info", ""),
                "stderr": data.get("stderr", ""),
                "traceback": data.get("traceback", ""),
                "requirement": data.get("requirement", "")
            }
            
            # 调用真实的 DebuggerSubAgent
            result_data = await self.debugger.entry_point(debugger_data.copy())
            result_data = await self.debugger.analyze_error(result_data)
            result_data = await self.debugger.generate_fix(result_data)
            result_data = await self.debugger.format_result(result_data)
            
            # 提取修复结果
            fixed_code = result_data.get("fixed_code", "")
            fix_explanation = result_data.get("fix_explanation", "")
            
            return {
                "fixed_code": fixed_code,
                "fix_explanation": fix_explanation,
                "success": True if fixed_code else False,
                "error": "" if fixed_code else "调试修复失败"
            }
            
        except Exception as e:
            logger.error(f"DebuggerSubAgent fix failed: {e}")
            return {"fixed_code": "", "success": False, "error": str(e)}
    
    def _format_debug_history(self, debug_history: List[Dict[str, Any]]) -> str:
        """格式化调试历史"""
        if not debug_history:
            return "无调试历史"
        
        formatted = ""
        for i, entry in enumerate(debug_history, 1):
            formatted += f"\n🔄 第 {entry.get('iteration', i)} 次调试：\n"
            formatted += f"   错误：{entry.get('error', '未知错误')}\n"
            formatted += f"   修复说明：{entry.get('fix_explanation', '无说明')}\n"
        
        return formatted


# CodeWorkflowSubAgent 已定义完成
