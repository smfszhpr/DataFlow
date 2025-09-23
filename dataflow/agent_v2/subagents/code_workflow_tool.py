#!/usr/bin/env python3
"""
CodeWorkflow Tool - 将 CodeWorkflowSubAgent 包装为可被 Master Agent 调用的工具
"""

import logging
import tempfile
import os
import subprocess
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

from dataflow.agent_v2.base.core import BaseTool

logger = logging.getLogger(__name__)


class CodeWorkflowToolParams(BaseModel):
    """CodeWorkflow Tool 参数"""
    requirement: str = Field(description="用户代码需求")
    max_iterations: Optional[int] = Field(default=5, description="最大迭代次数")
    timeout_seconds: Optional[int] = Field(default=300, description="总超时时间（秒）")
    apikey: Optional[str] = Field(default=None, description="编程模型API密钥（可选，暂不使用）")
    url: Optional[str] = Field(default=None, description="编程模型URL（可选，暂不使用）")


class CodeWorkflowTool(BaseTool):
    """CodeWorkflow Tool - 代码生成、测试、调试循环工具"""
    
    def __init__(self):
        self._code_workflow_agent = None
    
    @property
    def code_workflow_agent(self):
        """延迟导入CodeWorkflow SubAgent以避免循环导入"""
        if self._code_workflow_agent is None:
            from dataflow.agent_v2.subagents.code_workflow_subagent import CodeWorkflowSubAgent
            from dataflow.agent_v2.llm_client import get_llm_client
            from dataflow.agent_v2.common.states import BaseAgentState
            
            # 创建模拟的参数对象，模仿Master Agent的做法
            class MockLLM:
                def __init__(self):
                    self.model = get_llm_client()
                    self.api_available = True
            
            class MockContext:
                def __init__(self):
                    self.embedding_model = None
                    self.myscale_client = None
                    self.variables = {"knowledge_scopes": []}
            
            class MockMemory:
                pass
            
            # 创建CodeWorkflow SubAgent实例
            self._code_workflow_agent = CodeWorkflowSubAgent(
                ctx=MockContext(),
                llm=MockLLM(),
                memory=MockMemory()
            )
        return self._code_workflow_agent
    
    @classmethod
    def name(cls) -> str:
        return "code_workflow_agent"
    
    @classmethod
    def description(cls) -> str:
        return """代码生成、测试、调试循环工具，自动化完成代码开发全流程。

【前置工具】：former_agent - 建议先用Former工具分析需求
【后置工具】：无 - 此工具通常作为工作流终点

功能特点：
- 根据需求自动生成代码
- 自动执行代码并测试
- 发现错误时自动调试修复
- 支持多次迭代直到成功

适用场景：
- 已通过Former工具明确的代码开发需求
- 需要完整代码解决方案的场景
- 自动化编程和调试需求"""
    
    @classmethod
    def params(cls) -> type:
        return CodeWorkflowToolParams
    
    @classmethod
    def prerequisite_tools(cls) -> List[str]:
        """前置工具列表"""
        return ["former_agent"]  # 建议先用Former工具分析需求
    
    @classmethod
    def suggested_followup_tools(cls) -> List[str]:
        """建议的后置工具列表"""
        return []  # 通常作为工作流终点
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行 CodeWorkflow 工具"""
        try:
            # 从kwargs创建参数对象
            params = CodeWorkflowToolParams(**kwargs)
            requirement = params.requirement
            max_iterations = params.max_iterations or 5
            timeout_seconds = params.timeout_seconds or 300
            apikey = params.apikey
            url = params.url
            
            logger.info(f"CodeWorkflow Tool 开始处理: {requirement}")
            
            # 准备初始状态 - 使用适合真实 CodeWorkflowSubAgent 的格式
            initial_state = {
                "input": requirement,
                "query": requirement,
                "requirement": requirement,
                "max_iterations": max_iterations,
                "timeout_seconds": timeout_seconds,
                "apikey": apikey,
                "url": url,
                "current_iteration": 0,
                "chat_history": [],
                "agent_metadata": {},
                "intermediate_steps": [],
                "trace_id": None
            }
            
            # 构建并运行图
            try:
                app = self.code_workflow_agent.build_app()
                
                # 运行CodeWorkflow工作流
                result = await app.ainvoke(initial_state)
                
                # 提取结果
                agent_outcome = result.get("agent_outcome")
                if hasattr(agent_outcome, 'return_values'):
                    output = agent_outcome.return_values.get("output", "代码工作流完成")
                else:
                    output = str(agent_outcome) if agent_outcome else "代码工作流完成"
                
                return {
                    "success": True,
                    "output": output,
                    "current_code": result.get("current_code", ""),
                    "execution_successful": result.get("execution_successful", False),
                    "execution_output": result.get("execution_output", ""),
                    "execution_error": result.get("execution_error", ""),
                    "current_iteration": result.get("current_iteration", 0),
                    "workflow_result": result.get("workflow_result", "")
                }
                
            except Exception as graph_error:
                logger.warning(f"CodeWorkflow graph execution failed: {graph_error}")
                
                # 如果图构建失败，使用简化的后备实现
                return await self._fallback_code_workflow(requirement, max_iterations)
            
        except Exception as e:
            logger.error(f"CodeWorkflow Tool 执行失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "output": f"代码工作流执行失败: {str(e)}"
            }
    
    async def _fallback_code_workflow(self, requirement: str, max_iterations: int) -> Dict[str, Any]:
        """简化的代码工作流后备实现"""
        try:
            logger.info("使用后备代码工作流实现")
            
            from dataflow.agent_v2.llm_client import get_llm_client
            import tempfile
            import subprocess
            
            llm = get_llm_client()
            current_code = ""
            execution_history = []
            
            for iteration in range(max_iterations):
                logger.info(f"🔄 第 {iteration + 1} 次迭代...")
                
                # 生成或修复代码
                if iteration == 0:
                    # 第一次生成
                    prompt = f"""根据需求生成Python代码：

需求: {requirement}

请生成一个完整的Python程序，包含：
1. 必要的导入语句
2. 主要功能实现  
3. 主函数或示例调用
4. 适当的错误处理

**重要要求：**
- 代码必须是非交互式的，不能使用input()函数
- 如果需要输入参数，使用硬编码的示例值或命令行参数
- 代码应该能够直接运行并产生输出
- 添加示例调用来演示功能

只返回Python代码，不要任何解释或markdown格式。"""
                    
                    response = llm.call_llm("", prompt)
                    current_code = response.get('content', '').strip()
                    
                    # 清理代码格式
                    if current_code.startswith("```python"):
                        current_code = current_code[9:]
                    if current_code.endswith("```"):
                        current_code = current_code[:-3]
                    current_code = current_code.strip()
                    
                    logger.info("✨ 代码生成完成")
                else:
                    # 修复代码
                    last_exec = execution_history[-1] if execution_history else {}
                    error_info = last_exec.get('stderr', '')
                    
                    prompt = f"""根据错误信息修复代码：

原始需求: {requirement}

当前代码:
```python
{current_code}
```

错误信息:
{error_info}

**修复要求：**
- 修复代码中的错误，确保代码能正确运行
- 代码必须是非交互式的，不能使用input()函数
- 如果需要输入参数，使用硬编码的示例值或命令行参数
- 代码应该能够直接运行并产生输出

只返回修复后的完整Python代码，不要任何解释。"""
                    
                    response = llm.call_llm("", prompt)
                    current_code = response.get('content', '').strip()
                    
                    # 清理代码格式
                    if current_code.startswith("```python"):
                        current_code = current_code[9:]
                    if current_code.endswith("```"):
                        current_code = current_code[:-3]
                    current_code = current_code.strip()
                    
                    logger.info("🔧 代码修复完成")
                
                # 执行代码
                exec_result = self._execute_code_safely(current_code)
                execution_history.append(exec_result)
                
                logger.info(f"🧪 代码执行结果: {'成功' if exec_result['success'] else '失败'}")
                
                if exec_result['success']:
                    # 执行成功，结束循环
                    output_lines = []
                    output_lines.append(f"✅ **代码工作流执行成功** (第 {iteration + 1} 次迭代)")
                    output_lines.append(f"\n📋 **用户需求:** {requirement}")
                    output_lines.append(f"\n🎯 **执行结果:**")
                    if exec_result['stdout']:
                        output_lines.append(f"```\n{exec_result['stdout']}\n```")
                    
                    output_lines.append(f"\n💻 **生成的代码:**")
                    output_lines.append(f"```python\n{current_code}\n```")
                    
                    return {
                        "success": True,
                        "output": "\n".join(output_lines),
                        "current_code": current_code,
                        "execution_successful": True,
                        "execution_output": exec_result['stdout'],
                        "execution_error": "",
                        "current_iteration": iteration + 1,
                        "workflow_result": "代码生成并执行成功"
                    }
                
                # 执行失败，如果还有迭代次数就继续
                if iteration < max_iterations - 1:
                    logger.info(f"❌ 代码执行失败，准备第 {iteration + 2} 次迭代...")
                    logger.info(f"错误信息: {exec_result['stderr']}")
            
            # 所有迭代都失败了
            last_exec = execution_history[-1] if execution_history else {}
            
            output_lines = []
            output_lines.append(f"❌ **代码工作流执行失败** (已尝试 {max_iterations} 次)")
            output_lines.append(f"\n📋 **用户需求:** {requirement}")
            output_lines.append(f"\n💀 **最终错误:**")
            if last_exec.get('stderr'):
                output_lines.append(f"```\n{last_exec['stderr']}\n```")
            
            output_lines.append(f"\n💻 **最后生成的代码:**")
            output_lines.append(f"```python\n{current_code}\n```")
            
            return {
                "success": False,
                "output": "\n".join(output_lines),
                "current_code": current_code,
                "execution_successful": False,
                "execution_output": last_exec.get('stdout', ''),
                "execution_error": last_exec.get('stderr', ''),
                "current_iteration": max_iterations,
                "workflow_result": f"代码工作流失败，已尝试 {max_iterations} 次迭代"
            }
            
        except Exception as e:
            logger.error(f"后备代码工作流执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": f"代码工作流执行失败: {str(e)}"
            }
    
    def _execute_code_safely(self, code: str) -> Dict[str, Any]:
        """安全执行代码"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # 执行代码
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=os.path.dirname(temp_file)
                )
                
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
                
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "代码执行超时",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"代码执行异常: {str(e)}",
                "returncode": -1
            }
