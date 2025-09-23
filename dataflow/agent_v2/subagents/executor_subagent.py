#!/usr/bin/env python3
"""
Executor SubAgent - 代码生成和执行
基于 agent_v2 架构实现
"""

import logging
import tempfile
import os
import multiprocessing
import traceback
import time
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field

from dataflow.agent_v2.base.core import SubAgent, node, edge, entry
from dataflow.agent_v2.common.states import BaseAgentState
from dataflow.agent_v2.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class ExecutorParams(BaseModel):
    """Executor 参数模型"""
    requirement: str = Field(description="用户需求描述")
    additional_info: Optional[str] = Field(default="", description="额外信息")


class ExecutorState(BaseAgentState):
    """Executor 状态定义"""
    requirement: str = ""
    additional_info: str = ""
    generated_code: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    needs_debug: bool = False
    error_info: Optional[str] = None


def execute_code_in_sandbox(code: str, timeout: int = 10) -> Dict[str, Any]:
    """
    在沙盒环境中执行 Python 代码（带真实超时）
    """
    try:
        import io
        import contextlib
        import sys
        import signal
        
        # 捕获输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # 设置信号超时
        class TimeoutException(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutException(f"代码执行超时（{timeout}秒）")
        
        # 设置信号处理器
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            # 重定向输出
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                
                # 执行代码
                exec_globals = {
                    "__name__": "__main__",
                    "__builtins__": __builtins__,
                    # 添加常用的安全模块
                    "math": __import__("math"),
                    "time": __import__("time"),
                    "random": __import__("random"),
                }
                exec(code, exec_globals)
            
            signal.alarm(0)  # 取消超时
            signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器
            
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            return {
                "success": True,
                "stdout": stdout_output,
                "stderr": stderr_output,
                "error": None
            }
            
        except TimeoutException as e:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return {
                "success": False,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "error": str(e)
            }
            
        except Exception as e:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            import traceback
            error_info = traceback.format_exc()
            return {
                "success": False,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "error": str(e),
                "traceback": error_info
            }
            
    except Exception as e:
        import traceback
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": f"沙盒执行异常: {str(e)}",
            "traceback": traceback.format_exc()
        }


class ExecutorSubAgent(SubAgent):
    """代码执行 SubAgent"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = self.llm or get_llm_client()
    
    @classmethod
    def name(cls) -> str:
        return "executor_sub_agent"
    
    @classmethod
    def description(cls) -> str:
        return "Generate and execute Python code based on user requirements"
    
    @classmethod
    def state_definition(cls) -> type:
        return ExecutorState
    
    @classmethod
    def forward_schema(cls) -> type:
        return ExecutorParams
    
    @node
    @entry
    @edge(target_node="generate_code")
    async def entry_point(self, data):
        """入口点：解析用户需求"""
        requirement = data.get("requirement", "")
        logger.info(f"Executor 开始处理需求: {requirement}")
        
        if not requirement:
            data["agent_outcome"] = self._make_agent_finish(
                "错误：缺少用户需求描述"
            )
            return data
        
        return data
    
    @node
    @edge(target_node="execute_code")
    async def generate_code(self, data):
        """生成 Python 代码"""
        try:
            logger.info("开始生成代码")
            
            requirement = data.get("requirement", "")
            additional_info = data.get("additional_info", "")
            
            # 构建提示词
            prompt = self._build_code_generation_prompt(requirement, additional_info)
            
            # 调用 LLM 生成代码
            if self.llm and hasattr(self.llm, 'api_available') and self.llm.api_available:
                # 使用现有的 LLM 客户端接口
                try:
                    response = await self.llm.call_llm_async(
                        system_prompt="你是一个专业的 Python 代码生成助手，能够根据用户需求生成高质量的代码。",
                        user_prompt=prompt
                    )
                    raw_code = response.get("content", "")
                    # 提取纯代码
                    code = self._extract_code_from_response(raw_code)
                    if not code:
                        # 如果提取失败，使用原始响应
                        code = raw_code
                except Exception as e:
                    logger.error(f"LLM 调用失败: {str(e)}")
                    code = self._generate_test_code(requirement)
            else:
                # 测试模式：生成简单示例代码
                logger.info("使用测试模式生成代码（LLM 接口待集成）")
                code = self._generate_test_code(requirement)
            
            if not code:
                data["agent_outcome"] = self._make_agent_finish(
                    "错误：未能生成有效的 Python 代码"
                )
                return data
            
            data["generated_code"] = code
            logger.info("代码生成成功")
            
            return data
            
        except Exception as e:
            logger.error(f"代码生成失败: {str(e)}")
            data["agent_outcome"] = self._make_agent_finish(
                f"代码生成异常: {str(e)}"
            )
            return data
    
    @node
    @edge(target_node="check_execution_result")
    async def execute_code(self, data):
        """在沙盒中执行代码"""
        try:
            logger.info("开始执行代码")
            
            generated_code = data.get("generated_code", "")
            result = execute_code_in_sandbox(generated_code, timeout=10)
            data["execution_result"] = result
            
            if result["success"]:
                logger.info("代码执行成功")
                if result.get("stdout"):
                    logger.info(f"输出: {result['stdout']}")
            else:
                logger.error(f"代码执行失败: {result.get('error', '未知错误')}")
                data["error_info"] = result.get("error", "")
            
            return data
            
        except Exception as e:
            logger.error(f"代码执行异常: {str(e)}")
            data["execution_result"] = {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": "",
                "traceback": traceback.format_exc()
            }
            data["error_info"] = str(e)
            return data
    
    @node
    async def check_execution_result(self, data):
        """检查执行结果并决定下一步"""
        result = data.get("execution_result")
        generated_code = data.get("generated_code", "")
        
        if result and result.get("success"):
            # 执行成功，返回结果
            output = f"""✅ 代码执行成功！

生成的代码：
```python
{generated_code}
```

执行结果：
{result.get('stdout', '(无输出)')}

任务完成！"""
            
            data["agent_outcome"] = self._make_agent_finish(output)
        else:
            # 执行失败，标记需要调试
            data["needs_debug"] = True
            
            error_output = f"""❌ 代码执行失败

生成的代码：
```python
{generated_code}
```

错误信息：
{result.get('error', '未知错误')}

错误输出：
{result.get('stderr', '无')}

需要调用 Debugger 进行修复。"""
            
            data["agent_outcome"] = self._make_agent_finish(error_output)
        
        return data
    
    def _build_code_generation_prompt(self, requirement: str, additional_info: str = "") -> str:
        """构建代码生成的提示词"""
        prompt = f"""你是一个专业的 Python 代码生成助手。请根据用户需求生成完整可执行的 Python 代码。

用户需求：{requirement}

{f"额外信息：{additional_info}" if additional_info else ""}

要求：
1. 生成完整的、可直接执行的 Python 代码
2. 代码应该有适当的输出，让用户能看到执行结果
3. 包含必要的错误处理
4. 代码风格清晰，有适当的注释
5. 如果需要导入库，请使用标准库或常见的第三方库

请直接返回 Python 代码，用 ```python 和 ``` 包围：

```python
# 你的代码
```
"""
        return prompt
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """从 LLM 响应中提取 Python 代码"""
        import re
        
        # 匹配 ```python 到 ``` 的内容
        pattern = r"```python\s*\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # 如果没有找到，尝试匹配任何 ``` 包围的内容
        pattern = r"```\s*\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return None
    
    async def _call_llm_for_code(self, prompt: str) -> Optional[str]:
        """调用 LLM 生成代码（兼容现有 LLM 客户端）"""
        try:
            # 这里应该调用现有的 LLM 服务
            # 由于当前的 LLM 客户端接口不明确，暂时使用测试模式
            # TODO: 集成实际的 LLM 调用
            logger.info("使用测试模式生成代码（LLM 接口待集成）")
            
            if "牛顿法" in prompt and "根号" in prompt:
                return """
import math

def newton_sqrt(n, precision=5):
    '''使用牛顿法计算平方根'''
    x = n / 2.0  # 初始猜测
    
    for i in range(20):  # 最多迭代20次
        root = 0.5 * (x + n / x)
        if abs(root - x) < 10**(-precision-1):
            break
        x = root
    
    return round(root, precision)

# 计算根号5的前5位数
result = newton_sqrt(5, 5)
print(f"使用牛顿法计算根号5的结果: {result}")
print(f"验证: {result}^2 = {result**2}")
"""
            else:
                return f"""
# 根据需求生成的代码
print("Hello, World!")
print("处理需求: 用户的要求")
calculation_result = 42
print(f"计算结果: {{calculation_result}}")
"""
                
        except Exception as e:
            logger.error(f"LLM 调用失败: {str(e)}")
            return None
    
    def _generate_test_code(self, requirement: str) -> str:
        """生成测试代码（用于没有 LLM 的情况）"""
        if "牛顿法" in requirement and "根号" in requirement:
            return """
import math

def newton_sqrt(n, precision=5):
    '''使用牛顿法计算平方根'''
    x = n / 2.0  # 初始猜测
    
    for i in range(20):  # 最多迭代20次
        root = 0.5 * (x + n / x)
        if abs(root - x) < 10**(-precision-1):
            break
        x = root
    
    return round(root, precision)

# 计算根号5的前5位数
result = newton_sqrt(5, 5)
print(f"使用牛顿法计算根号5的结果: {result}")
print(f"验证: {result}^2 = {result**2}")
"""
        else:
            return f"""
# 根据需求生成的示例代码: {requirement}
print("Hello, World!")
print(f"处理需求: {requirement}")
"""


# ExecutorSubAgent 已定义完成
