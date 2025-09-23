#!/usr/bin/env python3
"""
Debugger SubAgent - 代码调试和修复
基于 agent_v2 架构实现
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field

from dataflow.agent_v2.base.core import SubAgent, node, edge, entry
from dataflow.agent_v2.common.states import BaseAgentState
from dataflow.agent_v2.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class DebuggerParams(BaseModel):
    """Debugger 参数模型"""
    code: str = Field(description="需要调试的代码")
    error: str = Field(description="错误信息")
    stderr: Optional[str] = Field(default="", description="标准错误输出")
    stdout: Optional[str] = Field(default="", description="标准输出")
    traceback: Optional[str] = Field(default="", description="错误堆栈")
    requirement: Optional[str] = Field(default="", description="原始需求")


class DebuggerState(BaseAgentState):
    """Debugger 状态定义"""
    original_code: str = ""
    error_info: str = ""
    stderr: str = ""
    stdout: str = ""
    traceback: str = ""
    requirement: str = ""
    error_analysis: Optional[str] = None
    fixed_code: Optional[str] = None
    fix_explanation: Optional[str] = None


class DebuggerSubAgent(SubAgent):
    """代码调试 SubAgent"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = self.llm or get_llm_client()
    
    @classmethod
    def name(cls) -> str:
        return "debugger_sub_agent"
    
    @classmethod
    def description(cls) -> str:
        return "Analyze execution errors and fix Python code"
    
    @classmethod
    def state_definition(cls) -> type:
        return DebuggerState
    
    @classmethod
    def forward_schema(cls) -> type:
        return DebuggerParams
    
    @node
    @entry
    @edge(target_node="analyze_error")
    async def entry_point(self, data):
        """入口点：接收错误信息"""
        logger.info("Debugger 开始分析错误")
        
        original_code = data.get("original_code", "")
        error_info = data.get("error_info", "")
        stderr = data.get("stderr", "")
        traceback_info = data.get("traceback", "")
        
        if not original_code:
            data["agent_outcome"] = self._make_agent_finish(
                "错误：缺少原始代码"
            )
            return data
        
        if not (error_info or stderr or traceback_info):
            data["agent_outcome"] = self._make_agent_finish(
                "错误：缺少错误信息"
            )
            return data
        
        return data
    
    @node
    @edge(target_node="generate_fix")
    async def analyze_error(self, data):
        """分析错误原因"""
        try:
            logger.info("开始分析错误")
            
            original_code = data.get("original_code", "")
            error_info = data.get("error_info", "")
            stderr = data.get("stderr", "")
            traceback_info = data.get("traceback", "")
            requirement = data.get("requirement", "")
            
            # 构建错误分析提示词
            prompt = self._build_error_analysis_prompt(
                original_code, error_info, stderr, traceback_info, requirement
            )
            
            # 调用 LLM 分析错误
            if self.llm and hasattr(self.llm, 'api_available') and self.llm.api_available:
                # 使用现有的 LLM 客户端接口
                try:
                    response = await self.llm.call_llm_async(
                        system_prompt="你是一个专业的代码调试助手，擅长分析代码错误。",
                        user_prompt=prompt
                    )
                    analysis = response.get("content", "无法分析错误")
                except Exception as e:
                    logger.error(f"LLM 调用失败: {str(e)}")
                    analysis = self._generate_test_analysis(error_info)
            else:
                # 测试模式：简单分析
                analysis = self._generate_test_analysis(error_info)
            
            data["error_analysis"] = analysis
            logger.info("错误分析完成")
            
            return data
            
        except Exception as e:
            logger.error(f"错误分析失败: {str(e)}")
            data["agent_outcome"] = self._make_agent_finish(
                f"错误分析异常: {str(e)}"
            )
            return data
    
    @node
    @edge(target_node="format_result")
    async def generate_fix(self, data):
        """根据错误分析生成修复代码"""
        try:
            logger.info("开始生成修复代码")
            
            original_code = data.get("original_code", "")
            error_analysis = data.get("error_analysis", "")
            requirement = data.get("requirement", "")
            
            # 构建代码修复提示词
            prompt = self._build_code_fix_prompt(
                original_code, error_analysis, requirement
            )
            
            # 调用 LLM 生成修复代码
            if self.llm and hasattr(self.llm, 'api_available') and self.llm.api_available:
                try:
                    response = await self.llm.call_llm_async(
                        system_prompt="你是一个专业的代码修复助手，能够根据错误分析生成修复后的代码。",
                        user_prompt=prompt
                    )
                    fix_response = response.get("content", "无法生成修复代码")
                except Exception as e:
                    logger.error(f"LLM 调用失败: {str(e)}")
                    error_info = data.get("error_info", "")
                    fix_response = self._generate_test_fix(original_code, error_info)
            else:
                # 测试模式：生成修复代码
                error_info = data.get("error_info", "")
                fix_response = self._generate_test_fix(original_code, error_info)
                fix_response = self._generate_test_fix(original_code, error_info)
            
            # 提取修复后的代码
            fixed_code = self._extract_code_from_response(fix_response)
            
            if not fixed_code:
                data["agent_outcome"] = self._make_agent_finish(
                    "错误：LLM 未生成有效的修复代码"
                )
                return data
            
            data["fixed_code"] = fixed_code
            data["fix_explanation"] = fix_response
            logger.info("代码修复完成")
            
            return data
            
        except Exception as e:
            logger.error(f"代码修复失败: {str(e)}")
            data["agent_outcome"] = self._make_agent_finish(
                f"代码修复异常: {str(e)}"
            )
            return data
    
    @node
    async def format_result(self, data):
        """格式化调试结果"""
        original_code = data.get("original_code", "")
        error_analysis = data.get("error_analysis", "")
        fixed_code = data.get("fixed_code", "")
        fix_explanation = data.get("fix_explanation", "")
        
        output = f"""🔧 代码调试分析完成

原始代码：
```python
{original_code}
```

错误分析：
{error_analysis}

修复后的代码：
```python
{fixed_code}
```

修复说明：
{fix_explanation}

建议：将修复后的代码重新提交给 Executor 执行验证。"""
        
        data["agent_outcome"] = self._make_agent_finish(output)
        return data
    
    def _build_error_analysis_prompt(self, 
                                   code: str, 
                                   error: str, 
                                   stderr: str, 
                                   traceback: str, 
                                   requirement: str = "") -> str:
        """构建错误分析的提示词"""
        prompt = f"""你是一个专业的 Python 代码调试专家。请分析以下代码的执行错误。

原始代码：
```python
{code}
```

{f"用户需求：{requirement}" if requirement else ""}

错误信息：
{error if error else "无"}

标准错误输出：
{stderr if stderr else "无"}

错误堆栈：
{traceback if traceback else "无"}

请分析：
1. 错误的根本原因是什么？
2. 错误发生在代码的哪一部分？
3. 可能的解决方案有哪些？
4. 需要注意的特殊情况

请提供详细的分析：
"""
        return prompt
    
    def _build_code_fix_prompt(self, 
                             original_code: str, 
                             error_analysis: str, 
                             requirement: str = "") -> str:
        """构建代码修复的提示词"""
        prompt = f"""你是一个专业的 Python 代码调试专家。请根据错误分析修复以下代码。

原始代码：
```python
{original_code}
```

{f"用户需求：{requirement}" if requirement else ""}

错误分析：
{error_analysis}

请修复代码，要求：
1. 保持原始功能不变
2. 修复导致错误的问题
3. 添加必要的错误处理
4. 确保代码的健壮性
5. 保持代码的可读性

请直接返回修复后的完整 Python 代码，用 ```python 和 ``` 包围：

```python
# 修复后的代码
```

修复说明：
（请解释你做了哪些修改）
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
    
    def _generate_test_analysis(self, error_info: str) -> str:
        """生成测试错误分析（用于没有 LLM 的情况）"""
        return f"""错误分析（测试模式）：

1. 错误类型：{error_info}
2. 可能原因：语法错误、导入错误或逻辑错误
3. 建议解决方案：
   - 检查语法
   - 确认导入模块
   - 验证变量和函数定义
   - 添加异常处理
"""
    
    def _generate_test_fix(self, original_code: str, error_info: str) -> str:
        """生成测试修复代码（用于没有 LLM 的情况）"""
        # 简单的修复示例
        fixed_code = original_code
        
        # 如果是导入错误，尝试添加基本导入
        if "import" in error_info.lower() or "module" in error_info.lower():
            if "import math" not in fixed_code:
                fixed_code = "import math\n" + fixed_code
        
        # 如果是语法错误，尝试基本修复
        if "syntax" in error_info.lower():
            # 简单的语法修复
            fixed_code = fixed_code.replace("print ", "print(").replace(")", ")")
        
        return f"""```python
{fixed_code}
```

修复说明（测试模式）：
- 添加了必要的导入语句
- 修复了基本的语法错误
- 建议手动检查代码逻辑
"""


# DebuggerSubAgent 已定义完成
