#!/usr/bin/env python3
"""
Debugger SubAgent - 代码调试和修复
"""

import json
import httpx
from typing import Dict, Any, Optional, List
from .base_subagent import BaseSubAgent

class DebuggerSubAgent(BaseSubAgent):
    """
    代码调试 SubAgent
    功能：分析执行错误并修复代码
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        super().__init__("DebuggerSubAgent")
        
        self.llm_config = llm_config
        self.debug_history = []
        
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """
        分析错误并修复代码
        
        Args:
            input_data: 包含错误信息的字典
            格式: {
                "code": "原始代码",
                "error": "错误信息",
                "stderr": "标准错误输出",
                "stdout": "标准输出",
                "traceback": "错误堆栈",
                "requirement": "原始需求（可选）"
            }
            
        Returns:
            Dict[str, Any]: 修复结果
        """
        try:
            original_code = input_data.get("code", "")
            error_info = input_data.get("error", "")
            stderr = input_data.get("stderr", "")
            stdout = input_data.get("stdout", "")
            traceback_info = input_data.get("traceback", "")
            requirement = input_data.get("requirement", "")
            
            if not original_code:
                return {
                    "success": False,
                    "error": "缺少原始代码",
                    "fixed_code": None
                }
            
            if not (error_info or stderr or traceback_info):
                return {
                    "success": False,
                    "error": "缺少错误信息",
                    "fixed_code": None
                }
            
            self.log_info("开始分析错误并修复代码")
            
            # 1. 分析错误
            error_analysis = await self._analyze_error(
                original_code, error_info, stderr, traceback_info, requirement
            )
            
            if not error_analysis["success"]:
                return error_analysis
            
            # 2. 生成修复代码
            fix_result = await self._generate_fixed_code(
                original_code, error_analysis["analysis"], requirement
            )
            
            # 3. 记录调试历史
            debug_record = {
                "original_code": original_code,
                "error_info": error_info,
                "stderr": stderr,
                "traceback": traceback_info,
                "error_analysis": error_analysis["analysis"],
                "fixed_code": fix_result.get("fixed_code"),
                "fix_explanation": fix_result.get("explanation")
            }
            self.debug_history.append(debug_record)
            
            return fix_result
            
        except Exception as e:
            self.log_error(f"调试过程中发生异常: {str(e)}")
            return {
                "success": False,
                "error": f"调试异常: {str(e)}",
                "fixed_code": None
            }
    
    async def _analyze_error(self, 
                           code: str, 
                           error: str, 
                           stderr: str, 
                           traceback: str, 
                           requirement: str = "") -> Dict[str, Any]:
        """
        分析错误原因
        
        Args:
            code: 原始代码
            error: 错误信息
            stderr: 标准错误输出
            traceback: 错误堆栈
            requirement: 原始需求
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            # 构建错误分析提示词
            prompt = self._build_error_analysis_prompt(
                code, error, stderr, traceback, requirement
            )
            
            # 调用 LLM 分析错误
            response = await self._call_llm(prompt)
            
            self.log_info("错误分析完成")
            self.log_debug(f"错误分析结果:\n{response}")
            
            return {
                "success": True,
                "analysis": response
            }
            
        except Exception as e:
            self.log_error(f"错误分析失败: {str(e)}")
            return {
                "success": False,
                "error": f"错误分析异常: {str(e)}",
                "analysis": None
            }
    
    async def _generate_fixed_code(self, 
                                 original_code: str, 
                                 error_analysis: str, 
                                 requirement: str = "") -> Dict[str, Any]:
        """
        根据错误分析生成修复后的代码
        
        Args:
            original_code: 原始代码
            error_analysis: 错误分析结果
            requirement: 原始需求
            
        Returns:
            Dict[str, Any]: 修复结果
        """
        try:
            # 构建代码修复提示词
            prompt = self._build_code_fix_prompt(
                original_code, error_analysis, requirement
            )
            
            # 调用 LLM 生成修复代码
            response = await self._call_llm(prompt)
            
            # 提取修复后的代码
            fixed_code = self._extract_code_from_response(response)
            
            if not fixed_code:
                return {
                    "success": False,
                    "error": "LLM 未生成有效的修复代码",
                    "fixed_code": None
                }
            
            self.log_info("代码修复完成")
            self.log_debug(f"修复后的代码:\n{fixed_code}")
            
            return {
                "success": True,
                "fixed_code": fixed_code,
                "explanation": response,
                "original_code": original_code
            }
            
        except Exception as e:
            self.log_error(f"代码修复失败: {str(e)}")
            return {
                "success": False,
                "error": f"代码修复异常: {str(e)}",
                "fixed_code": None
            }
    
    def _build_error_analysis_prompt(self, 
                                   code: str, 
                                   error: str, 
                                   stderr: str, 
                                   traceback: str, 
                                   requirement: str = "") -> str:
        """
        构建错误分析的提示词
        """
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
        """
        构建代码修复的提示词
        """
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
        """
        从 LLM 响应中提取 Python 代码
        """
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
    
    async def _call_llm(self, prompt: str) -> str:
        """
        调用 LLM API
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_config.get('api_key', '')}"
            }
            
            payload = {
                "model": self.llm_config.get("model", "gpt-3.5-turbo"),
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    self.llm_config.get("api_url", "https://api.openai.com/v1/chat/completions"),
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                return content
                
        except Exception as e:
            raise Exception(f"LLM API 调用失败: {str(e)}")
    
    def get_debug_history(self) -> List[Dict[str, Any]]:
        """获取调试历史"""
        return self.debug_history.copy()
    
    def clear_debug_history(self):
        """清空调试历史"""
        self.debug_history = []
