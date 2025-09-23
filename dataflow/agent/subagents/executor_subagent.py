#!/usr/bin/env python3
"""
Executor SubAgent - 代码生成和执行
"""

import json
import httpx
from typing import Dict, Any, Optional
from .base_subagent import BaseSubAgent
from .sandbox import CodeSandbox

class ExecutorSubAgent(BaseSubAgent):
    """
    代码执行 SubAgent
    功能：根据用户需求生成 Python 代码并在沙盒中执行
    """
    
    def __init__(self, 
                 llm_config: Dict[str, Any],
                 sandbox_timeout: int = 30):
        super().__init__("ExecutorSubAgent")
        
        self.llm_config = llm_config
        self.sandbox = CodeSandbox(timeout=sandbox_timeout)
        self.generated_code = None
        self.execution_result = None
        
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """
        处理用户需求，生成并执行代码
        
        Args:
            input_data: 包含用户需求的字典
            格式: {
                "requirement": "用户需求描述",
                "additional_info": "额外信息（可选）"
            }
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        try:
            requirement = input_data.get("requirement", "")
            additional_info = input_data.get("additional_info", "")
            
            if not requirement:
                return {
                    "success": False,
                    "error": "缺少用户需求描述",
                    "code": None,
                    "execution_result": None
                }
            
            self.log_info(f"开始处理需求: {requirement}")
            
            # 1. 生成代码
            code_result = await self._generate_code(requirement, additional_info)
            if not code_result["success"]:
                return code_result
                
            self.generated_code = code_result["code"]
            
            # 2. 执行代码
            execution_result = self._execute_code(self.generated_code, requirement)
            self.execution_result = execution_result
            
            # 3. 返回结果
            return {
                "success": execution_result["success"],
                "code": self.generated_code,
                "execution_result": execution_result,
                "error": execution_result.get("error"),
                "stdout": execution_result.get("stdout", ""),
                "stderr": execution_result.get("stderr", ""),
                "needs_debug": not execution_result["success"]
            }
            
        except Exception as e:
            self.log_error(f"处理过程中发生异常: {str(e)}")
            return {
                "success": False,
                "error": f"处理异常: {str(e)}",
                "code": None,
                "execution_result": None
            }
    
    async def _generate_code(self, requirement: str, additional_info: str = "") -> Dict[str, Any]:
        """
        使用 LLM 生成 Python 代码
        
        Args:
            requirement: 用户需求
            additional_info: 额外信息
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        try:
            # 构建提示词
            prompt = self._build_code_generation_prompt(requirement, additional_info)
            
            # 调用 LLM
            response = await self._call_llm(prompt)
            
            # 提取代码
            code = self._extract_code_from_response(response)
            
            if not code:
                return {
                    "success": False,
                    "error": "LLM 未生成有效的 Python 代码",
                    "code": None
                }
            
            self.log_info("代码生成成功")
            self.log_debug(f"生成的代码:\n{code}")
            
            return {
                "success": True,
                "code": code,
                "llm_response": response
            }
            
        except Exception as e:
            self.log_error(f"代码生成失败: {str(e)}")
            return {
                "success": False,
                "error": f"代码生成异常: {str(e)}",
                "code": None
            }
    
    def _execute_code(self, code: str, description: str) -> Dict[str, Any]:
        """
        在沙盒中执行代码
        
        Args:
            code: Python 代码
            description: 执行描述
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        try:
            self.log_info("开始执行代码")
            
            result = self.sandbox.execute(code, description)
            
            if result["success"]:
                self.log_info("代码执行成功")
                if result.get("stdout"):
                    self.log_info(f"输出: {result['stdout']}")
            else:
                self.log_error(f"代码执行失败: {result.get('error', '未知错误')}")
                if result.get("stderr"):
                    self.log_error(f"错误输出: {result['stderr']}")
            
            return result
            
        except Exception as e:
            self.log_error(f"代码执行异常: {str(e)}")
            return {
                "success": False,
                "error": f"执行异常: {str(e)}",
                "stdout": "",
                "stderr": "",
                "traceback": ""
            }
    
    def _build_code_generation_prompt(self, requirement: str, additional_info: str = "") -> str:
        """
        构建代码生成的提示词
        """
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
        """
        从 LLM 响应中提取 Python 代码
        """
        # 查找 ```python 和 ``` 之间的代码
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
        
        # 如果还是没有找到，返回整个响应（可能是纯代码）
        # 但先检查是否看起来像 Python 代码
        if "def " in response or "import " in response or "print(" in response:
            return response.strip()
        
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
    
    def get_execution_history(self) -> list:
        """获取执行历史"""
        return self.sandbox.get_history()
    
    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """获取最后一次执行结果"""
        return {
            "code": self.generated_code,
            "execution_result": self.execution_result
        } if self.generated_code else None
