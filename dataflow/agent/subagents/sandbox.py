#!/usr/bin/env python3
"""
沙盒执行工具
使用 multiprocessing 进行简单的进程隔离
"""

import os
import sys
import tempfile
import multiprocessing
import traceback
import time
from typing import Dict, Any, Tuple
from pathlib import Path

def execute_code_in_sandbox(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    在沙盒环境中执行 Python 代码
    
    Args:
        code: 要执行的 Python 代码
        timeout: 执行超时时间（秒）
        
    Returns:
        Dict[str, Any]: 执行结果
    """
    
    def _run_code(code_str: str, result_queue):
        """在子进程中执行代码"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code_str)
                temp_file = f.name
                
            # 重定向标准输出和错误输出
            import io
            import contextlib
            
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            # 执行代码
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                
                # 使用 exec 执行代码
                exec_globals = {"__name__": "__main__"}
                exec(code_str, exec_globals)
            
            # 获取输出
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            result = {
                "success": True,
                "stdout": stdout_output,
                "stderr": stderr_output,
                "error": None,
                "temp_file": temp_file
            }
            
            result_queue.put(result)
            
        except Exception as e:
            error_info = traceback.format_exc()
            result = {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": str(e),
                "traceback": error_info,
                "temp_file": temp_file if 'temp_file' in locals() else None
            }
            result_queue.put(result)
        
        finally:
            # 清理临时文件
            if 'temp_file' in locals() and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    # 创建进程和队列
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_run_code, args=(code, result_queue))
    
    try:
        # 启动进程
        process.start()
        
        # 等待结果或超时
        process.join(timeout=timeout)
        
        if process.is_alive():
            # 超时，强制终止进程
            process.terminate()
            process.join(timeout=5)
            
            if process.is_alive():
                process.kill()
                process.join()
                
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"执行超时（{timeout}秒）",
                "traceback": "",
                "temp_file": None
            }
        
        # 获取结果
        if not result_queue.empty():
            result = result_queue.get()
            return result
        else:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": "进程异常退出，无返回结果",
                "traceback": "",
                "temp_file": None
            }
            
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": f"沙盒执行异常: {str(e)}",
            "traceback": traceback.format_exc(),
            "temp_file": None
        }
    
    finally:
        # 确保进程被清理
        if process.is_alive():
            process.terminate()
            process.join()


class CodeSandbox:
    """
    代码沙盒执行器
    """
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.execution_history = []
        
    def execute(self, code: str, description: str = "") -> Dict[str, Any]:
        """
        执行代码并记录历史
        
        Args:
            code: Python 代码
            description: 执行描述
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        start_time = time.time()
        
        result = execute_code_in_sandbox(code, self.timeout)
        
        execution_time = time.time() - start_time
        
        # 记录执行历史
        history_item = {
            "timestamp": time.time(),
            "description": description,
            "code": code,
            "result": result,
            "execution_time": execution_time
        }
        
        self.execution_history.append(history_item)
        
        return result
    
    def get_history(self) -> list:
        """获取执行历史"""
        return self.execution_history.copy()
    
    def clear_history(self):
        """清空执行历史"""
        self.execution_history = []
        
    def is_execution_successful(self, result: Dict[str, Any]) -> bool:
        """
        判断执行是否成功
        成功标准：没有异常且有输出（stdout 不为空或者没有错误）
        """
        if not result["success"]:
            return False
            
        # 有错误输出且 stderr 不是空的
        if result.get("error") or result.get("stderr", "").strip():
            return False
            
        # 检查是否有输出（允许空输出，只要没有错误就算成功）
        return True
