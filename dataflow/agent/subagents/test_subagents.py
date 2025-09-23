#!/usr/bin/env python3
"""
测试 Executor 和 Debugger SubAgent
"""

import asyncio
import logging
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataflow.agent.subagents.executor_subagent import ExecutorSubAgent
from dataflow.agent.subagents.debugger_subagent import DebuggerSubAgent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_executor_debugger():
    """
    测试 Executor 和 Debugger 的配合工作
    """
    
    # LLM 配置（这里需要配置实际的 API）
    llm_config = {
        "model": "gpt-3.5-turbo",
        "api_key": "your-api-key-here",  # 需要配置实际的 API Key
        "api_url": "https://api.openai.com/v1/chat/completions"
    }
    
    # 创建 SubAgent 实例
    executor = ExecutorSubAgent(llm_config=llm_config, sandbox_timeout=30)
    debugger = DebuggerSubAgent(llm_config=llm_config)
    
    # 测试需求
    test_requirement = "生成一个代码来使用牛顿法来计算根号5的前5位数"
    
    print(f"🎯 测试需求: {test_requirement}")
    print("=" * 60)
    
    # 第一轮：Executor 生成并执行代码
    print("📝 第一步：Executor 生成并执行代码")
    
    executor_input = {
        "requirement": test_requirement,
        "additional_info": "请确保代码有清晰的输出，显示计算过程和最终结果"
    }
    
    executor_result = await executor.process(executor_input)
    
    print(f"✅ Executor 执行完成")
    print(f"   成功: {executor_result['success']}")
    print(f"   代码:\n{executor_result.get('code', '无')}")
    print(f"   输出: {executor_result.get('stdout', '无')}")
    
    if executor_result.get('stderr'):
        print(f"   错误输出: {executor_result['stderr']}")
    
    if executor_result.get('error'):
        print(f"   错误信息: {executor_result['error']}")
    
    # 如果需要调试，调用 Debugger
    if executor_result.get('needs_debug', False):
        print("\n🔧 第二步：Debugger 分析并修复错误")
        
        debugger_input = {
            "code": executor_result['code'],
            "error": executor_result.get('error', ''),
            "stderr": executor_result.get('stderr', ''),
            "stdout": executor_result.get('stdout', ''),
            "traceback": executor_result.get('execution_result', {}).get('traceback', ''),
            "requirement": test_requirement
        }
        
        debugger_result = await debugger.process(debugger_input)
        
        print(f"✅ Debugger 分析完成")
        print(f"   成功: {debugger_result['success']}")
        
        if debugger_result['success']:
            print(f"   修复后的代码:\n{debugger_result.get('fixed_code', '无')}")
            
            # 用修复后的代码再次执行
            print("\n🔄 第三步：用修复后的代码重新执行")
            
            retry_input = {
                "requirement": f"执行修复后的代码: {test_requirement}",
                "additional_info": "这是经过调试修复的代码"
            }
            
            # 直接执行修复后的代码
            retry_result = executor.sandbox.execute(
                debugger_result['fixed_code'], 
                "重新执行修复后的代码"
            )
            
            print(f"✅ 重新执行完成")
            print(f"   成功: {retry_result['success']}")
            print(f"   输出: {retry_result.get('stdout', '无')}")
            
            if retry_result.get('stderr'):
                print(f"   错误输出: {retry_result['stderr']}")
        else:
            print(f"   调试失败: {debugger_result.get('error', '未知错误')}")
    
    print("\n" + "=" * 60)
    print("📊 测试完成！")

async def test_simple_case():
    """
    测试一个简单的成功案例
    """
    # 模拟 LLM 配置（测试模式）
    llm_config = {
        "model": "test-model",
        "api_key": "test-key",
        "api_url": "test-url"
    }
    
    # 创建 executor（用于测试沙盒功能）
    executor = ExecutorSubAgent(llm_config=llm_config)
    
    # 测试一个简单的 Python 代码
    test_code = """
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
    
    print("🧪 测试沙盒执行功能")
    print("=" * 50)
    
    execution_result = executor.sandbox.execute(test_code, "测试牛顿法计算根号5")
    
    print(f"✅ 执行结果:")
    print(f"   成功: {execution_result['success']}")
    print(f"   输出:\n{execution_result.get('stdout', '无')}")
    
    if execution_result.get('stderr'):
        print(f"   错误输出: {execution_result['stderr']}")
    
    if execution_result.get('error'):
        print(f"   错误信息: {execution_result['error']}")

def main():
    """
    主函数
    """
    print("🚀 开始测试 Executor 和 Debugger SubAgent")
    print()
    
    # 先测试沙盒功能
    print("📋 第一部分：测试沙盒执行")
    asyncio.run(test_simple_case())
    
    print("\n" + "="*60 + "\n")
    
    # 然后测试完整流程（需要配置实际的 LLM API）
    print("📋 第二部分：测试完整的 Executor + Debugger 流程")
    print("注意：需要配置实际的 LLM API 才能完整测试")
    
    # 如果有 API 配置，可以取消注释下面这行
    # asyncio.run(test_executor_debugger())

if __name__ == "__main__":
    main()
