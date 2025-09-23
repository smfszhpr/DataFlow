#!/usr/bin/env python3
"""
CodeWorkflow SubAgent 实用工具
输入需求，自动生成代码并保存到指定路径
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径到 sys.path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.agent_v2.subagents.code_workflow_subagent import CodeWorkflowSubAgent


def save_generated_code(code: str, requirement: str, success: bool = True, execution_output: str = ""):
    """保存生成的代码到指定路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_code_{timestamp}.py"
    filepath = Path("/Users/zyd/zyd/zydtest") / filename
    
    # 创建目录
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建完整的代码文件
    header = f'''"""
自动生成的代码
需求: {requirement}
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
状态: {"✅ 成功" if success else "⚠️ 未完成"}
"""

'''
    
    if execution_output:
        header += f'''"""
执行输出:
{execution_output[:500]}{"..." if len(execution_output) > 500 else ""}
"""

'''
    
    full_code = header + code
    
    try:
        filepath.write_text(full_code, encoding='utf-8')
        print(f"💾 代码已保存到: {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return None


async def run_code_workflow(requirement: str, max_iterations: int = 3):
    """运行代码工作流并保存结果"""
    print(f"🎯 需求: {requirement}")
    print("=" * 60)
    
    workflow_agent = CodeWorkflowSubAgent()
    
    input_data = {
        "requirement": requirement,
        "max_iterations": max_iterations,
        "timeout_seconds": 180
    }
    
    try:
        # 调用工作流图
        graph = workflow_agent.graph()
        result = await graph.ainvoke(input_data)
        
        # 提取生成的代码和结果
        if isinstance(result, dict):
            current_code = result.get("current_code", "")
            success = result.get("execution_successful", False)
            execution_output = result.get("execution_output", "")
            iterations = result.get("current_iteration", 0)
            
            if current_code:
                saved_path = save_generated_code(current_code, requirement, success, execution_output)
                
                print(f"\n✅ 工作流完成")
                print(f"🔄 迭代次数: {iterations}/{max_iterations}")
                print(f"✅ 执行状态: {'成功' if success else '失败'}")
                
                if not success and "execution_error" in result:
                    print(f"⚠️ 最后错误: {result['execution_error'][:200]}...")
                
                return saved_path
            else:
                # 尝试从agent_outcome中提取
                if "agent_outcome" in result:
                    outcome = result["agent_outcome"]
                    if hasattr(outcome, 'return_values'):
                        output = outcome.return_values.get("output", "")
                    else:
                        output = str(outcome)
                        
                    print(f"\n📊 工作流结果: {output[:300]}...")
                    
                    # 尝试提取代码
                    if "```python" in output:
                        code_start = output.find("```python") + 9
                        code_end = output.find("```", code_start)
                        if code_end > code_start:
                            code = output[code_start:code_end].strip()
                            return save_generated_code(code, requirement, False)
                
                print(f"\n📊 结果: {str(result)[:300]}...")
        else:
            print(f"\n📊 结果: {str(result)[:300]}...")
            
    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
    return None


async def main():
    """主函数"""
    print("🤖 CodeWorkflow SubAgent 实用工具")
    print("输入代码需求，自动生成并保存代码到 /Users/zyd/zyd/zydtest\n")
    
    while True:
        requirement = input("请输入代码需求 (输入 'quit' 退出): ").strip()
        
        if requirement.lower() in ['quit', 'exit', 'q']:
            print("� 再见!")
            break
            
        if not requirement:
            print("❌ 请输入有效需求")
            continue
            
        print()  # 空行
        await run_code_workflow(requirement)
        print("\n" + "="*60 + "\n")

async def main():
    """主函数 - 直接设置需求，不需要用户输入"""
    print("🤖 CodeWorkflow SubAgent 实用工具")
    print("自动生成并保存代码到 /Users/zyd/zyd/zydtest\n")
    
    # 直接设置需求，不需要用户输入
    requirement = "给我一个函数来使用在x=0处泰勒展开的前n项来计算sin(x)"
    max_iterations = 10
    
    print(f"📋 设定需求: {requirement}")
    print(f"🔄 最大迭代: {max_iterations}")
    print("=" * 60)
    
    await run_code_workflow(requirement, max_iterations)

if __name__ == "__main__":
    asyncio.run(main())
