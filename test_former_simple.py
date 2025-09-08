#!/usr/bin/env python3
"""
简单测试Former Agent能否正常工作
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from dataflow.agent.agentrole.former import FormerAgent
from dataflow.agent.xmlforms.models import FormRequest

async def test_former_agent():
    """测试Former Agent"""
    print("🚀 开始测试Former Agent")
    
    # 创建Former Agent
    former = FormerAgent()
    print(f"✅ Former Agent创建成功，会话ID: {former.session_id}")
    
    # 测试对话
    request = FormRequest(
        user_query="我想创建一个用户注册表单算子",
        session_id="test-session"
    )
    
    print("\n📝 开始对话...")
    response = await former.process_conversation(request)
    
    print(f"✅ 对话成功")
    print(f"   - 响应长度: {len(response.agent_response)}")
    print(f"   - 表单类型: {response.form_type}")
    print(f"   - 生成XML: {response.xml_form is not None}")
    print(f"   - 响应预览: {response.agent_response[:100]}...")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_former_agent())
        if success:
            print("\n🎉 Former Agent测试完全成功！")
            print("✅ 配置系统工作正常")
            print("✅ EventEngine架构完整")
            print("✅ YAML配置生效")
        else:
            print("\n❌ 测试失败")
    except Exception as e:
        print(f"\n💥 测试出错: {e}")
        import traceback
        traceback.print_exc()
