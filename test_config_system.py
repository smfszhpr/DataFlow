#!/usr/bin/env python3
"""
测试配置系统
验证EventEngine配置管理是否正常工作
"""

import sys
import os

# 添加DataFlow路径
sys.path.insert(0, '/Users/zyd/DataFlow')

from dataflow.agent.eventengine.config_manager import get_config, get_llm_config, get_former_config
from dataflow.agent.agentrole.former import FormerAgent
from dataflow.agent.xmlforms.models import FormRequest

def test_config_loading():
    """测试配置加载"""
    print("🔧 测试配置加载...")
    
    # 测试获取配置
    try:
        config = get_config()
        print(f"✅ 配置加载成功")
        print(f"   - LLM配置: {config.llm.api_key[:10]}...（已脱敏）")
        print(f"   - API URL: {config.llm.api_url}")
        print(f"   - 模型: {config.llm.model}")
        print(f"   - 队列大小: {config.engine.max_queue_size}")
        print(f"   - 最大重试: {config.engine.max_retry_attempts}")
        
        llm_config = get_llm_config()
        former_config = get_former_config()
        
        print(f"   - Former Agent最大历史: {former_config.max_history}")
        print(f"   - 使用LLM检测: {former_config.use_llm_detection}")
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    return True

import asyncio

def test_former_agent_with_config():
    """测试Former Agent使用配置"""
    print("\n🤖 测试Former Agent配置集成...")
    
    try:
        # 创建Former Agent
        former = FormerAgent()
        print(f"✅ Former Agent创建成功")
        print(f"   - API可用性: {former.api_available}")
        print(f"   - 会话ID: {former.session_id}")
        print(f"   - 最大历史: {former.max_history}")
        
        # 测试表单对话
        request = FormRequest(
            user_query="我想创建一个用户注册表单",
            session_id="test-session"
        )
        
        print(f"\n📝 测试表单对话...")
        # 由于process_conversation是异步方法，需要使用asyncio.run
        response = asyncio.run(former.process_conversation(request))
        print(f"✅ 对话响应成功")
        print(f"   - 响应内容长度: {len(response.agent_response)}")
        print(f"   - 是否生成XML: {response.xml_form is not None}")
        print(f"   - 表单类型: {response.form_type}")
        
        # 显示响应内容片段
        if len(response.agent_response) > 200:
            print(f"   - 响应预览: {response.agent_response[:200]}...")
        else:
            print(f"   - 完整响应: {response.agent_response}")
        
    except Exception as e:
        print(f"❌ Former Agent测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_config_update():
    """测试配置更新"""
    print("\n🔄 测试配置更新...")
    
    try:
        from dataflow.agent.eventengine.config_manager import config_manager
        
        # 获取当前配置
        current_config = get_config()
        original_debug = current_config.debug_mode
        
        # 更新配置
        config_manager.config.debug_mode = not original_debug
        
        # 验证更新
        updated_config = get_config()
        if updated_config.debug_mode != original_debug:
            print(f"✅ 配置更新成功: debug_mode从{original_debug}变为{updated_config.debug_mode}")
            
            # 恢复原始配置
            config_manager.config.debug_mode = original_debug
            print(f"✅ 配置已恢复")
        else:
            print(f"❌ 配置更新失败")
            return False
            
    except Exception as e:
        print(f"❌ 配置更新测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始测试EventEngine配置系统\n")
    
    tests = [
        ("配置加载", test_config_loading),
        ("Former Agent集成", test_former_agent_with_config),
        ("配置更新", test_config_update)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ 测试 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 显示测试结果
    print(f"\n{'='*50}")
    print("📊 测试结果汇总")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！配置系统工作正常。")
    else:
        print("⚠️  部分测试失败，请检查配置系统。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
