#!/usr/bin/env python3
"""
快速测试实时事件推送
"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataflow.agent_v2.event_executor import create_event_executor
from dataflow.agent_v2.events import PrintSink
from dataflow.agent_v2.master.agent import create_master_agent

async def test_real_time_events():
    """测试实时事件推送"""
    print("🎭 开始测试实时事件推送...")
    
    # 创建Master Agent和事件执行器
    master_agent, _ = create_master_agent()
    executor = create_event_executor(master_agent)
    
    # 创建带时间戳的打印接收器
    class TimestampPrintSink(PrintSink):
        async def emit(self, event):
            from datetime import datetime
            now = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # 精确到毫秒
            print(f"🎭 [{now}] {event.type.value}: {event.data}")
    
    sink = TimestampPrintSink()
    
    # 测试API密钥获取（应该看到实时事件流）
    print("\n📡 测试1: API密钥获取（观察实时事件流）")
    result = await executor.run_with_events(
        user_input="我需要API密钥123",
        session_id="test_realtime_001",
        sink=sink
    )
    
    print(f"\n✅ 测试完成，最终结果: {result['output'][:100]}...")
    
    # 测试多次请求
    print("\n📡 测试2: 多次快速请求")
    for i in range(3):
        print(f"\n--- 第{i+1}次请求 ---")
        result = await executor.run_with_events(
            user_input=f"获取第{i+1}个API密钥",
            session_id=f"test_multi_{i+1}",
            sink=sink
        )

if __name__ == "__main__":
    asyncio.run(test_real_time_events())
