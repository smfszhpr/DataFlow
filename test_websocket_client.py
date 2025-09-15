#!/usr/bin/env python3
"""
WebSocket客户端测试 - 测试实时事件推送
"""

import asyncio
import websockets
import json
import time

async def test_websocket_client():
    """测试WebSocket客户端"""
    uri = "ws://localhost:8000/ws/agent/test_client_001"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("🔗 WebSocket连接成功")
            
            # 发送测试消息
            test_message = {
                "type": "user_input",
                "input": "我需要获取3个不同的API密钥"
            }
            
            print(f"📤 发送消息: {test_message}")
            await websocket.send(json.dumps(test_message))
            
            # 监听事件
            event_count = 0
            start_time = time.time()
            
            print("\n📡 开始监听实时事件:")
            print("-" * 60)
            
            while True:
                try:
                    # 设置超时避免无限等待
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    event = json.loads(message)
                    
                    event_count += 1
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # 格式化显示事件
                    event_type = event.get('type', 'unknown')
                    timestamp = event.get('timestamp', '')
                    data = event.get('data', {})
                    
                    print(f"[{elapsed:6.2f}s] #{event_count:2d} {event_type}")
                    
                    # 显示重要数据
                    if event_type == 'tool_started':
                        tool_name = data.get('tool_name', 'unknown')
                        print(f"    → 开始执行工具: {tool_name}")
                    
                    elif event_type == 'tool_finished':
                        tool_name = data.get('tool_name', 'unknown')
                        tool_output = data.get('tool_output', {})
                        if isinstance(tool_output, dict) and 'apikey' in tool_output:
                            print(f"    → 工具完成: {tool_name}")
                            print(f"    → 获得API密钥: {tool_output['apikey']}")
                        else:
                            print(f"    → 工具完成: {tool_name}")
                    
                    elif event_type == 'run_finished':
                        print(f"    → 执行完成")
                        print(f"    → 总事件数: {event_count}")
                        break
                    
                    elif event_type == 'run_error':
                        error = data.get('error', 'unknown error')
                        print(f"    → 执行错误: {error}")
                        break
                
                except asyncio.TimeoutError:
                    print("⏰ 接收超时，连接可能断开")
                    break
                except Exception as e:
                    print(f"❌ 接收消息错误: {e}")
                    break
            
            print("-" * 60)
            print(f"✅ 测试完成，总耗时: {time.time() - start_time:.2f}秒")
    
    except Exception as e:
        print(f"❌ WebSocket连接失败: {e}")

if __name__ == "__main__":
    print("🧪 启动WebSocket客户端测试...")
    asyncio.run(test_websocket_client())
