#!/usr/bin/env python3
"""
DataFlow WebSocket Server启动脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataflow.agent_v2.websocket_server import run_server

if __name__ == "__main__":
    print("🚀 启动DataFlow WebSocket服务器...")
    print("📡 测试页面: http://127.0.0.1:8000")
    print("🔌 WebSocket端点: ws://127.0.0.1:8000/ws/agent/{session_id}")
    print("🏥 健康检查: http://127.0.0.1:8000/api/health")
    
    run_server(
        host="127.0.0.1", 
        port=8000, 
        reload=True
    )
