# DataFlow Agent V2 系统架构文档

## 📋 项目概述

DataFlow Agent V2 是基于 LangGraph 和 MyScaleKB-Agent 架构的智能代理系统，提供了模块化的代理架构、事件驱动的执行引擎和实时WebSocket通信能力。

### 🎯 核心特性

- **混合架构设计**: 结合 MyScaleKB-Agent 的成熟架构与自研的事件系统
- **事件驱动**: 完整的事件系统支持实时状态推送和监控
- **模块化工具系统**: 插件式工具注册和管理
- **WebSocket实时通信**: 前后端实时状态同步
- **多子代理支持**: 支持不同类型的专业化子代理

## 🏗️ 系统架构

### 架构层次

```
┌─────────────────────────────────────────────────┐
│                前端 WebUI                        │
├─────────────────────────────────────────────────┤
│          WebSocket 通信层                        │
├─────────────────────────────────────────────────┤
│            Master Agent                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   工具系统   │ │   事件系统   │ │   路由系统   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────┤
│                子代理层                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │Former Agent │ │Code Agent  │ │Pipeline     │ │
│  │(表单生成)   │ │(代码生成)   │ │Agent       │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────┤
│              基础设施层                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  LLM客户端  │ │   配置管理   │ │   日志系统   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────┘
```

### 目录结构说明

```
agent_v2/
├── README.md              # 本文档
├── config.py/config.yaml  # 配置管理
├── llm_client.py          # LLM客户端统一接口
├── start_server.py        # WebSocket服务器启动入口
├── tool_result.py         # 工具结果处理
│
├── master/                # 🔥 主代理模块 (核心)
│   ├── agent.py          # MasterAgent主类 - 系统入口
│   ├── tools.py          # 工具管理和注册 - 核心工具系统
│   ├── tool_input_builder.py # 工具参数构建
│   ├── adapters/         # 适配器模式实现
│   ├── executor.py       # ❌ 未使用 - 早期工具执行器尝试
│   ├── llm_processor.py  # ❌ 未使用 - 早期LLM处理器尝试  
│   ├── policy.py         # ❌ 未使用 - 早期策略模块尝试
│   ├── router.py         # ❌ 未使用 - 早期路由器尝试
│   └── summarizer.py     # ❌ 未使用 - 早期总结器尝试
│
├── events/               # 🔥 事件系统 (核心)
│   ├── core.py          # 事件核心定义和EventSink抽象
│   ├── builder.py       # 事件构建器和组合模式
│   └── __init__.py      # 事件系统统一导出
│
├── websocket/           # 🔥 WebSocket通信 (核心)
│   ├── server.py        # FastAPI WebSocket服务器
│   ├── events.py        # WebSocket事件处理
│   └── __init__.py
│
├── subagents/           # 🔥 子代理集合 (核心)
│   ├── apikey_agent.py      # 🔥 API密钥代理 (使用base/core)
│   ├── former_tool.py       # 🔥 Former代理工具包装
│   ├── code_workflow_tool.py # 🔥 代码工作流工具
│   ├── pipeline_workflow_tool.py # 🔥 流水线工具
│   ├── csvtools.py          # 🔥 CSV处理工具集
│   ├── mock_tools.py        # 测试用模拟工具
│   └── [其他工具...]        # 其他专业化工具
│
├── former/              # 🔥 Former Agent V2 (核心)
│   ├── agent.py         # FormerAgentV2主类
│   ├── compat.py        # 兼容层 - 与旧版本接口兼容
│   ├── tools.py         # Former专用工具集
│   ├── prompt.py        # Former提示模板
│   └── __init__.py      # 统一导出
│
├── base/                # ❌ 基础架构模拟 (未使用)
│   └── core.py          # 模仿MyScaleKB-Agent的基础类
│                        # 仅apikey_agent.py使用，其他都用myscalekb_agent_base
│
└── common/              # ❓ 通用模块 (部分使用)
    └── [工具类和辅助函数]
```

## 🔥 核心模块详解

### 1. Master Agent (master/agent.py)

**职责**: 系统的核心控制器，负责任务规划、工具调度和状态管理

**关键特性**:
- 基于 MyScaleKB-Agent 的 SubAgent 架构
- 使用 LangGraph 实现状态转换工作流
- 集成事件系统进行实时状态推送
- 支持多轮对话和上下文管理

**核心工作流**:
```python
entry -> planner -> execute_tools -> summarize -> END
```

**关键方法**:
- `build_app()`: 构建LangGraph工作流
- `planner()`: 任务规划和工具选择  
- `execute_tools()`: 工具执行协调
- `summarize()`: 结果总结和输出

### 2. 工具系统 (master/tools.py)

**职责**: 统一工具注册、管理和LangChain适配

**核心组件**:
- `ToolsMixin`: 工具管理混入类
- `WorkflowRegistry`: 工作流配置管理
- `to_langchain_tool()`: 工具适配器

**工具注册流程**:
1. 在 `_register_tools()` 中添加工具实例
2. 通过 `to_langchain_tool()` 转换为LangChain兼容格式
3. 初始化 ToolExecutor 进行统一管理

### 3. 事件系统 (events/)

**职责**: 实时事件推送和系统监控

**核心设计**:
- `EventSink`: 抽象事件接收器
- `CompositeSink`: 组合模式支持多接收器
- `WebSocketSink`: WebSocket实时推送
- `PrintSink`: 控制台输出

**事件类型**:
- 执行生命周期: RUN_STARTED, RUN_FINISHED, RUN_ERROR
- 工具执行: TOOL_STARTED, TOOL_FINISHED, TOOL_ERROR  
- 规划决策: PLAN_STARTED, PLAN_DECISION
- 状态更新: STATE_UPDATE

### 4. Former Agent V2 (former/)

**职责**: 专业化表单生成代理

**架构特点**:
- 独立的SubAgent实现
- 兼容层支持旧版本接口
- 支持会话状态持久化和增量更新

**工作流程**:
```python
需求分析 -> 字段验证 -> XML生成 -> 完成
```

## 🔧 开发指南

### 添加新的 SubAgent

1. **创建 SubAgent 类**

```python
# dataflow/agent_v2/subagents/your_agent.py
from myscalekb_agent_base.sub_agent import SubAgent
from myscalekb_agent_base.graph_builder import GraphBuilder, node

class YourAgent(SubAgent):
    @classmethod
    def name(cls) -> str:
        return "your_agent"
    
    @classmethod  
    def description(cls) -> str:
        return "您的代理描述"
    
    def build_app(self):
        # 实现工作流逻辑
        pass
```

2. **创建工具包装器**

```python
# dataflow/agent_v2/subagents/your_tool.py
from dataflow.agent_v2.base.core import BaseTool
from pydantic import BaseModel

class YourTool(BaseTool):
    @classmethod
    def name(cls) -> str:
        return "your_tool"
    
    @classmethod
    def description(cls) -> str:
        return "工具描述"
    
    def params(self) -> type[BaseModel]:
        class YourParams(BaseModel):
            param1: str
            param2: int = 10
        return YourParams
    
    async def execute(self, param1: str, param2: int = 10):
        # 调用您的Agent
        agent = YourAgent()
        result = await agent.execute({"param1": param1, "param2": param2})
        return result
```

3. **注册到工具系统**

在 `master/tools.py` 的 `_register_tools()` 方法中添加:

```python
from dataflow.agent_v2.subagents.your_tool import YourTool

def _register_tools(self):
    self.tools = [
        # 现有工具...
        YourTool(),  # 添加您的工具
    ]
```

## 🚀 快速开始

### 安装依赖
```shell
pip install -r requirement.txt
```

### 启动WebSocket服务器
```shell
python dataflow/agent_v2/start_server.py
# 或
python -m dataflow.agent_v2.websocket.server
```

服务器将在 `http://localhost:8765` 启动，提供:
- WebSocket接口: `ws://localhost:8765/ws`
- 健康检查: `http://localhost:8765/health`
- 测试页面: `http://localhost:8765/test`

---

> 📚 更多详细信息请参考开发者文档
> 🐛 遇到问题请查看日志或提交Issue
> 🚀 欢迎贡献代码和改进建议