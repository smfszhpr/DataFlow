# DataFlow Agent V2 开发者指南

## 📚 目录

- [系统架构深度解析](#系统架构深度解析)
- [核心组件详解](#核心组件详解)
- [开发流程指南](#开发流程指南)
- [代码贡献指南](#代码贡献指南)
- [调试和测试](#调试和测试)
- [性能优化](#性能优化)
- [常见问题解决](#常见问题解决)

## 🏛️ 系统架构深度解析

### 架构设计哲学

DataFlow Agent V2 采用**混合架构**设计，结合了多种成熟的设计模式：

1. **分层架构**: 清晰的分层设计，职责分离
2. **事件驱动**: 松耦合的事件系统，支持实时响应
3. **插件化**: 模块化的工具和子代理系统
4. **适配器模式**: 兼容多种外部框架和库

### 关键设计决策

#### 为什么选择 MyScaleKB-Agent 而不是完全自研？

```python
# 当前实现 - 使用成熟框架
from myscalekb_agent_base.sub_agent import SubAgent
from myscalekb_agent_base.graph_builder import GraphBuilder

# 早期尝试 - 自研实现 (已废弃)
from ..base.core import SubAgent, GraphBuilder  # ❌ 仅apikey_agent使用
```

**原因**:
- ✅ **成熟稳定**: MyScaleKB-Agent 经过生产环境验证
- ✅ **功能完整**: 提供完整的 LangGraph 集成
- ✅ **维护成本低**: 减少重复开发和维护工作
- ✅ **生态兼容**: 与 LangChain 生态无缝集成

#### 事件驱动 vs 传统回调

```python
# 传统回调方式
def on_tool_start(tool_name):
    print(f"Tool {tool_name} started")

# 事件驱动方式 (当前实现)
await event_sink.send_event(Event(
    type=EventType.TOOL_STARTED,
    data={"tool": tool_name, "timestamp": datetime.now()}
))
```

**事件驱动的优势**:
- 🔄 **解耦**: 组件间松耦合
- 📡 **实时性**: 支持 WebSocket 实时推送
- 🔧 **可扩展**: 易于添加新的事件处理器
- 🐛 **可观测**: 完整的执行链路追踪

## 🔧 核心组件详解

### 1. Master Agent 核心实现

#### 状态管理系统

```python
class AgentState(TypedDict, total=False):
    """Master Agent 状态定义"""
    # myscalekb_agent_base 标准字段
    input: Any
    query: str
    chat_history: List[Any]
    agent_metadata: AgentMetadata
    agent_outcome: Union[Any, None]
    intermediate_steps: Annotated[List[Tuple[Any, Any]], operator.add]
    
    # DataFlow 扩展字段
    session_id: Optional[str]
    current_step: str
    form_data: Optional[Dict[str, Any]]
    generated_code: Optional[str]  # 前端代码标签页
    execution_result: Optional[str]
    conversation_history: List[Dict[str, str]]
    
    # 多轮编排支持
    pending_actions: List[Any]
    tool_results: List[Dict[str, Any]]
    context_vars: Dict[str, Any]  # 跨步共享数据
```

#### 工作流设计

```python
def build_app(self):
    """构建 LangGraph 工作流"""
    workflow = self._build_graph(AgentState, compiled=False)
    
    # 入口点设置
    workflow.set_conditional_entry_point(
        self.entry,
        {"planner": "planner"}
    )
    
    # 核心流程
    workflow.add_conditional_edges(
        "planner",
        self.planner_router,
        {
            "continue": "execute_tools",
            "finish": "summarize"
        }
    )
    
    # 工具执行后的路由决策
    workflow.add_conditional_edges(
        "execute_tools",
        self.action_forward,
        {
            "planner": "planner",     # 继续规划
            "summarize": "summarize", # 总结结果
            "end": GraphBuilder.END   # 结束流程
        }
    )
    
    return workflow.compile()
```

#### 关键节点实现

**1. 规划器节点 (Planner)**

```python
@node
async def planner(self, data: AgentState) -> AgentState:
    """智能任务规划器"""
    
    # 1. 构建对话历史上下文
    history_context = self._build_history_text(
        data.get("conversation_history", []), 
        k=8, clip=200
    )
    
    # 2. 工作流发现
    available_workflows = self.workflow_registry.get_all_workflows()
    workflow_desc = "\n".join([
        f"- {name}: {info['description']}" 
        for name, info in available_workflows.items()
    ])
    
    # 3. LLM 规划
    planning_prompt = f"""
    基于对话历史和可用工具，制定执行计划：
    
    对话历史：{history_context}
    当前输入：{data.get('query', '')}
    可用工具：{workflow_desc}
    
    请选择合适的工具并构建参数。
    """
    
    # 4. 执行规划并更新状态
    response = await self.llm.ainvoke(planning_prompt)
    # ... 解析响应并更新 agent_outcome
    
    return data
```

**2. 工具执行节点 (Execute Tools)**

```python
@node 
async def execute_tools(self, data: AgentState) -> AgentState:
    """统一工具执行器"""
    
    agent_outcome = data.get('agent_outcome')
    if not agent_outcome:
        return data
    
    # 获取工具动作
    if isinstance(agent_outcome, list):
        action = agent_outcome[0] if agent_outcome else None
    else:
        action = agent_outcome
    
    if not action or not hasattr(action, 'tool'):
        return data
    
    # 事件通知：工具开始执行
    await self.event_sink.send_event(Event(
        type=EventType.TOOL_STARTED,
        data={"tool": action.tool, "input": action.tool_input}
    ))
    
    try:
        # 执行工具
        result = await self.tool_executor.ainvoke(action)
        
        # 更新状态
        data["tool_results"] = data.get("tool_results", []) + [result]
        data["last_tool_results"] = result
        
        # 事件通知：工具执行完成
        await self.event_sink.send_event(Event(
            type=EventType.TOOL_FINISHED,
            data={"tool": action.tool, "result": result}
        ))
        
    except Exception as e:
        # 错误处理和事件通知
        await self.event_sink.send_event(Event(
            type=EventType.TOOL_ERROR,
            data={"tool": action.tool, "error": str(e)}
        ))
        
    return data
```

### 2. 工具系统详解

#### 工具抽象基类

```python
class BaseTool(ABC):
    """工具基类定义"""
    
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """工具唯一标识"""
        pass
    
    @classmethod
    @abstractmethod
    def description(cls) -> str:
        """工具功能描述，用于 LLM 选择"""
        pass
    
    @abstractmethod
    def params(self) -> Type[BaseModel]:
        """参数模型定义"""
        pass
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """异步执行方法"""
        pass
```

#### 工具注册和转换机制

```python
def to_langchain_tool(tool: BaseTool) -> StructuredTool:
    """将自定义工具转换为 LangChain 兼容格式"""
    
    ArgsSchema = tool.params()
    
    return StructuredTool.from_function(
        coroutine=tool.execute,  # 异步执行
        name=tool.name(),
        description=tool.description(),
        args_schema=ArgsSchema,
        return_direct=False,
    )

class ToolsMixin:
    """工具管理混入类"""
    
    def _register_tools(self):
        """注册所有工具"""
        self.tools = [
            APIKeyTool(),              # API密钥管理
            FormerTool(),              # 表单生成
            CodeWorkflowTool(),        # 代码工作流
            PipelineWorkflowTool(),    # 数据流水线
            CSVProfileTool(),          # CSV分析
            # ... 更多工具
        ]
        
        # 转换为 LangChain 格式
        self.lc_tools = [to_langchain_tool(tool) for tool in self.tools]
        
        # 初始化工具执行器
        self.tool_executor = ToolExecutor(self.lc_tools)
```

### 3. 事件系统架构

#### 事件抽象层

```python
class EventSink(ABC):
    """事件接收器抽象基类"""
    
    @abstractmethod
    async def send_event(self, event: Event) -> None:
        """发送事件"""
        pass

class Event(BaseModel):
    """事件数据模型"""
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    trace_id: Optional[str] = None
```

#### 组合模式实现

```python
class CompositeSink(EventSink):
    """组合事件接收器 - 支持多目标推送"""
    
    def __init__(self, sinks: List[EventSink]):
        self.sinks = sinks
    
    async def send_event(self, event: Event) -> None:
        """并发发送到所有接收器"""
        tasks = [sink.send_event(event) for sink in self.sinks]
        await asyncio.gather(*tasks, return_exceptions=True)
```

#### WebSocket 实时推送

```python
class WebSocketSink(EventSink):
    """WebSocket 事件推送器"""
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
    
    async def send_event(self, event: Event) -> None:
        """推送事件到前端"""
        try:
            await self.websocket.send_json({
                "type": event.type.value,
                "data": event.data,
                "timestamp": event.timestamp.isoformat()
            })
        except Exception as e:
            logger.error(f"WebSocket 推送失败: {e}")
```

### 4. Former Agent V2 架构

#### 状态驱动设计

```python
class FormerAgentV2(SubAgent):
    """表单生成专业代理"""
    
    def _setup_graph(self):
        """定义工作流图"""
        # 节点定义
        self.graph_builder.add_node("entry", self.entry_node, NodeType.ENTRY)
        self.graph_builder.add_node("analyze_requirement", self.analyze_requirement_node)
        self.graph_builder.add_node("validate_fields", self.validate_fields_node)
        self.graph_builder.add_node("generate_xml", self.generate_xml_node)
        
        # 流程控制
        self.graph_builder.add_edge("entry", "analyze_requirement")
        self.graph_builder.add_edge("analyze_requirement", "validate_fields")
        self.graph_builder.add_conditional_edge(
            "validate_fields",
            self._should_generate_xml,
            {
                "generate": "generate_xml",
                "incomplete": GraphBuilder.END,
                "error": GraphBuilder.END
            }
        )
```

#### 兼容层设计

```python
class FormerAgentCompat:
    """向后兼容层 - 支持旧版本接口"""
    
    def __init__(self):
        self.agent_v2 = FormerAgentV2()
        self.form_states = {}  # 会话状态持久化
    
    def _update_form_state(self, session_id: str, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """增量更新表单状态 - 解决记忆丢失问题"""
        form_state = self._get_or_create_form_state(session_id)
        
        # 保留已有的表单类型
        if new_data.get("form_type") and not form_state["form_type"]:
            form_state["form_type"] = new_data["form_type"]
        
        # 合并字段信息（保留已有字段，添加新字段）
        if "validated_fields" in new_data:
            new_fields = new_data["validated_fields"]
            form_state["fields"].update(new_fields)  # 增量更新
        
        return form_state
    
    async def process_conversation(self, form_request: FormRequest) -> FormResponse:
        """处理对话请求 - 支持会话记忆"""
        session_id = form_request.session_id or "default"
        
        # 获取历史状态
        form_state = self._get_or_create_form_state(session_id)
        
        # 执行处理（基于历史状态）
        result = self.agent_v2.process_request(
            user_requirement=form_request.user_query,
            user_input=form_request.user_query
        )
        
        # 增量更新状态
        updated_state = self._update_form_state(session_id, result)
        
        # 构建响应
        return FormResponse(
            need_more_info=not updated_state["is_complete"],
            agent_response=self._build_response_message(updated_state, result),
            xml_form=updated_state["xml_content"],
            form_type=updated_state["form_type"]
        )
```

## 🚀 开发流程指南

### 添加新的SubAgent详细步骤

#### 1. 选择架构模式

根据复杂度选择合适的架构：

**简单工具模式** (推荐用于简单功能):
```python
from dataflow.agent_v2.base.core import BaseTool

class SimpleCalculatorTool(BaseTool):
    # 实现简单的计算功能
    pass
```

**SubAgent模式** (推荐用于复杂工作流):
```python
from myscalekb_agent_base.sub_agent import SubAgent

class ComplexWorkflowAgent(SubAgent):
    # 实现复杂的多步骤工作流
    pass
```

#### 2. 实现SubAgent

```python
# dataflow/agent_v2/subagents/math_agent.py
from typing import Dict, Any, Type
from pydantic import BaseModel, Field
from myscalekb_agent_base.sub_agent import SubAgent
from myscalekb_agent_base.graph_builder import GraphBuilder, node

class MathState(BaseModel):
    """数学计算状态"""
    expression: str = ""
    result: float = 0.0
    error: str = ""
    steps: List[str] = Field(default_factory=list)

class MathAgent(SubAgent):
    """数学计算代理"""
    
    @classmethod
    def name(cls) -> str:
        return "math_calculator"
    
    @classmethod
    def description(cls) -> str:
        return "执行数学计算和表达式求值"
    
    def state_definition(self) -> Type[BaseModel]:
        return MathState
    
    def build_app(self):
        """构建计算工作流"""
        workflow = self._build_graph(MathState)
        
        # 设置节点
        workflow.add_node("parse_expression", self.parse_expression)
        workflow.add_node("calculate", self.calculate)
        workflow.add_node("format_result", self.format_result)
        
        # 设置流程
        workflow.set_entry_point("parse_expression")
        workflow.add_edge("parse_expression", "calculate")
        workflow.add_edge("calculate", "format_result")
        
        return workflow.compile()
    
    @node
    async def parse_expression(self, state: MathState) -> MathState:
        """解析数学表达式"""
        try:
            # 表达式验证和预处理
            state.steps.append(f"解析表达式: {state.expression}")
            # ... 实现解析逻辑
        except Exception as e:
            state.error = f"表达式解析错误: {e}"
        
        return state
    
    @node
    async def calculate(self, state: MathState) -> MathState:
        """执行计算"""
        if state.error:
            return state
        
        try:
            # 安全的表达式计算
            result = eval(state.expression)  # 注意：生产环境需要使用安全的eval
            state.result = float(result)
            state.steps.append(f"计算结果: {result}")
        except Exception as e:
            state.error = f"计算错误: {e}"
        
        return state
    
    @node 
    async def format_result(self, state: MathState) -> MathState:
        """格式化输出"""
        if not state.error:
            state.steps.append("计算完成")
        return state
```

#### 3. 创建工具包装器

```python
# dataflow/agent_v2/subagents/math_tool.py
from typing import Dict, Any
from pydantic import BaseModel, Field
from dataflow.agent_v2.base.core import BaseTool
from .math_agent import MathAgent, MathState

class MathToolParams(BaseModel):
    """数学工具参数"""
    expression: str = Field(description="要计算的数学表达式")
    precision: int = Field(default=2, description="结果精度")

class MathTool(BaseTool):
    """数学计算工具包装器"""
    
    def __init__(self):
        self.agent = MathAgent()
    
    @classmethod
    def name(cls) -> str:
        return "math_calculator"
    
    @classmethod
    def description(cls) -> str:
        return "执行数学计算，支持基本算术、函数运算等。输入数学表达式，返回计算结果。"
    
    def params(self) -> Type[BaseModel]:
        return MathToolParams
    
    async def execute(self, expression: str, precision: int = 2) -> Dict[str, Any]:
        """执行数学计算"""
        try:
            # 初始化状态
            initial_state = MathState(expression=expression)
            
            # 执行计算工作流
            app = self.agent.build_app()
            final_state = await app.ainvoke(initial_state.dict())
            
            # 格式化返回结果
            if final_state.get("error"):
                return {
                    "success": False,
                    "error": final_state["error"],
                    "expression": expression,
                    "result": None
                }
            else:
                result = round(final_state["result"], precision)
                return {
                    "success": True,
                    "expression": expression,
                    "result": result,
                    "steps": final_state["steps"],
                    "formatted": f"{expression} = {result}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"执行失败: {str(e)}",
                "expression": expression,
                "result": None
            }
```

#### 4. 注册到工具系统

```python
# 在 master/tools.py 中添加
from dataflow.agent_v2.subagents.math_tool import MathTool

class ToolsMixin:
    def _register_tools(self):
        self.tools = [
            # 现有工具...
            MathTool(),  # 🆕 添加数学工具
        ]
```

#### 5. 添加到工作流注册表（可选）

如果需要复杂的参数配置，添加到工作流注册表：

```python
# 在 master/tools.py 的 WorkflowRegistry 中
def _discover_available_workflows(self):
    workflows = {
        # 现有工作流...
        "math_calculator": {
            "description": "数学计算和表达式求值工具",
            "params_schema": self._extract_params_from_pydantic_model(MathToolParams),
            "tool_class": "MathTool"
        }
    }
    return workflows
```

### 事件系统集成

为你的Agent添加事件支持：

```python
class MathAgent(SubAgent):
    def __init__(self, event_sink: EventSink = None):
        super().__init__()
        self.event_sink = event_sink
    
    @node
    async def calculate(self, state: MathState) -> MathState:
        # 发送计算开始事件
        if self.event_sink:
            await self.event_sink.send_event(Event(
                type=EventType.TOOL_STARTED,
                data={"agent": "math_calculator", "expression": state.expression}
            ))
        
        try:
            # 执行计算...
            result = eval(state.expression)
            
            # 发送成功事件
            if self.event_sink:
                await self.event_sink.send_event(Event(
                    type=EventType.TOOL_FINISHED,
                    data={"agent": "math_calculator", "result": result}
                ))
                
        except Exception as e:
            # 发送错误事件
            if self.event_sink:
                await self.event_sink.send_event(Event(
                    type=EventType.TOOL_ERROR,
                    data={"agent": "math_calculator", "error": str(e)}
                ))
            raise
        
        return state
```

## 🧪 调试和测试

### 日志记录最佳实践

```python
import logging
logger = logging.getLogger(__name__)

class YourAgent(SubAgent):
    @node
    async def your_method(self, state):
        logger.info(f"开始处理: {state}")
        
        try:
            # 业务逻辑
            result = await some_operation()
            logger.debug(f"中间结果: {result}")
            
        except Exception as e:
            logger.error(f"处理失败: {e}", exc_info=True)
            raise
            
        logger.info(f"处理完成: {result}")
        return state
```

### 单元测试

```python
# tests/test_math_agent.py
import pytest
from dataflow.agent_v2.subagents.math_agent import MathAgent, MathState

class TestMathAgent:
    @pytest.fixture
    def agent(self):
        return MathAgent()
    
    @pytest.mark.asyncio
    async def test_simple_calculation(self, agent):
        """测试简单计算"""
        state = MathState(expression="2 + 3")
        app = agent.build_app()
        result = await app.ainvoke(state.dict())
        
        assert result["result"] == 5.0
        assert not result["error"]
    
    @pytest.mark.asyncio
    async def test_invalid_expression(self, agent):
        """测试无效表达式"""
        state = MathState(expression="invalid")
        app = agent.build_app()
        result = await app.ainvoke(state.dict())
        
        assert result["error"]
        assert result["result"] == 0.0
```

### 集成测试

```python
# tests/test_integration.py
import pytest
from dataflow.agent_v2.master.agent import MasterAgent

class TestIntegration:
    @pytest.mark.asyncio
    async def test_math_tool_integration(self):
        """测试数学工具集成"""
        agent = MasterAgent()
        
        # 模拟用户输入
        initial_state = {
            "input": "请计算 25 * 4 + 10",
            "query": "请计算 25 * 4 + 10"
        }
        
        # 执行
        app = agent.build_app()
        result = await app.ainvoke(initial_state)
        
        # 验证结果
        assert "110" in str(result.get("agent_outcome", ""))
```

## ⚡ 性能优化

### 异步最佳实践

```python
# ✅ 推荐：并发执行
async def process_multiple_tools(self, tools_data):
    tasks = [
        self.execute_tool(tool_name, data)
        for tool_name, data in tools_data.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# ❌ 避免：顺序执行
async def process_multiple_tools_bad(self, tools_data):
    results = []
    for tool_name, data in tools_data.items():
        result = await self.execute_tool(tool_name, data)  # 顺序等待
        results.append(result)
    return results
```

### 内存管理

```python
class YourAgent(SubAgent):
    def __init__(self):
        super().__init__()
        self._cache = {}
        self._cache_size_limit = 1000
    
    async def execute_with_cache(self, key, computation):
        """带缓存的执行"""
        if key in self._cache:
            return self._cache[key]
        
        result = await computation()
        
        # 缓存大小控制
        if len(self._cache) >= self._cache_size_limit:
            # 移除最旧的条目 (简单LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = result
        return result
```

### 事件系统优化

```python
class OptimizedWebSocketSink(EventSink):
    """优化的WebSocket事件推送器"""
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.event_queue = asyncio.Queue(maxsize=100)
        self.sender_task = asyncio.create_task(self._event_sender())
    
    async def send_event(self, event: Event) -> None:
        """异步入队，避免阻塞"""
        try:
            await self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            # 队列满时丢弃最旧的事件
            try:
                self.event_queue.get_nowait()
                await self.event_queue.put_nowait(event)
            except asyncio.QueueEmpty:
                pass
    
    async def _event_sender(self):
        """后台事件发送任务"""
        while True:
            try:
                event = await self.event_queue.get()
                await self.websocket.send_json(event.dict())
            except Exception as e:
                logger.error(f"事件发送失败: {e}")
```

## 🐛 常见问题解决

### 1. 工具注册失败

**问题**: 工具无法被 Master Agent 识别

**解决方案**:
```python
# 检查工具是否正确实现基类
class YourTool(BaseTool):  # ✅ 继承 BaseTool
    @classmethod
    def name(cls) -> str:
        return "unique_tool_name"  # ✅ 唯一名称
    
    def params(self) -> Type[BaseModel]:
        return YourParams  # ✅ 返回 Pydantic 模型类

# 检查是否在 tools.py 中注册
def _register_tools(self):
    self.tools = [
        YourTool(),  # ✅ 添加到工具列表
    ]
```

### 2. 事件推送失败

**问题**: WebSocket 事件无法推送到前端

**解决方案**:
```python
# 检查 WebSocket 连接状态
class SafeWebSocketSink(EventSink):
    async def send_event(self, event: Event) -> None:
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.send_json(event.dict())
        except WebSocketDisconnect:
            logger.info("WebSocket 已断开")
        except Exception as e:
            logger.error(f"事件推送失败: {e}")
```

### 3. 状态丢失问题

**问题**: 多轮对话中状态信息丢失

**解决方案**:
```python
# 实现持久化状态管理
class StatefulAgent:
    def __init__(self):
        self.session_states = {}
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """获取会话状态"""
        if session_id not in self.session_states:
            self.session_states[session_id] = self._create_default_state()
        return self.session_states[session_id]
    
    def update_session_state(self, session_id: str, updates: Dict[str, Any]):
        """增量更新会话状态"""
        current_state = self.get_session_state(session_id)
        current_state.update(updates)
```

### 4. LLM 调用超时

**问题**: LLM 请求超时导致工具执行失败

**解决方案**:
```python
import asyncio

async def safe_llm_call(self, prompt: str, timeout: int = 30):
    """带超时的 LLM 调用"""
    try:
        response = await asyncio.wait_for(
            self.llm.ainvoke(prompt),
            timeout=timeout
        )
        return response
    except asyncio.TimeoutError:
        logger.error(f"LLM 调用超时 ({timeout}s)")
        return "抱歉，处理超时，请稍后重试。"
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return f"处理失败: {str(e)}"
```

## 🚀 部署建议

### Docker 容器化

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY dataflow/ ./dataflow/

EXPOSE 8765

CMD ["python", "dataflow/agent_v2/start_server.py"]
```

### 配置管理

```yaml
# config.yaml
llm:
  api_key: ${LLM_API_KEY}
  api_url: ${LLM_API_URL}
  model: ${LLM_MODEL:-gpt-4}
  timeout: 30

server:
  host: ${SERVER_HOST:-0.0.0.0}
  port: ${SERVER_PORT:-8765}

logging:
  level: ${LOG_LEVEL:-INFO}
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 监控和日志

```python
# 结构化日志
import structlog

logger = structlog.get_logger()

@node
async def monitored_operation(self, state):
    """带监控的操作"""
    start_time = time.time()
    
    try:
        result = await self.do_operation(state)
        
        logger.info(
            "operation_completed",
            operation="your_operation",
            duration=time.time() - start_time,
            success=True
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "operation_failed",
            operation="your_operation", 
            duration=time.time() - start_time,
            error=str(e),
            success=False
        )
        raise
```

---

## 📚 参考资源

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [MyScaleKB-Agent 参考](https://github.com/myscale/myscalekb)
- [Pydantic 文档](https://docs.pydantic.dev/)
- [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/)

---

> 📖 本文档持续更新，如有问题请提交 Issue
> 💡 欢迎贡献更多最佳实践和解决方案
> 🚀 Happy Coding!
