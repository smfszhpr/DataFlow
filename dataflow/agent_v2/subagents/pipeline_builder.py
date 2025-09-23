"""
Pipeline Builder SubAgent
专门用于构建数据处理管道的子代理
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from dataclasses import dataclass

from dataflow.agent_v2.base.core import SubAgent, BaseTool, node, entry, conditional_edge


logger = logging.getLogger(__name__)


class PipelineState(BaseModel):
    """Pipeline Builder 状态定义"""
    input: str = ""
    pipeline_config: Optional[Dict[str, Any]] = None
    operators: List[str] = []
    connections: List[Dict[str, str]] = []
    validation_result: Optional[Dict[str, Any]] = None
    generated_code: Optional[str] = None
    agent_outcome: Optional[Any] = None


class PipelineValidatorTool(BaseTool):
    """管道验证工具"""
    
    @classmethod
    def name(cls) -> str:
        return "pipeline_validator"
    
    @classmethod
    def description(cls) -> str:
        return "验证管道配置的有效性，检查算子连接和数据流。"
    
    def params(self) -> type[BaseModel]:
        class ValidatorParams(BaseModel):
            operators: List[str]
            connections: List[Dict[str, str]]
            data_schema: Optional[Dict[str, Any]] = None
        return ValidatorParams
    
    async def execute(self, operators: List[str], connections: List[Dict[str, str]], 
                     data_schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """验证管道"""
        await asyncio.sleep(0.5)  # 模拟验证时间
        
        # 简单的验证逻辑
        issues = []
        
        if len(operators) < 2:
            issues.append("管道至少需要2个算子")
        
        if len(connections) < len(operators) - 1:
            issues.append("算子连接不足，可能存在孤立节点")
        
        # 检查连接的有效性
        for conn in connections:
            if conn.get("from") not in operators or conn.get("to") not in operators:
                issues.append(f"无效连接: {conn}")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "validation_score": max(0, 100 - len(issues) * 20),
            "recommendations": [
                "建议添加数据预处理算子",
                "考虑添加异常处理机制"
            ] if len(issues) == 0 else []
        }


class PipelineCodeGeneratorTool(BaseTool):
    """管道代码生成工具"""
    
    @classmethod
    def name(cls) -> str:
        return "pipeline_code_generator"
    
    @classmethod
    def description(cls) -> str:
        return "根据管道配置生成可执行的 DataFlow 代码。"
    
    def params(self) -> type[BaseModel]:
        class CodeGenParams(BaseModel):
            operators: List[str]
            connections: List[Dict[str, str]]
            pipeline_name: str = "custom_pipeline"
        return CodeGenParams
    
    async def execute(self, operators: List[str], connections: List[Dict[str, str]], 
                     pipeline_name: str = "custom_pipeline") -> Dict[str, Any]:
        """生成管道代码"""
        await asyncio.sleep(1.0)  # 模拟代码生成
        
        # 生成 Python 代码
        code_lines = [
            f'"""',
            f'{pipeline_name} - 自动生成的数据处理管道',
            f'算子: {", ".join(operators)}',
            f'"""',
            f'',
            f'from dataflow.core import Pipeline, Operator',
            f'',
            f'def create_{pipeline_name}():',
            f'    """创建 {pipeline_name} 管道"""',
            f'    pipeline = Pipeline(name="{pipeline_name}")',
            f'    ',
        ]
        
        # 添加算子定义
        for i, op in enumerate(operators):
            code_lines.append(f'    # 算子 {i+1}: {op}')
            code_lines.append(f'    {op.lower()}_op = Operator(')
            code_lines.append(f'        name="{op}",')
            code_lines.append(f'        func=your_{op.lower()}_function,')
            code_lines.append(f'        input_schema=input_schema,')
            code_lines.append(f'        output_schema=output_schema')
            code_lines.append(f'    )')
            code_lines.append(f'    pipeline.add_operator({op.lower()}_op)')
            code_lines.append(f'    ')
        
        # 添加连接
        code_lines.append(f'    # 算子连接')
        for conn in connections:
            from_op = conn.get("from", "").lower()
            to_op = conn.get("to", "").lower()
            code_lines.append(f'    pipeline.connect({from_op}_op, {to_op}_op)')
        
        code_lines.extend([
            f'    ',
            f'    return pipeline',
            f'',
            f'if __name__ == "__main__":',
            f'    pipeline = create_{pipeline_name}()',
            f'    result = pipeline.execute(input_data)',
            f'    print(result)'
        ])
        
        generated_code = '\n'.join(code_lines)
        
        return {
            "success": True,
            "generated_code": generated_code,
            "pipeline_name": pipeline_name,
            "file_path": f"{pipeline_name}.py",
            "operators_count": len(operators),
            "connections_count": len(connections)
        }


class PipelineBuilderAgent(SubAgent):
    """Pipeline Builder SubAgent - 管道构建专家"""
    
    def __init__(self):
        super().__init__()
        self.tools = [
            PipelineValidatorTool(),
            PipelineCodeGeneratorTool()
        ]
    
    def state_definition(self) -> type[BaseModel]:
        return PipelineState
    
    def _setup_graph(self):
        """设置图结构"""
        # 添加节点
        self.graph_builder.add_node("entry", self.entry)
        self.graph_builder.add_node("parse_requirements", self.parse_requirements)
        self.graph_builder.add_node("validate_pipeline", self.validate_pipeline)
        self.graph_builder.add_node("generate_code", self.generate_code)
        self.graph_builder.add_node("finalize", self.finalize)
        
        # 设置边
        self.graph_builder.add_edge("entry", "parse_requirements")
        self.graph_builder.add_conditional_edge("parse_requirements", self.should_validate, {
            "validate": "validate_pipeline",
            "skip": "generate_code"
        })
        self.graph_builder.add_edge("validate_pipeline", "generate_code")
        self.graph_builder.add_edge("generate_code", "finalize")
        
        # 设置入口点
        self.graph_builder.set_entry_point("entry")
    
    @entry
    @node()
    async def entry(self, state: PipelineState) -> PipelineState:
        """入口节点 - 接收用户需求"""
        logger.info(f"Pipeline Builder Entry: {state.input}")
        return state
    
    @node()
    async def parse_requirements(self, state: PipelineState) -> PipelineState:
        """解析需求并设计管道结构"""
        user_input = state.input.lower()
        
        # 简化的需求解析逻辑
        operators = []
        connections = []
        
        # 根据关键词推断需要的算子
        if "清洗" in user_input or "clean" in user_input:
            operators.append("DataCleaner")
        
        if "转换" in user_input or "transform" in user_input:
            operators.append("DataTransformer")
        
        if "分析" in user_input or "analysis" in user_input:
            operators.append("DataAnalyzer")
        
        if "验证" in user_input or "validate" in user_input:
            operators.append("DataValidator")
        
        if "输出" in user_input or "export" in user_input:
            operators.append("DataExporter")
        
        # 如果没有明确指定，使用默认管道
        if not operators:
            operators = ["DataLoader", "DataProcessor", "DataExporter"]
        
        # 创建线性连接
        for i in range(len(operators) - 1):
            connections.append({
                "from": operators[i],
                "to": operators[i + 1]
            })
        
        state.operators = operators
        state.connections = connections
        state.pipeline_config = {
            "name": "auto_generated_pipeline",
            "operators": operators,
            "connections": connections,
            "description": f"基于需求自动生成: {state.input}"
        }
        
        logger.info(f"解析结果 - 算子: {operators}, 连接: {connections}")
        return state
    
    def should_validate(self, state: PipelineState) -> str:
        """决定是否需要验证"""
        # 如果算子数量较多或连接复杂，进行验证
        if len(state.operators) > 3 or len(state.connections) > 2:
            return "validate"
        return "skip"
    
    @node()
    async def validate_pipeline(self, state: PipelineState) -> PipelineState:
        """验证管道配置"""
        validator_tool = PipelineValidatorTool()
        
        try:
            validation_result = await validator_tool.execute(
                operators=state.operators,
                connections=state.connections
            )
            state.validation_result = validation_result
            logger.info(f"验证结果: {validation_result}")
        except Exception as e:
            logger.error(f"验证失败: {e}")
            state.validation_result = {
                "success": False,
                "issues": [f"验证过程出错: {str(e)}"]
            }
        
        return state
    
    @node()
    async def generate_code(self, state: PipelineState) -> PipelineState:
        """生成管道代码"""
        code_gen_tool = PipelineCodeGeneratorTool()
        
        try:
            code_result = await code_gen_tool.execute(
                operators=state.operators,
                connections=state.connections,
                pipeline_name=state.pipeline_config.get("name", "custom_pipeline")
            )
            
            if code_result["success"]:
                state.generated_code = code_result["generated_code"]
                logger.info("代码生成成功")
            else:
                logger.error("代码生成失败")
        except Exception as e:
            logger.error(f"代码生成出错: {e}")
        
        return state
    
    @node()
    async def finalize(self, state: PipelineState) -> PipelineState:
        """最终处理 - 生成响应"""
        
        # 构建响应信息
        response_parts = []
        
        response_parts.append(f"✅ 管道设计完成!")
        response_parts.append(f"📋 算子列表: {', '.join(state.operators)}")
        response_parts.append(f"🔗 连接数量: {len(state.connections)}")
        
        if state.validation_result:
            if state.validation_result["success"]:
                response_parts.append(f"✅ 验证通过 (得分: {state.validation_result['validation_score']})")
            else:
                response_parts.append(f"⚠️  验证发现问题: {', '.join(state.validation_result['issues'])}")
        
        if state.generated_code:
            response_parts.append(f"💻 代码已生成 ({len(state.generated_code)} 字符)")
            response_parts.append(f"\n```python\n{state.generated_code[:500]}...\n```")
        
        final_response = "\n".join(response_parts)
        
        finish = AgentFinish(
            return_values={"output": final_response}
        )
        state.agent_outcome = finish
        
        return state


# 工厂函数
def create_pipeline_builder() -> PipelineBuilderAgent:
    """创建 Pipeline Builder Agent"""
    return PipelineBuilderAgent()


if __name__ == "__main__":
    # 测试代码
    async def test_pipeline_builder():
        agent = create_pipeline_builder()
        
        # 模拟执行
        test_state = PipelineState(
            input="我需要一个数据清洗和分析的管道"
        )
        
        # 简单的执行流程
        state = await agent.entry(test_state)
        state = await agent.parse_requirements(state)
        
        if agent.should_validate(state) == "validate":
            state = await agent.validate_pipeline(state)
        
        state = await agent.generate_code(state)
        state = await agent.finalize(state)
        
        if isinstance(state.agent_outcome, AgentFinish):
            print(state.agent_outcome.return_values["output"])
    
    # asyncio.run(test_pipeline_builder())
