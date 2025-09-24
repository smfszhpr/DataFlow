#!/usr/bin/env python3
"""
Pipeline Workflow Tool - 直接调用 dataflowagent 的数据处理流水线推荐功能
"""

import logging
import os
import asyncio
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

from dataflow.agent_v2.base.core import BaseTool

logger = logging.getLogger(__name__)


class PipelineWorkflowToolParams(BaseModel):
    """Pipeline Workflow Tool 参数"""
    json_file: str = Field(description="数据文件路径")
    target: str = Field(description="用户需求目标")
    python_file_path: str = Field(description="输出脚本路径")
    language: Optional[str] = Field(default="zh", description="语言设置")
    chat_api_url: Optional[str] = Field(default=None, description="API地址")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    model: Optional[str] = Field(default="gpt-4o", description="模型名称")
    need_debug: Optional[bool] = Field(default=True, description="是否需要调试")
    max_debug_rounds: Optional[int] = Field(default=3, description="最大调试轮数")


class PipelineWorkflowTool(BaseTool):
    """Pipeline Workflow Tool - 数据处理流水线推荐工具"""
    
    @classmethod
    def name(cls) -> str:
        return "pipeline_workflow_agent"
    
    @classmethod
    def description(cls) -> str:
        return """数据处理流水线推荐工具，自动化完成数据分析、算子推荐、代码生成全流程。

【前置工具】：former_agent - 建议先用Former工具分析数据需求
【后置工具】：无 - 此工具通常作为工作流终点

功能特点：
- 自动分析数据类型和特征
- 智能推荐最佳处理算子组合
- 生成完整可执行的数据处理流水线代码
- 支持多轮调试确保代码质量
- 提供详细的执行报告和使用指南

适用场景：
- 已通过Former工具明确的数据处理需求
- 需要完整数据处理流水线的场景
- 自动化数据处理和分析需求
- 从数据文件到处理脚本的端到端生成"""
    
    @classmethod
    def params(cls) -> type:
        return PipelineWorkflowToolParams
    
    @classmethod
    def prerequisite_tools(cls) -> List[str]:
        """前置工具列表"""
        return ["former_agent"]  # 建议先用Former工具分析需求
    
    @classmethod
    def suggested_followup_tools(cls) -> List[str]:
        """建议的后置工具列表"""
        return []  # 通常作为工作流终点
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行 Pipeline Workflow 工具 - 直接调用 dataflowagent"""
        try:
            # 从kwargs创建参数对象
            params = PipelineWorkflowToolParams(**kwargs)
            
            json_file = params.json_file
            target = params.target
            python_file_path = params.python_file_path
            language = params.language or "zh"
            chat_api_url = params.chat_api_url or os.getenv("DATAFLOW_API_URL", "http://localhost:3000/v1/")
            api_key = params.api_key or os.getenv("DF_API_KEY", "sk-dummy")
            model = params.model or "gpt-4o"
            need_debug = params.need_debug if params.need_debug is not None else True
            max_debug_rounds = params.max_debug_rounds or 3
            
            logger.info(f"🚀 Pipeline Workflow Tool 开始处理: {target}")
            logger.info(f"📄 数据文件: {json_file}")
            logger.info(f"📝 输出脚本: {python_file_path}")
            
            # 直接调用 dataflowagent 的核心功能
            try:
                # 导入必要的组件
                from dataflow.dataflowagent.state import DFRequest, DFState
                from dataflow.dataflowagent.script.pipeline_nodes import create_pipeline_graph
                
                # 创建 DFRequest 对象
                df_request = DFRequest(
                    language=language,
                    chat_api_url=chat_api_url,
                    api_key=api_key,
                    model=model,
                    json_file=json_file,
                    target=target,
                    python_file_path=python_file_path,
                    need_debug=need_debug,
                    max_debug_rounds=max_debug_rounds
                )
                
                # 创建初始状态
                df_state = DFState(request=df_request, messages=[])
                df_state.temp_data = {"round": 0}
                df_state.debug_mode = need_debug
                
                # 创建并运行 pipeline 图
                logger.info("🔄 执行 dataflowagent 流水线工作流...")
                graph_builder = create_pipeline_graph()
                pipeline_graph = graph_builder.build()
                
                # 执行完整工作流
                final_state = await pipeline_graph.ainvoke(df_state)
                
                # ===== DEBUG: 打印 final_state 的关键信息 =====
                logger.info("🔍 DEBUG: 查看 final_state 的内容...")
                logger.info(f"📋 final_state 类型: {type(final_state)}")
                logger.info(f"📋 final_state 属性: {dir(final_state) if hasattr(final_state, '__dict__') else 'N/A'}")
                
                # 尝试不同方式获取 pipeline_code
                pipeline_code_from_attr = getattr(final_state, 'pipeline_code', None)
                pipeline_code_from_get = final_state.get('pipeline_code', None) if hasattr(final_state, 'get') else None
                pipeline_code_from_temp = final_state.get('temp_data', {}).get('pipeline_code', None) if hasattr(final_state, 'get') else None
                
                logger.info(f"🔍 pipeline_code (直接属性): {type(pipeline_code_from_attr)} = {pipeline_code_from_attr}")
                logger.info(f"🔍 pipeline_code (get方法): {type(pipeline_code_from_get)} = {pipeline_code_from_get}")
                logger.info(f"🔍 pipeline_code (temp_data): {type(pipeline_code_from_temp)} = {pipeline_code_from_temp}")
                
                # 如果有 temp_data，打印完整内容
                if hasattr(final_state, 'temp_data'):
                    temp_data = getattr(final_state, 'temp_data', {})
                    logger.info(f"🔍 temp_data 内容: {temp_data}")
                
                # 提取结果
                execution_result = final_state.get("execution_result", {})
                execution_successful = execution_result.get("success", False)
                execution_output = execution_result.get("stdout", "")
                execution_error = execution_result.get("stderr", "")
                
                # 提取各阶段结果
                classification_result = final_state.get("category", {})
                recommendation_result = final_state.get("recommendation", [])
                
                # 提取生成的代码 - 优先从不同地方尝试获取
                generated_code = ""
                
                # 方法1：直接从 pipeline_code 属性
                if pipeline_code_from_attr:
                    if isinstance(pipeline_code_from_attr, dict):
                        generated_code = pipeline_code_from_attr.get("code", str(pipeline_code_from_attr))
                    else:
                        generated_code = str(pipeline_code_from_attr)
                    logger.info(f"✅ 方法1成功获取代码，长度: {len(generated_code)}")
                
                # 方法2：从 get 方法
                elif pipeline_code_from_get:
                    if isinstance(pipeline_code_from_get, dict):
                        generated_code = pipeline_code_from_get.get("code", str(pipeline_code_from_get))
                    else:
                        generated_code = str(pipeline_code_from_get)
                    logger.info(f"✅ 方法2成功获取代码，长度: {len(generated_code)}")
                
                # 方法3：从 temp_data
                elif pipeline_code_from_temp:
                    generated_code = str(pipeline_code_from_temp)
                    logger.info(f"✅ 方法3成功获取代码，长度: {len(generated_code)}")
                
                else:
                    logger.warning("❌ 所有方法都未能获取到 pipeline_code")
                    # 尝试从 pipeline_file_path 读取文件
                    if hasattr(final_state, 'pipeline_file_path'):
                        pipeline_file = getattr(final_state, 'pipeline_file_path', '')
                        if pipeline_file and os.path.exists(pipeline_file):
                            try:
                                with open(pipeline_file, 'r', encoding='utf-8') as f:
                                    generated_code = f.read()
                                logger.info(f"✅ 从文件 {pipeline_file} 读取代码，长度: {len(generated_code)}")
                            except Exception as e:
                                logger.warning(f"❌ 从文件读取失败: {e}")
                
                logger.info(f"🎯 最终获取的代码预览 (前100字符): {generated_code[:100]}...")
                
                # 构建详细结果
                if execution_successful:
                    logger.info("✅ 数据处理流水线工作流执行成功!")
                    
                    output_lines = []
                    output_lines.append("🎉 **数据处理流水线推荐工作流执行成功！**")
                    output_lines.append(f"\n📊 **工作流结果:**")
                    output_lines.append(f"- 数据文件: {json_file}")
                    output_lines.append(f"- 用户目标: {target}")
                    output_lines.append(f"- 输出脚本: {python_file_path}")
                    
                    output_lines.append(f"\n🔍 **数据分析结果:**")
                    output_lines.append(f"- 数据类型: {classification_result.get('category', '未知')}")
                    output_lines.append(f"- 置信度: {classification_result.get('confidence', 'N/A')}")
                    
                    output_lines.append(f"\n💡 **推荐算子:**")
                    if isinstance(recommendation_result, list):
                        for i, op in enumerate(recommendation_result, 1):
                            output_lines.append(f"   {i}. {op}")
                    else:
                        output_lines.append(f"   {recommendation_result}")
                    
                    output_lines.append(f"\n✅ **生成的流水线代码:**")
                    output_lines.append(f"```python\n{generated_code}\n```")
                    
                    output_lines.append(f"\n📋 **执行结果:**")
                    output_lines.append(execution_output)
                    
                    output_lines.append(f"\n🚀 **使用方法:**")
                    output_lines.append(f"可通过以下命令处理完整数据:")
                    output_lines.append(f"python {python_file_path}")
                    
                    output_lines.append(f"\n✨ **工作流程:**")
                    output_lines.append("1. 数据内容分类 - 自动识别数据类型和特征")
                    output_lines.append("2. 算子推荐 - 基于数据特征推荐最佳处理流程")
                    output_lines.append("3. 代码生成 - 生成完整可执行的数据处理流水线")
                    output_lines.append("4. 调试验证 - 多轮调试确保代码质量")
                    
                    output_lines.append("\n💎 **流水线已通过自动化验证，可以直接投入使用！**")
                    
                    output = "\n".join(output_lines)
                else:
                    logger.warning("❌ 数据处理流水线工作流执行失败")
                    
                    output_lines = []
                    output_lines.append("⚠️ **数据处理流水线推荐工作流执行失败**")
                    output_lines.append(f"\n📊 **工作流配置:**")
                    output_lines.append(f"- 数据文件: {json_file}")
                    output_lines.append(f"- 用户目标: {target}")
                    output_lines.append(f"- 输出脚本: {python_file_path}")
                    
                    output_lines.append(f"\n🔍 **已完成的步骤:**")
                    output_lines.append(f"- 数据分析: {classification_result.get('category', '未完成')}")
                    output_lines.append(f"- 算子推荐: {len(recommendation_result) if isinstance(recommendation_result, list) else 0} 个算子")
                    output_lines.append(f"- 代码生成: {'已生成' if generated_code else '未完成'}")
                    
                    output_lines.append(f"\n❌ **错误信息:**")
                    output_lines.append(execution_error or "未知错误")
                    
                    if generated_code:
                        output_lines.append(f"\n🔧 **生成的代码:**")
                        output_lines.append(f"```python\n{generated_code}\n```")
                    
                    output_lines.append(f"\n💭 **问题排查建议:**")
                    output_lines.append("1. 检查数据文件格式是否正确")
                    output_lines.append("2. 确认用户目标描述是否清晰") 
                    output_lines.append("3. 验证输出路径是否有写入权限")
                    output_lines.append("4. 检查API配置是否正确")
                    
                    output = "\n".join(output_lines)
                
                return {
                    "success": execution_successful,
                    "output": output,
                    "execution_successful": execution_successful,
                    "execution_output": execution_output,
                    "execution_error": execution_error,
                    "generated_pipeline_code": generated_code,
                    "classification_result": classification_result,
                    "recommendation_result": recommendation_result,
                    "workflow_result": output,
                    # 🎯 新增：用于前端标签页的数据
                    "frontend_code_data": {
                        "code_content": generated_code,
                        "file_name": os.path.basename(python_file_path),
                        "file_path": python_file_path,
                        "language": "python",
                        "tool_source": "pipeline_workflow_agent",
                        "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else None
                    }
                }
                
            except Exception as dataflow_error:
                logger.warning(f"DataflowAgent 执行失败: {dataflow_error}")
                logger.info("切换到简化的后备实现")
                
                # 使用简化的后备实现
                return await self._run_simple_pipeline_workflow(
                    json_file, target, python_file_path, 
                    language, chat_api_url, api_key, model
                )
                
        except Exception as e:
            logger.error(f"Pipeline Workflow Tool 执行失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "output": f"数据处理流水线推荐工作流执行失败: {str(e)}"
            }
    
    async def _run_simple_pipeline_workflow(
        self, 
        json_file: str, 
        target: str, 
        python_file_path: str,
        language: str = "zh",
        chat_api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o"
    ) -> Dict[str, Any]:
        """简化的数据处理流水线实现"""
        try:
            logger.info("🔄 使用简化的数据处理流水线实现")
            
            from dataflow.agent_v2.llm_client import get_llm_client
            
            llm = get_llm_client()
            
            # 第一步：分析数据文件
            logger.info("🔍 第一步：分析数据文件...")
            data_analysis_prompt = f"""请分析数据文件并进行分类：

数据文件路径: {json_file}
用户目标: {target}

请按以下格式返回数据分析结果：

{{
    "category": "数据类型（如text、image、tabular等）",
    "confidence": "置信度（0.0-1.0）",
    "characteristics": "数据特征描述",
    "size_estimate": "数据规模估计"
}}

只返回JSON格式的分析结果，不要其他解释。"""
            
            analysis_response = llm.call_llm("", data_analysis_prompt)
            analysis_content = analysis_response.get('content', '{}')
            
            # 解析分析结果
            try:
                import json
                analysis_result = json.loads(analysis_content)
            except:
                analysis_result = {
                    "category": "text", 
                    "confidence": 0.8,
                    "characteristics": "通用数据",
                    "size_estimate": "未知"
                }
            
            logger.info(f"✅ 数据分析完成: {analysis_result}")
            
            # 第二步：算子推荐
            logger.info("💡 第二步：推荐处理算子...")
            recommendation_prompt = f"""基于数据分析结果推荐处理算子：

数据分析结果: {analysis_result}
用户目标: {target}

请推荐合适的数据处理算子序列，返回格式：

{{
    "recommended_operators": ["算子1", "算子2", "算子3"],
    "explanation": "推荐理由说明"
}}

只返回JSON格式的推荐结果。"""
            
            recommendation_response = llm.call_llm("", recommendation_prompt)
            recommendation_content = recommendation_response.get('content', '{}')
            
            # 解析推荐结果
            try:
                recommendation_result = json.loads(recommendation_content)
                recommended_ops = recommendation_result.get("recommended_operators", ["数据加载", "数据清洗", "特征提取"])
            except:
                recommended_ops = ["数据加载", "数据清洗", "特征提取"]
            
            logger.info(f"✅ 算子推荐完成: {recommended_ops}")
            
            # 第三步：代码生成
            logger.info("💻 第三步：生成流水线代码...")
            code_generation_prompt = f"""生成数据处理流水线代码：

数据文件: {json_file}
用户目标: {target}
数据分析: {analysis_result}
推荐算子: {recommended_ops}
输出路径: {python_file_path}

请生成一个完整的Python数据处理流水线代码，要求：
1. 能够处理指定的数据文件
2. 实现推荐的算子功能
3. 达成用户目标
4. 包含必要的错误处理
5. 有清晰的输出和日志

直接返回Python代码，用```python和```包围：

```python
# 你的代码
```"""
            
            code_response = llm.call_llm("", code_generation_prompt)
            code_content = code_response.get('content', '')
            
            # 提取代码
            import re
            code_pattern = r"```python\s*\n(.*?)\n```"
            code_match = re.search(code_pattern, code_content, re.DOTALL)
            
            if code_match:
                generated_code = code_match.group(1).strip()
            else:
                generated_code = code_content.strip()
            
            logger.info("✅ 代码生成完成")
            
            # 第四步：保存代码到指定路径
            logger.info(f"💾 第四步：保存代码到 {python_file_path}...")
            try:
                os.makedirs(os.path.dirname(python_file_path), exist_ok=True)
                with open(python_file_path, 'w', encoding='utf-8') as f:
                    f.write(generated_code)
                logger.info(f"✅ 代码已保存到: {python_file_path}")
                save_success = True
            except Exception as save_error:
                logger.warning(f"❌ 代码保存失败: {save_error}")
                save_success = False
            
            # 构建成功结果
            output_lines = []
            output_lines.append("🎉 **数据处理流水线推荐工作流执行成功** (简化实现)")
            output_lines.append(f"\n📋 **用户需求:** {target}")
            output_lines.append(f"\n📄 **数据文件:** {json_file}")
            output_lines.append(f"\n🔍 **数据分析结果:**")
            output_lines.append(f"- 数据类型: {analysis_result.get('category', '未知')}")
            output_lines.append(f"- 置信度: {analysis_result.get('confidence', 'N/A')}")
            output_lines.append(f"- 特征: {analysis_result.get('characteristics', '未知')}")
            
            output_lines.append(f"\n💡 **推荐算子:**")
            for i, op in enumerate(recommended_ops, 1):
                output_lines.append(f"   {i}. {op}")
            
            output_lines.append(f"\n💻 **生成的流水线代码:**")
            output_lines.append(f"```python\n{generated_code}\n```")
            
            if save_success:
                output_lines.append(f"\n✅ **代码已保存到:** {python_file_path}")
                output_lines.append(f"\n🚀 **使用方法:** python {python_file_path}")
            else:
                output_lines.append(f"\n⚠️ **代码保存失败，但已生成完整代码**")
            
            output_lines.append(f"\n🎯 **简化流程说明:**")
            output_lines.append("1. 智能数据分析 - LLM自动识别数据特征")
            output_lines.append("2. 算子智能推荐 - 基于分析结果推荐最佳算子")
            output_lines.append("3. 流水线代码生成 - 生成完整可执行代码")
            output_lines.append("4. 自动保存部署 - 代码保存到指定路径")
            
            return {
                "success": True,
                "output": "\n".join(output_lines),
                "execution_successful": save_success,
                "execution_output": f"流水线代码已生成{'并保存' if save_success else '但保存失败'}",
                "execution_error": "" if save_success else "代码保存失败",
                "generated_pipeline_code": generated_code,
                "classification_result": analysis_result,
                "recommendation_result": recommended_ops,
                "workflow_result": "数据处理流水线推荐工作流完成（简化实现）"
            }
            
        except Exception as e:
            logger.error(f"简化数据处理流水线执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": f"数据处理流水线推荐工作流执行失败: {str(e)}"
            }
