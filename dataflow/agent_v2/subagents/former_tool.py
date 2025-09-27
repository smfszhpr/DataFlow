"""
Former Tool - 简化版智能需求分析和表单处理工具
基于LLM的需求分析，动态了解工作流参数需求
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from ..llm_client import get_llm_client

logger = logging.getLogger(__name__)


class FormerToolParams(BaseModel):
    """Former Tool参数模型"""
    user_query: str = Field(description="用户查询或需求描述")
    session_id: Optional[str] = Field(default=None, description="会话ID，用于保持上下文")
    action: Optional[str] = Field(default="create_form", description="操作类型：create_form, continue_chat, submit_form")
    user_response: Optional[str] = Field(default=None, description="用户响应（用于表单交互）")
    form_data: Optional[Dict[str, Any]] = Field(default=None, description="现有表单数据（已填写为非空值，未填写为空值）")


class FormerTool:
    """Former Tool - 简化版智能需求分析工具"""
    
    def __init__(self):
        self.llm = get_llm_client()
        # 使用统一的工作流注册表
        from dataflow.agent_v2.master.tools import WorkflowRegistry
        self.workflow_registry_manager = WorkflowRegistry()
        self.workflow_registry = self.workflow_registry_manager.get_all_workflows()

    

    
    @classmethod
    def name(cls) -> str:
        """工具名称"""
        return "former"
    
    @classmethod
    def description(cls) -> str:
        """工具描述"""
        return "智能表单生成和用户交互处理工具，用于收集和整理用户需求"
    
    @classmethod
    def prerequisite_tools(cls) -> List[str]:
        """前置工具列表"""
        return []
    
    @classmethod
    def suggested_followup_tools(cls) -> List[str]:
        """建议的后置工具列表"""
        return ["codeworkflow"]
    
    @classmethod
    def get_tool_metadata(cls) -> Dict[str, Any]:
        """获取工具的完整元数据"""
        return {
            "name": cls.name(),
            "description": cls.description(),
            "prerequisite_tools": cls.prerequisite_tools(),
            "suggested_followup_tools": cls.suggested_followup_tools()
        }
    
    def params(self) -> type:
        """工具参数模型"""
        return FormerToolParams

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """执行Former工具
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            执行结果
        """
        try:
            params = FormerToolParams(**kwargs)
            logger.info(f"🔍 Former Tool 执行开始")
            logger.info(f"🔍 参数详情:")
            logger.info(f"  - Action: {params.action}")
            logger.info(f"  - Session ID: {params.session_id}")
            logger.info(f"  - User Query: {params.user_query}")
            logger.info(f"  - User Response: {params.user_response}")
            logger.info(f"🔍 完整参数字典: {params.dict()}")
            
            # 检查user_query中是否包含上下文信息
            if "[上下文信息]" in params.user_query:
                logger.info(f"🔍 检测到增强查询，包含上下文信息")
                parts = params.user_query.split("[上下文信息]")
                if len(parts) > 1:
                    original_query = parts[0].strip()
                    context_info = parts[1].strip()
                    logger.info(f"🔍 原始查询: {original_query}")
                    logger.info(f"🔍 上下文信息: {context_info}")
            else:
                logger.info(f"🔍 未检测到上下文信息，这是新的查询")
            
            # 会话状态全部由外部传入和返回，不再维护 self.sessions
            session_id = params.session_id or str(uuid.uuid4())
            session_data = params.dict()  # 仅用参数传递会话状态
            
            logger.info(f"🔍 即将执行的action: {params.action}")
            
            # 🔥 简化：统一使用 _create_form 处理所有情况
            # _create_form 已经包含了检测和处理现有表单数据的逻辑
            if params.action == "submit_form":
                result = self._submit_form(params, session_id, session_data)
            else:
                # 所有其他情况（create_form, collect_user_response 等）都用 _create_form
                result = await self._create_form(params, session_id)
            
            logger.info(f"🔍 Former Tool 执行结果概览:")
            logger.info(f"  - Success: {result.get('success', 'unknown')}")
            logger.info(f"  - Session ID: {result.get('session_id', 'unknown')}")
            logger.info(f"  - Form Stage: {result.get('form_stage', 'unknown')}")
            logger.info(f"  - Waiting for Input: {result.get('waiting_for_input', 'unknown')}")
            if result.get('missing_params'):
                logger.info(f"  - Missing Params Count: {len(result['missing_params'])}")
            if result.get('extracted_params'):
                logger.info(f"  - Extracted Params: {list(result['extracted_params'].keys())}")
            if result.get('form_data', {}).get('fields'):
                logger.info(f"  - Form Fields: {list(result['form_data']['fields'].keys())}")
            
            return result
            
        except Exception as e:
            logger.error(f"Former Tool 执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Former Tool 执行失败: {str(e)}",
                "session_id": params.session_id
            }
    
    async def _create_form(self, params: FormerToolParams, session_id: str) -> Dict[str, Any]:
        """创建表单 - 使用LLM智能分析用户需求并了解工作流参数"""
        try:
            logger.info(f"🔍 _create_form 开始 - Session: {session_id}")
            
            user_prompt = params.user_query
            existing_form_data = params.form_data or {}
            
            logger.info(f"🔍 原始用户查询: {user_prompt}")
            logger.info(f"🔍 现有表单数据: {existing_form_data}")
            logger.info(f"🔍 现有表单数据类型: {type(existing_form_data)}")
            
            # 🔥 新增：如果有现有表单数据，在用户查询中添加上下文
            if existing_form_data:
                logger.info(f"🔍 检测到现有表单数据，准备添加上下文")
                filled_fields = {k: v for k, v in existing_form_data.items() if v and str(v).strip()}
                logger.info(f"🔍 有效填写字段: {filled_fields}")
                if filled_fields:
                    context_info = f"\n\n[现有表单数据] 用户已填写的字段: {filled_fields}"
                    user_prompt += context_info
                    logger.info(f"🔄 已将现有表单数据添加到查询上下文中")
                    logger.info(f"🔄 增强后的查询: {user_prompt}")
                else:
                    logger.info(f"🔍 现有表单数据为空或无有效字段")
            else:
                logger.info(f"🔍 无现有表单数据")
            
            # 🎯 使用LLM进行需求分析和工作流匹配
            logger.info(f"🔍 开始LLM分析和工作流匹配")
            analysis_result = await self._llm_analyze_and_match_workflow(user_prompt)
            logger.info(f"🔍 LLM分析结果: {analysis_result}")
            
            if not analysis_result.get("success"):
                return {
                    "success": False,
                    "error": "LLM分析失败",
                    "message": analysis_result.get("response_message", "❌ 无法分析用户需求，请重新描述您的需求"),
                    "session_id": session_id
                }
            
            # 从分析结果中提取信息
            target_workflow = analysis_result.get("target_workflow")
            extracted_params = analysis_result.get("extracted_params", {})
            missing_params = analysis_result.get("missing_params", [])
            decision = analysis_result.get("decision")  # "ready_to_execute", "need_more_info", "clarification_needed"
            response_message = analysis_result.get("response_message")
            
            # 🔥 新增：合并现有表单数据和提取的参数
            if existing_form_data:
                # 将非空的现有表单数据合并到extracted_params中
                filled_data = {k: v for k, v in existing_form_data.items() if v and str(v).strip()}
                extracted_params.update(filled_data)
                logger.info(f"🔄 合并现有表单数据: {filled_data}")
            
            # 🔥 新增：重新评估missing_params，排除已填写的字段
            if extracted_params:
                # 过滤掉已经有值的参数
                missing_params = [p for p in missing_params if p.get("name", p) not in extracted_params]
                logger.info(f"🔄 重新评估缺失参数: {[p.get('name', p) for p in missing_params]}")
            
            # 🔥 新增：基于实际情况重新决策，但始终需要用户确认
            if not missing_params and extracted_params:
                # 参数已完整，但仍需用户确认（除非用户明确表示确认）
                user_confirmed = any(keyword in params.user_query.lower() for keyword in ["确认", "确定", "开始", "执行", "提交", "是的", "yes"])
                if user_confirmed:
                    decision = "ready_to_execute"
                    logger.info(f"🔄 用户已确认，决策为ready_to_execute")
                else:
                    decision = "need_more_info"  # 参数完整但需要确认
                    logger.info(f"🔄 参数已完整但需要用户确认，保持need_more_info决策")
            elif missing_params:
                # 仍有缺失参数，需要更多信息
                decision = "need_more_info"
                logger.info(f"🔄 仍有缺失参数，决策为need_more_info")
            
            # 构建会话状态 - 不重复存储已在顶层的字段
            session_data = {
                "user_prompt": user_prompt,
                "target_workflow": target_workflow,
                "decision": decision,
                "form_stage": "parameter_collection",
                "created_at": str(datetime.now())
            }
            
            # 🔄 根据决策确定下一步动作  
            if decision == "ready_to_execute":
                # 参数完整，直接提供完整代码
                response_message = None
            elif decision == "need_more_info":
                # 需要收集更多参数，等待用户输入
                response_message = analysis_result.get("response_message", "请提供更多信息")
            else:  # clarification_needed
                # 需要澄清需求，等待用户输入
                response_message = analysis_result.get("response_message", "请澄清您的需求")
            
            # 🔥 构建统一的form_data结构
            form_data = {}
            # 添加所有参数（已填写和未填写）
            for param in missing_params:
                param_name = param.get("name", param) if isinstance(param, dict) else param
                form_data[param_name] = extracted_params.get(param_name, "")  # 未填写为空字符串
            
            # 添加已提取的参数
            for param_name, param_value in extracted_params.items():
                form_data[param_name] = param_value
            
            return {
                "success": True,
                "message": response_message,
                "session_id": session_id,
                "form_stage": "parameter_collection",
                "target_workflow": target_workflow,
                "form_data": form_data,  # 统一的表单数据结构
                "requires_user_input": decision != "ready_to_execute",
                "form_complete": decision == "ready_to_execute",
                "routing_reason": f"需求分析决策: {decision}",
                # ✅ 添加前端渲染必需的字段
                "missing_params": missing_params,
                "extracted_params": extracted_params
            }
            
        except Exception as e:
            logger.error(f"创建表单失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"分析需求失败: {str(e)}",
                "session_id": session_id
            }

    def _submit_form(self, params: FormerToolParams, session_id: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """提交表单 - 跳转到对应工作流"""
        try:
            logger.info(f"提交表单 - Session: {session_id}")
            
            # 直接调用表单提交处理，它会包含工作流跳转逻辑
            result = self._handle_form_submission(session_data)
            
            # 添加会话数据
            result["session_data"] = session_data
            
            return result
            
        except Exception as e:
            logger.error(f"提交表单失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"提交表单失败: {str(e)}",
                "session_id": session_id
            }
    
    def _handle_form_submission(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理表单提交 - 验证并跳转到对应工作流"""
        logger.info("处理表单提交...")
        
        # 从会话数据中获取信息
        target_workflow = session_data.get("target_workflow")
        extracted_params = session_data.get("extracted_params", {})
        session_id = session_data.get("session_id", "unknown")
        
        # 🎯 核心功能：验证参数完整性
        workflow_info = self.workflow_registry.get(target_workflow)
        if not workflow_info:
            return {
                "success": False,
                "message": f"❌ 找不到目标工作流: {target_workflow}",
                "session_id": session_id,
                
                "requires_user_input": True
            }
        
        # 验证必需参数
        required_params = workflow_info.get("required_params", [])
        missing_params = []
        for param in required_params:
            if param not in extracted_params or not extracted_params[param]:
                missing_params.append(param)
        
        if missing_params:
            missing_params_str = "、".join(missing_params)
            
            return {
                "success": False,
                "message": f"❌ 缺少必需参数: {missing_params_str}\n\n请提供这些参数后重新提交。",
                "session_id": session_id,
                
                "requires_user_input": True,
                "missing_params": missing_params
            }
        
        # 🎯 参数验证通过，准备跳转工作流
        session_data["form_stage"] = "submitted"
        session_data["form_validated"] = True
        session_data["waiting_for_input"] = False
        
        # 💫 构建工作流执行参数
        workflow_execution_params = self._build_workflow_params(target_workflow, extracted_params)
        
        success_msg = f"""
✅ **表单提交成功！正在跳转到工作流执行...**

**目标工作流：** {target_workflow}
**提交时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 **执行参数：**
{self._format_params_display(extracted_params)}

🚀 **即将启动工作流处理...**
"""
        
        # 🎯 关键：工作流跳转指令
        return {
            "success": True,
            "message": success_msg,
            "session_id": session_id,
            "form_stage": "submitted",
            "form_validated": True,
            "form_data": extracted_params,
            "submitted": True,
            "requires_user_input": False,
            "target_workflow": target_workflow,
            "workflow_execution_params": workflow_execution_params,
            "routing_reason": "表单提交完成"
        }
    
    def _build_workflow_params(self, workflow_name: str, extracted_params: Dict[str, Any]) -> Dict[str, Any]:
        """构建工作流执行参数"""
        if workflow_name == "code_workflow_agent":
            # 构建CodeWorkflow需要的参数格式
            return {
                "requirement": extracted_params.get("requirement", ""),
                "max_iterations": extracted_params.get("max_iterations", 5),
                "timeout_seconds": extracted_params.get("timeout_seconds", 300),
                "apikey": extracted_params.get("apikey"),
                "url": extracted_params.get("url")
            }
        elif workflow_name == "pipeline_workflow_agent":
            # 构建PipelineWorkflow需要的参数格式
            return {
                "json_file": extracted_params.get("json_file", ""),
                "target": extracted_params.get("target", ""),
                "python_file_path": extracted_params.get("python_file_path", ""),
                "language": extracted_params.get("language", "zh"),
                "chat_api_url": extracted_params.get("chat_api_url"),
                "api_key": extracted_params.get("api_key"),
                "model": extracted_params.get("model", "gpt-4o"),
                "need_debug": extracted_params.get("need_debug", True),
                "max_debug_rounds": extracted_params.get("max_debug_rounds", 3)
            }
        else:
            # 其他工作流的参数构建逻辑
            return extracted_params
    
    def _format_params_display(self, params: Dict[str, Any]) -> str:
        """格式化参数显示"""
        lines = []
        for key, value in params.items():
            if value is not None:
                lines.append(f"- **{key}**: {value}")
        return "\n".join(lines) if lines else "无特殊参数"
    
    async def _llm_analyze_and_match_workflow(self, user_input: str) -> Dict[str, Any]:
        """使用LLM深度分析用户需求并智能匹配工作流"""
        try:
            # 🎯 构建真实的工作流参数信息
            workflow_descriptions = []
            for workflow_name, workflow_info in self.workflow_registry.items():
                params_schema = workflow_info.get("params_schema", {})
                
                # 分析必填和可选参数
                required_params = []
                optional_params = []
                
                for param_name, param_info in params_schema.items():
                    param_desc = f"{param_name}: {param_info.get('description', 'No description')}"
                    if param_info.get("required", False):
                        required_params.append(param_desc)
                    else:
                        default_val = param_info.get("default", "无默认值")
                        optional_params.append(f"{param_desc} (默认: {default_val})")
                
                workflow_desc = f"""
工作流: {workflow_name}
描述: {workflow_info.get('description', 'No description')}
必填参数:
{chr(10).join(f"  - {p}" for p in required_params) if required_params else "  无"}
可选参数:
{chr(10).join(f"  - {p}" for p in optional_params) if optional_params else "  无"}"""
                
                workflow_descriptions.append(workflow_desc)
            
            workflows_text = "\n".join(workflow_descriptions)
            
            # 🧠 深度理解提示词 - 避免关键词匹配
            system_prompt = f"""你是一个高级AI需求分析师，具备深度理解用户意图的能力。

你的任务是：
1. 深度理解用户的真实需求和意图
2. 选择最适合的工作流来满足用户需求
3. 智能提取或推断工作流所需的参数
4. 评估信息完整性并决定下一步行动

可用的工作流：
{workflows_text}

分析原则：
- 理解用户需求背后的真实意图
- 考虑用户的技术水平和表达习惯
- 从上下文中智能推断缺失的参数
- 如果信息不完整，明确指出需要什么额外信息

决策标准：
- ready_to_execute: 只有当用户明确表示要执行、确认或提交时才选择
- need_more_info: 需求明确但缺少关键参数，如果参数已完整就简短回答让用户确认是否执行
- clarification_needed: 需求本身不够清晰，需要澄清意图

输出JSON格式：
{{
    "success": true,
    "target_workflow": "最适合的工作流名称",
    "extracted_params": {{"参数名": "智能提取或推断的值"}},
    "missing_params": [{{
        "name": "参数名",
        "description": "参数说明"
    }}],
    "decision": "ready_to_execute|need_more_info|clarification_needed",
    "response_message": "给用户的自然、个性化回复",
    "reasoning": "详细的决策推理过程"
}}"""

            user_prompt = f"""请深度分析以下用户需求：

用户输入: "{user_input}"

1. 理解用户的真实意图和目标
2. 考虑用户可能的技术背景
3. 分析需求的复杂度和范围
4. 智能推断可能的参数值
5. 评估信息的完整性"""
            
            # 调用LLM进行深度分析
            response = await self.llm.acall_llm(system_prompt, user_prompt)
            content = response.get('content', '').strip()
            
            if not content:
                return {
                    "success": False,
                    "response_message": "抱歉，我暂时无法分析您的需求。请稍后再试。"
                }
            
            # 清理和解析JSON
            content = self._clean_json_response(content)
            
            try:
                result = json.loads(content)
                result["success"] = True
                
                # 验证和修正工作流选择
                target_workflow = result.get("target_workflow")
                if target_workflow not in self.workflow_registry:
                    raise ValueError("选择了不存在的工作流")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}, 内容: {content}")
                return {
                    "success": False,
                    "response_message": "抱歉，需求分析过程中出现问题。请重新描述您的需求。",
                    "raw_response": content
                }
                
        except Exception as e:
            logger.error(f"深度需求分析失败: {e}")
            return {
                "success": False,
                "response_message": f"分析过程出错: {str(e)}。请重新描述您的需求。"
            }
    
    def _clean_json_response(self, content: str) -> str:
        """清理LLM响应中的JSON格式"""
        # 移除markdown代码块标记
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
        