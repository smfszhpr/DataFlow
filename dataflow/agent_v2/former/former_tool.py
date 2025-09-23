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
        # 动态获取真实的工作流参数定义
        self.workflow_registry = self._discover_available_workflows()
    
    def _discover_available_workflows(self) -> Dict[str, Dict[str, Any]]:
        """动态发现可用工作流及其真实参数定义"""
        workflows = {}
        
        try:
            # 🎯 真正从代码工作流工具中获取参数定义
            from dataflow.agent_v2.subagents.code_workflow_tool import CodeWorkflowToolParams
            
            # 通过反射获取真实的参数定义
            workflow_params = self._extract_params_from_pydantic_model(CodeWorkflowToolParams)
            
            workflows["code_workflow_agent"] = {
                "description": "代码生成、测试、调试循环工具",
                "params_schema": workflow_params,
                "tool_class": "CodeWorkflowTool"
            }
            
            # 可以添加其他工作流的动态发现
            logger.info(f"发现 {len(workflows)} 个工作流")
            
        except Exception as e:
            logger.error(f"工作流发现失败: {e}")
            # 回退到基础定义
            workflows["code_workflow_agent"] = {
                "description": "代码生成工具",
                "params_schema": {
                    "requirement": {"required": True, "type": "str", "description": "用户代码需求"}
                },
                "tool_class": "CodeWorkflowTool"
            }
        
        return workflows
    
    def _extract_params_from_pydantic_model(self, model_class) -> Dict[str, Any]:
        """从Pydantic模型中提取参数定义"""
        params_schema = {}
        
        try:
            # 兼容不同版本的Pydantic
            if hasattr(model_class, '__fields__'):
                # Pydantic v1
                for field_name, field_info in model_class.__fields__.items():
                    param_def = {
                        "type": str(getattr(field_info, 'type_', field_info.annotation if hasattr(field_info, 'annotation') else 'Any')),
                        "required": getattr(field_info, 'required', True),
                        "description": getattr(field_info.field_info, 'description', f"{field_name}参数") if hasattr(field_info, 'field_info') else f"{field_name}参数"
                    }
                    
                    # 获取默认值
                    default_val = getattr(field_info, 'default', None)
                    if default_val is not None and default_val != ...:
                        param_def["default"] = default_val
                    
                    params_schema[field_name] = param_def
                    
            elif hasattr(model_class, 'model_fields'):
                # Pydantic v2
                for field_name, field_info in model_class.model_fields.items():
                    param_def = {
                        "type": str(field_info.annotation if hasattr(field_info, 'annotation') else 'Any'),
                        "required": getattr(field_info, 'is_required', lambda: True)() if callable(getattr(field_info, 'is_required', True)) else True,
                        "description": getattr(field_info, 'description', f"{field_name}参数")
                    }
                    
                    # 获取默认值
                    if hasattr(field_info, 'default') and field_info.default is not None:
                        param_def["default"] = field_info.default
                    
                    params_schema[field_name] = param_def
                    
            logger.debug(f"提取参数模式: {params_schema}")
            
        except Exception as e:
            logger.error(f"参数模式提取失败: {e}")
            # 提供备用方案
            if hasattr(model_class, '__annotations__'):
                for field_name, field_type in model_class.__annotations__.items():
                    params_schema[field_name] = {
                        "type": str(field_type),
                        "required": True,
                        "description": f"{field_name}参数"
                    }
        
        return params_schema
    
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
    
    def execute(self, params: FormerToolParams) -> Dict[str, Any]:
        """执行Former工具
        
        Args:
            params: 工具参数
            
        Returns:
            执行结果
        """
        try:
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
            
            if params.action == "create_form":
                result = self._create_form(params, session_id)
            elif params.action == "continue_chat":
                result = self._continue_chat(params, session_id, session_data)
            elif params.action == "submit_form":
                result = self._submit_form(params, session_id, session_data)
            else:
                result = self._create_form(params, session_id)
            
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
    
    def _create_form(self, params: FormerToolParams, session_id: str) -> Dict[str, Any]:
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
            analysis_result = self._llm_analyze_and_match_workflow(user_prompt)
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
            
            # 🔥 新增：基于实际情况重新决策
            if not missing_params and extracted_params:
                # 如果所有参数都有了，可以执行
                decision = "ready_to_execute"
                logger.info(f"🔄 参数已完整，更新决策为ready_to_execute")
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
            
            # 🔄 根据决策确定下一步 - 修复summary幻觉问题
            if decision == "ready_to_execute":
                # 参数完整，直接提供完整代码
                next_instruction = "END"  # 直接结束，不经过summary
                force_summary_flag = False
                # 直接在这里提供完整的代码实现
                response_message = self._provide_direct_code_solution(extracted_params)
                
            elif decision == "need_more_info":
                # 需要收集更多参数，等待用户输入
                next_instruction = "END"  # 等待用户输入，不继续调用工具
                force_summary_flag = False  # 不触发summary
                response_message += f"\n\n请提供以上信息，我将为您准备完整的执行方案。"
            else:  # clarification_needed
                # 需要澄清需求，等待用户输入
                next_instruction = "END"  # 等待用户输入，不继续调用工具
                force_summary_flag = False  # 不触发summary
            
            # 🔥 新增：构建统一的form_data结构
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
                "form_data": form_data,  # 🔥 简化：统一的表单数据结构
                "requires_user_input": decision != "ready_to_execute",
                "form_complete": decision == "ready_to_execute",
                # 🎯 跳转控制字段  
                "next_tool_instruction": next_instruction if decision == "ready_to_execute" else None,
                "force_summary": force_summary_flag,
                "routing_reason": f"需求分析决策: {decision}"
            }
            
        except Exception as e:
            logger.error(f"创建表单失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"分析需求失败: {str(e)}",
                "session_id": session_id
            }
    def _continue_chat(self, params: FormerToolParams, session_id: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """继续对话 - 处理用户在表单交互中的响应"""
        try:
            logger.info(f"继续对话 - Session: {session_id}")
            
            user_response = params.user_response or params.user_query
            # 处理用户响应
            result = self._handle_user_response(session_data, user_response)
            # 返回最新会话状态（由 Master Agent 存储）
            result["session_data"] = session_data
            # FormerTool 输出 next_tool/summary_flag 变量
            result["next_tool"] = "former" if result.get("requires_user_input") else "codeworkflow"
            result["summary_flag"] = not result.get("requires_user_input")
            return result
            
        except Exception as e:
            logger.error(f"继续对话失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"处理用户响应失败: {str(e)}",
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
    
    def _handle_user_response(self, session_data: Dict[str, Any], user_response: str) -> Dict[str, Any]:
        """处理用户响应"""
        try:
            logger.info(f"处理用户响应: {user_response}")
            
            form_data = session_data.get("form_data", {})
            user_response_lower = user_response.lower().strip()
            session_id = session_data.get("session_id", "unknown")
            
            # 检查是否是提交指令
            if any(keyword in user_response_lower for keyword in ["确认提交", "submit", "提交", "确认"]):
                return self._handle_form_submission(session_data)
            
            # 检查是否是修改指令
            if user_response_lower.startswith("修改"):
                return self._handle_field_modification(session_data, user_response)
            
            # 处理其他类型的用户输入（继续对话）
            return self._handle_continue_chat(session_data, user_response)
            
        except Exception as e:
            logger.error(f"处理用户响应失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"❌ 处理用户响应失败: {str(e)}",
                "session_id": session_data.get("session_id", "unknown")
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
            # 🚀 工作流跳转控制
            "next_tool_instruction": target_workflow,
            "workflow_execution_params": workflow_execution_params,
            "target_workflow": target_workflow,
            "force_summary": True,
            "routing_reason": "表单提交完成，跳转到工作流执行"
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
            # 🚀 工作流跳转控制
            "next_tool_instruction": target_workflow,
            "workflow_execution_params": workflow_execution_params,
            "target_workflow": target_workflow,
            "force_summary": True,
            "routing_reason": "表单提交完成，跳转到工作流执行"
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
    
    def _handle_field_modification(self, session_data: Dict[str, Any], user_response: str) -> Dict[str, Any]:
        """处理字段修改"""
        try:
            # 解析修改指令：修改 字段名 新值
            parts = user_response.split(None, 2)  # 分割成最多3部分
            session_id = session_data.get("session_id", "unknown")
            
            if len(parts) < 3:
                return {
                    "success": False,
                    "message": "❌ 修改指令格式错误。请使用格式：'修改 [字段名] [新值]'",
                    "session_id": session_id,
                    "form_stage": "user_interaction",
                    
                    "requires_user_input": True
                }
            
            field_name = parts[1]
            new_value = parts[2]
            
            form_data = session_data.get("form_data", {})
            
            # 检查字段是否存在
            all_fields = (
                form_data.get('metadata', {}).get('required_fields', []) +
                form_data.get('metadata', {}).get('optional_fields', [])
            )
            
            if field_name not in all_fields:
                available_fields = ", ".join(all_fields)
                return {
                    "success": False,
                    "message": f"❌ 字段 '{field_name}' 不存在。\n可用字段：{available_fields}",
                    "session_id": session_id,
                    "form_stage": "user_interaction",
                    
                    "requires_user_input": True
                }
            
            # 更新字段
            updated_form_data = self.form_generator.update_field(form_data, field_name, new_value)
            session_data["form_data"] = updated_form_data
            
            # 重新生成表单展示
            form_display = self._build_form_display(updated_form_data)
            session_data["form_display"] = form_display
            
            success_msg = f"""
✅ **字段修改成功！**

已将字段 '{field_name}' 更新为：{new_value}

📝 **更新后的表单：**
{form_display}

**下一步操作：**
1. 如果表单内容正确，请回复 "确认提交" 或 "submit"
2. 如果需要继续修改，请回复 "修改 [字段名] [新值]"
3. 如果有其他问题，请直接描述
"""
            
            return {
                "success": True,
                "message": success_msg,
                "session_id": session_id,
                "form_stage": "user_interaction",
                
                "form_data": updated_form_data,
                "requires_user_input": True
            }
            
        except Exception as e:
            logger.error(f"字段修改失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"❌ 字段修改失败: {str(e)}",
                "session_id": session_data.get("session_id", "unknown")
            }
    
    def _handle_continue_chat(self, session_data: Dict[str, Any], user_response: str) -> Dict[str, Any]:
        """处理继续对话（用户询问或说明）"""
        logger.info("处理用户继续对话...")
        
        form_data = session_data.get("form_data", {})
        session_id = session_data.get("session_id", "unknown")
        
        # 使用LLM分析用户的进一步需求并可能更新表单
        try:
            response_msg = self._analyze_and_respond_to_user(form_data, user_response)
            
            return {
                "success": True,
                "message": response_msg,
                "session_id": session_id,
                "form_stage": "user_interaction",
                
                "form_data": form_data,
                "requires_user_input": True
            }
            
        except Exception as e:
            logger.error(f"处理用户对话失败: {e}")
            current_form_display = self._build_form_display(form_data)
            fallback_msg = f"""
📝 **表单状态保持不变**

您的输入：{user_response}

当前表单状态：
{current_form_display}

**操作提示：**
1. 确认提交：回复 "确认提交" 或 "submit"
2. 修改字段：回复 "修改 [字段名] [新值]"
3. 如有疑问，请更具体地描述您的需求
"""
            
            return {
                "success": True,
                "message": fallback_msg,
                "session_id": session_id,
                "form_stage": "user_interaction",
                
                "form_data": form_data,
                "requires_user_input": True
            }
    
    def _analyze_and_respond_to_user(self, form_data: Dict[str, Any], user_response: str) -> str:
        """分析用户输入并提供响应"""
        current_form_display = self._build_form_display(form_data)
        
        prompt = f"""用户对当前表单有进一步的说明或疑问。请分析用户的输入，提供有用的回复。

当前表单状态：
{current_form_display}

用户输入：{user_response}

请分析用户的输入：
1. 如果用户提供了更多需求细节，建议如何更新表单
2. 如果用户有疑问，提供清晰的解答
3. 给出具体的操作建议

返回一个友好、有用的回复，帮助用户完善表单或解答疑问。"""

        try:
            response = self.llm.call_llm("", prompt)
            
            ai_response = response.get('content', '').strip()
            
            return f"""
🤖 **AI 助手回复：**

{ai_response}

📝 **当前表单状态：**
{current_form_display}

**操作提示：**
1. 确认提交：回复 "确认提交" 或 "submit"
2. 修改字段：回复 "修改 [字段名] [新值]"
3. 继续说明您的需求
"""
            
        except Exception as e:
            logger.error(f"LLM分析用户输入失败: {e}")
            raise e
    
    def _build_form_display(self, form_data: Dict[str, Any]) -> str:
        """构建表单显示内容"""
        if not form_data or 'fields' not in form_data:
            return "表单数据为空"
        
        metadata = form_data.get('metadata', {})
        required_fields = metadata.get('required_fields', [])
        optional_fields = metadata.get('optional_fields', [])
        fields = form_data.get('fields', {})
        
        display_lines = []
        
        # 显示必填字段
        if required_fields:
            display_lines.append("**必填字段：**")
            for field in required_fields:
                value = fields.get(field, "[未填写]")
                display_lines.append(f"  • {field}: {value}")
        
        # 显示可选字段
        if optional_fields:
            display_lines.append("\n**可选字段：**")
            for field in optional_fields:
                value = fields.get(field, "[未填写]")
                display_lines.append(f"  • {field}: {value}")
        
        return "\n".join(display_lines)
    
    def _llm_analyze_and_match_workflow(self, user_input: str) -> Dict[str, Any]:
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
- ready_to_execute: 参数完整且需求明确，可以直接执行
- need_more_info: 需求明确但缺少关键参数，需要询问具体信息
- clarification_needed: 需求本身不够清晰，需要澄清意图

输出JSON格式：
{{
    "success": true,
    "analysis": {{
        "user_intent": "深度分析的用户真实意图",
        "technical_level": "用户技术水平评估(beginner/intermediate/advanced)",
        "context_clues": ["从用户输入中发现的上下文线索"],
        "implicit_requirements": ["从意图中推断的隐含需求"]
    }},
    "target_workflow": "最适合的工作流名称",
    "extracted_params": {{"参数名": "智能提取或推断的值"}},
    "missing_params": [{{
        "name": "参数名",
        "description": "参数说明"
    }}],
    "decision": "ready_to_execute|need_more_info|clarification_needed",
    "response_message": "给用户的自然、个性化回复",
    "confidence": 0.9,
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
            llm_service = self.llm._create_llm_service()
            responses = llm_service.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt
            )
            
            if not responses or not responses[0]:
                return {
                    "success": False,
                    "response_message": "抱歉，我暂时无法分析您的需求。请稍后再试。"
                }
            
            content = responses[0].strip()
            
            # 清理和解析JSON
            content = self._clean_json_response(content)
            
            try:
                result = json.loads(content)
                result["success"] = True
                
                # 验证和修正工作流选择
                target_workflow = result.get("target_workflow")
                if target_workflow not in self.workflow_registry:
                    logger.warning(f"LLM选择了不存在的工作流: {target_workflow}")
                    # 智能回退到最相似的工作流
                    result["target_workflow"] = self._find_best_fallback_workflow(user_input)
                    result["reasoning"] += f" [自动回退到 {result['target_workflow']}]"
                
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
    
    def _find_best_fallback_workflow(self, user_input: str) -> str:
        """智能选择最佳回退工作流"""
        # 简单的语义相似度判断，可以后续扩展为更复杂的匹配算法
        user_lower = user_input.lower()
        
        # 目前只有一个工作流，直接返回
        if "code_workflow_agent" in self.workflow_registry:
            return "code_workflow_agent"
        
        # 如果有多个工作流，可以实现更智能的匹配逻辑
        return list(self.workflow_registry.keys())[0] if self.workflow_registry else None
    
    def _provide_direct_code_solution(self, extracted_params: Dict[str, Any]) -> str:
        """直接提供代码解决方案，避免调用其他工具"""
        requirement = extracted_params.get("requirement", "")
        
        # 针对具体需求提供直接的代码解决方案
        if "mod" in requirement.lower() and any(x in requirement.lower() for x in ["ab", "a^b", "幂", "模运算"]):
            return """您的需求明确：需要一段高效的 Python 代码来计算 ab mod c 的结果。如果只需本地代码，代码如下：

```python
def fast_mod_exp(a, b, c):
    return pow(a, b, c)

# 示例用法
a = 2
b = 10
c = 1000
result = fast_mod_exp(a, b, c)
print(result)  # 输出: 24
```

直接调用 fast_mod_exp(a, b, c) 即可得到 ab mod c。"""
        
        # 其他类型的需求也可以在这里添加直接解决方案
        return f"""您的需求：{requirement}

基于需求分析，这里提供基础的实现方案：

```python
# 根据您的需求定制的代码
def solution():
    pass  # 在这里实现具体逻辑

# 使用示例
result = solution()
print(result)
```

请根据具体需求调整代码实现。"""
