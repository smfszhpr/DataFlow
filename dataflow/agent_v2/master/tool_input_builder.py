"""
工具输入参数统一构建器
解决 former 等工具的特判问题，通过声明式配置自动从不同来源拼装参数
"""
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# 工具参数来源规范
# 每个参数按优先级提供多个来源：llm_args / state / const / user_input
# 优先级 = 列表顺序，找到第一个非空值就使用
ToolSpecs: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    "former": {
        "action": [
            ("llm_args", "action"),  # 优先使用LLM指定的action
            ("const", "collect_user_response"),  # 默认为收集用户响应
        ],
        "user_query": [
            ("llm_args", "user_query"),  # LLM可能指定特殊查询
            ("user_input", "raw"),  # 通常使用原始用户输入
        ],
        "session_id": [
            ("llm_args", "session_id"),  # LLM可能指定session_id
            ("state", "form_session.session_id"),  # 从表单状态获取
        ],
        "form_data": [
            ("llm_args", "form_data"),  # LLM可能提供表单数据
            ("state", "form_session.form_data.fields"),  # 从表单状态获取fields
            ("state", "form_session.form_data"),  # 从表单状态获取整个form_data
            ("const", "{}"),  # 默认为空字典
        ],
        "user_response": [
            ("llm_args", "user_response"),  # LLM可能指定用户响应
        ],
    },

    # 后续可以添加其他工具的参数规范
}


def _get_from_state(state: Dict[str, Any], path: str) -> Any:
    """从state中按路径获取值，支持点分割的嵌套路径"""
    try:
        current = state
        for part in path.split("."):
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
        return current
    except Exception as e:
        logger.debug(f"从state路径 '{path}' 获取值失败: {e}")
        return None


def build_tool_input(
    tool_name: str, 
    llm_args: Optional[Dict[str, Any]], 
    state: Dict[str, Any], 
    user_input: str
) -> Dict[str, Any]:
    """
    统一构建工具输入参数
    
    Args:
        tool_name: 工具名称
        llm_args: LLM决策返回的参数
        state: Agent状态数据
        user_input: 用户原始输入
        
    Returns:
        构建完成的工具输入参数
    """
    spec = ToolSpecs.get(tool_name)
    
    if spec:
        # 有配置规范的工具，按规范构建
        final_input: Dict[str, Any] = {}
        
        for param_name, sources in spec.items():
            value = None
            used_source = None
            
            # 按优先级尝试各个来源
            for source_type, source_key in sources:
                if source_type == "llm_args":
                    value = (llm_args or {}).get(source_key)
                    used_source = f"llm_args.{source_key}"
                    
                elif source_type == "state":
                    value = _get_from_state(state, source_key)
                    used_source = f"state.{source_key}"
                    
                elif source_type == "const":
                    # 特殊处理常量值
                    if source_key == "{}":
                        value = {}
                    else:
                        value = source_key
                    used_source = f"const:{source_key}"
                    
                elif source_type == "user_input":
                    if source_key == "raw":
                        value = user_input
                    used_source = "user_input"
                
                # 找到第一个非空值就停止
                if value is not None:
                    break
            
            # 记录参数来源，便于调试
            if value is not None:
                logger.debug(f"工具 {tool_name} 参数 {param_name} = {value} (来源: {used_source})")
            else:
                logger.debug(f"工具 {tool_name} 参数 {param_name} = None (所有来源均为空)")
                
            final_input[param_name] = value
            
    else:
        # 🎯 没有配置规范的工具，使用动态表单字段传递策略
        logger.debug(f"工具 {tool_name} 没有参数规范，使用动态表单字段传递")
        final_input: Dict[str, Any] = {}
        
        # 1. 首先包含LLM参数
        if llm_args:
            final_input.update(llm_args)
            logger.debug(f"包含LLM参数: {list(llm_args.keys())}")
        
        # 2. 然后添加表单字段（优先级高于LLM参数）
        form_fields = _get_from_state(state, "form_session.form_data.fields")
        if form_fields and isinstance(form_fields, dict):
            # 只包含非空字段
            for field_name, field_value in form_fields.items():
                if field_value is not None and field_value != "":
                    final_input[field_name] = field_value
                    logger.debug(f"添加表单字段: {field_name} = {field_value}")
            logger.debug(f"动态添加表单字段: {list(form_fields.keys())}")
    
    logger.info(f"为工具 {tool_name} 构建参数: {list(final_input.keys())}")
    return final_input


def register_tool_spec(tool_name: str, param_specs: Dict[str, List[Tuple[str, str]]]):
    """
    注册新工具的参数规范
    
    Args:
        tool_name: 工具名称
        param_specs: 参数规范字典
    """
    ToolSpecs[tool_name] = param_specs
    logger.info(f"已注册工具 {tool_name} 的参数规范")


def get_tool_spec(tool_name: str) -> Optional[Dict[str, List[Tuple[str, str]]]]:
    """获取工具的参数规范"""
    return ToolSpecs.get(tool_name)


def list_supported_tools() -> List[str]:
    """列出所有已配置参数规范的工具"""
    return list(ToolSpecs.keys())


def create_unified_action(
    tool_name: str,
    llm_args: Optional[Dict[str, Any]],
    state: Dict[str, Any],
    user_input: str,
    log_message: str = ""
):
    """
    统一创建 LangChain AgentAction，自动构建工具参数
    
    Args:
        tool_name: 工具名称
        llm_args: LLM提供的参数（可以是最小参数集）
        state: Agent状态
        user_input: 用户输入
        log_message: 日志信息
        
    Returns:
        构建好参数的 AgentAction
    """
    # 导入放在函数内部避免循环导入
    from langchain_core.agents import AgentAction as LCAgentAction
    
    # 统一构建工具参数
    unified_input = build_tool_input(tool_name, llm_args, state, user_input)
    
    # 创建 Action
    action = LCAgentAction(
        tool=tool_name,
        tool_input=unified_input,
        log=log_message or f"统一构建: {tool_name}"
    )
    
    logger.info(f"创建统一Action: {tool_name} with params {list(unified_input.keys())}")
    return action
