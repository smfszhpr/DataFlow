"""
Former Agent V2 MCP 适配器
将 SubAgent 模式的 Former Agent V2 适配到 MCP 协议接口
"""
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from .agent import FormerAgentV2


class FormerAgentMCPV2:
    """Former Agent V2 的 MCP 适配器"""
    
    def __init__(self):
        self.agent = FormerAgentV2()
        self.sessions = {}  # 会话管理
        
    def create_session(self, session_id: str = None) -> str:
        """创建新会话"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.sessions[session_id] = {
            'created_at': datetime.now().isoformat(),
            'last_request': None,
            'history': []
        }
        
        return session_id
    
    def analyze_requirement(self, session_id: str, user_requirement: str, 
                          context: str = "") -> Dict[str, Any]:
        """MCP接口：分析需求"""
        try:
            # 确保会话存在
            if session_id not in self.sessions:
                session_id = self.create_session(session_id)
            
            # 记录请求
            request_data = {
                'action': 'analyze_requirement',
                'timestamp': datetime.now().isoformat(),
                'user_requirement': user_requirement,
                'context': context
            }
            self.sessions[session_id]['last_request'] = request_data
            self.sessions[session_id]['history'].append(request_data)
            
            # 执行分析（只到分析阶段）
            result = self.agent.process_request(user_requirement, context)
            
            # 提取分析结果
            analysis_result = {
                'success': True,
                'session_id': session_id,
                'form_type': result.get('form_type'),
                'confidence': result.get('confidence', 0.0),
                'reasoning': self._extract_reasoning_from_history(result.get('processing_history', [])),
                'suggested_template': result.get('form_type', '').lower().replace(' ', '_'),
                'timestamp': datetime.now().isoformat()
            }
            
            # 记录结果
            self.sessions[session_id]['analysis_result'] = analysis_result
            
            return {
                'status': 'success',
                'data': analysis_result,
                'message': '需求分析完成'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'data': None,
                'message': f'需求分析失败: {str(e)}'
            }
    
    def check_fields(self, session_id: str, extracted_fields: Dict[str, Any] = None) -> Dict[str, Any]:
        """MCP接口：检查字段完整性"""
        try:
            if session_id not in self.sessions:
                return {
                    'status': 'error',
                    'data': None,
                    'message': '会话不存在，请先调用analyze_requirement'
                }
            
            session = self.sessions[session_id]
            last_request = session.get('last_request')
            
            if not last_request:
                return {
                    'status': 'error',
                    'data': None,
                    'message': '会话中没有找到之前的请求记录'
                }
            
            # 获取分析结果
            analysis_result = session.get('analysis_result', {})
            form_type = analysis_result.get('form_type', '通用表单')
            
            # 重新执行完整流程以获取字段验证结果
            result = self.agent.process_request(
                last_request.get('user_requirement', ''),
                last_request.get('context', '')
            )
            
            # 构建字段检查结果
            field_check_result = {
                'success': True,
                'session_id': session_id,
                'form_type': form_type,
                'is_complete': result.get('is_complete', False),
                'missing_fields': result.get('missing_fields', []),
                'validated_fields': self._extract_validated_fields(result.get('processing_history', [])),
                'suggestions': [f"请提供{field}的详细信息" for field in result.get('missing_fields', [])],
                'timestamp': datetime.now().isoformat()
            }
            
            # 记录结果
            session['field_check_result'] = field_check_result
            
            return {
                'status': 'success',
                'data': field_check_result,
                'message': '字段检查完成'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'data': None,
                'message': f'字段检查失败: {str(e)}'
            }
    
    def generate_xml(self, session_id: str, final_fields: Dict[str, Any] = None) -> Dict[str, Any]:
        """MCP接口：生成XML表单"""
        try:
            if session_id not in self.sessions:
                return {
                    'status': 'error',
                    'data': None,
                    'message': '会话不存在，请先调用analyze_requirement'
                }
            
            session = self.sessions[session_id]
            last_request = session.get('last_request')
            
            if not last_request:
                return {
                    'status': 'error',
                    'data': None,
                    'message': '会话中没有找到之前的请求记录'
                }
            
            # 执行完整流程
            result = self.agent.process_request(
                last_request.get('user_requirement', ''),
                last_request.get('context', '')
            )
            
            if result.get('success') and result.get('xml_content'):
                xml_result = {
                    'success': True,
                    'session_id': session_id,
                    'xml_content': result['xml_content'],
                    'form_type': result.get('form_type'),
                    'field_count': len(self._extract_validated_fields(result.get('processing_history', []))),
                    'template_used': result.get('form_type', '').lower().replace(' ', '_'),
                    'generation_timestamp': datetime.now().isoformat(),
                    'confidence': result.get('confidence', 0.0)
                }
                
                # 记录结果
                session['xml_result'] = xml_result
                
                return {
                    'status': 'success',
                    'data': xml_result,
                    'message': 'XML生成完成'
                }
            else:
                return {
                    'status': 'error',
                    'data': None,
                    'message': f'XML生成失败: {result.get("error_message", "未知错误")}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'data': None,
                'message': f'XML生成失败: {str(e)}'
            }
    
    def get_current_form(self, session_id: str) -> Dict[str, Any]:
        """MCP接口：获取当前表单状态"""
        try:
            if session_id not in self.sessions:
                return {
                    'status': 'error',
                    'data': None,
                    'message': '会话不存在'
                }
            
            session = self.sessions[session_id]
            
            # 获取最新状态
            current_form = {
                'session_id': session_id,
                'created_at': session.get('created_at'),
                'analysis_result': session.get('analysis_result'),
                'field_check_result': session.get('field_check_result'),
                'xml_result': session.get('xml_result'),
                'history_count': len(session.get('history', [])),
                'last_activity': session.get('last_request', {}).get('timestamp')
            }
            
            return {
                'status': 'success',
                'data': current_form,
                'message': '获取表单状态成功'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'data': None,
                'message': f'获取表单状态失败: {str(e)}'
            }
    
    def _extract_reasoning_from_history(self, history: List[Dict]) -> str:
        """从处理历史中提取推理过程"""
        for step in history:
            if step.get('step') == 'analyze_requirement' and 'result' in step:
                return step['result'].get('reasoning', '基于需求分析的结果')
        return '基于需求分析的结果'
    
    def _extract_validated_fields(self, history: List[Dict]) -> Dict[str, Any]:
        """从处理历史中提取验证的字段"""
        for step in history:
            if step.get('step') == 'validate_fields' and 'result' in step:
                return step['result'].get('validated_fields', {})
        return {}
    
    def get_session_list(self) -> List[str]:
        """获取所有会话ID"""
        return list(self.sessions.keys())
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
