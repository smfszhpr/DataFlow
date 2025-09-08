"""
Event Engine 配置管理模块
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from dataflow import get_logger

logger = get_logger()

@dataclass
class LLMConfig:
    """LLM配置"""
    api_key: str = ""
    api_url: str = "https://api.deepseek.com/v1/chat/completions"
    model: str = "deepseek-v3"
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 2000

@dataclass
class EngineConfig:
    """引擎配置"""
    max_queue_size: int = 1000
    check_interval: float = 0.1
    max_retry_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    max_workers: int = 5
    task_timeout: int = 300

@dataclass
class FormerConfig:
    """Former Agent配置"""
    max_history: int = 20
    session_timeout: int = 3600
    use_llm_detection: bool = True
    fallback_to_keywords: bool = True
    auto_generate_xml: bool = True
    validate_xml: bool = True

@dataclass
class EventEngineConfig:
    """Event Engine完整配置"""
    llm: LLMConfig
    engine: EngineConfig
    former: FormerConfig
    debug_mode: bool = False
    mock_llm: bool = False
    log_level: str = "INFO"

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # 默认配置文件路径
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> EventEngineConfig:
        """加载配置文件"""
        try:
            if not self.config_path.exists():
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                return self._get_default_config()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            # 解析配置
            llm_config = self._parse_llm_config(config_data.get('api', {}).get('llm', {}))
            engine_config = self._parse_engine_config(config_data.get('engine', {}))
            former_config = self._parse_former_config(config_data.get('former', {}))
            
            # 开发配置
            dev_config = config_data.get('development', {})
            debug_mode = dev_config.get('debug_mode', False)
            mock_llm = dev_config.get('mock_llm_responses', False)
            
            # 日志配置
            log_config = config_data.get('logging', {})
            log_level = log_config.get('level', 'INFO')
            
            return EventEngineConfig(
                llm=llm_config,
                engine=engine_config,
                former=former_config,
                debug_mode=debug_mode,
                mock_llm=mock_llm,
                log_level=log_level
            )
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _parse_llm_config(self, llm_data: Dict[str, Any]) -> LLMConfig:
        """解析LLM配置"""
        return LLMConfig(
            api_key=llm_data.get('api_key', ''),
            api_url=llm_data.get('api_url', 'https://api.deepseek.com/v1/chat/completions'),
            model=llm_data.get('model', 'deepseek-v3'),
            timeout=llm_data.get('timeout', 60),
            max_retries=llm_data.get('max_retries', 3),
            temperature=llm_data.get('temperature', 0.7),
            max_tokens=llm_data.get('max_tokens', 2000)
        )
    
    def _parse_engine_config(self, engine_data: Dict[str, Any]) -> EngineConfig:
        """解析引擎配置"""
        queue_config = engine_data.get('queue', {})
        retry_config = engine_data.get('retry', {})
        concurrency_config = engine_data.get('concurrency', {})
        
        return EngineConfig(
            max_queue_size=queue_config.get('max_size', 1000),
            check_interval=queue_config.get('check_interval', 0.1),
            max_retry_attempts=retry_config.get('max_attempts', 3),
            base_delay=retry_config.get('base_delay', 1.0),
            max_delay=retry_config.get('max_delay', 60.0),
            max_workers=concurrency_config.get('max_workers', 5),
            task_timeout=concurrency_config.get('timeout', 300)
        )
    
    def _parse_former_config(self, former_data: Dict[str, Any]) -> FormerConfig:
        """解析Former配置"""
        conversation_config = former_data.get('conversation', {})
        detection_config = former_data.get('form_detection', {})
        xml_config = former_data.get('xml_generation', {})
        
        return FormerConfig(
            max_history=conversation_config.get('max_history', 20),
            session_timeout=conversation_config.get('session_timeout', 3600),
            use_llm_detection=detection_config.get('use_llm', True),
            fallback_to_keywords=detection_config.get('fallback_to_keywords', True),
            auto_generate_xml=xml_config.get('auto_generate', True),
            validate_xml=xml_config.get('validate_on_generate', True)
        )
    
    def _get_default_config(self) -> EventEngineConfig:
        """获取默认配置"""
        return EventEngineConfig(
            llm=LLMConfig(),
            engine=EngineConfig(),
            former=FormerConfig()
        )
    
    def get_llm_config(self) -> LLMConfig:
        """获取LLM配置"""
        return self.config.llm
    
    def get_engine_config(self) -> EngineConfig:
        """获取引擎配置"""
        return self.config.engine
    
    def get_former_config(self) -> FormerConfig:
        """获取Former配置"""
        return self.config.former
    
    def is_debug_mode(self) -> bool:
        """是否调试模式"""
        return self.config.debug_mode
    
    def should_mock_llm(self) -> bool:
        """是否模拟LLM响应"""
        return self.config.mock_llm
    
    def reload_config(self):
        """重新加载配置"""
        self.config = self._load_config()
        logger.info("配置已重新加载")
    
    def save_config(self, config_dict: Dict[str, Any]):
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置已保存到: {self.config_path}")
            self.reload_config()
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def update_llm_config(self, api_key: str = None, api_url: str = None, model: str = None):
        """更新LLM配置"""
        config_dict = self._config_to_dict()
        
        if api_key is not None:
            config_dict['api']['llm']['api_key'] = api_key
        if api_url is not None:
            config_dict['api']['llm']['api_url'] = api_url
        if model is not None:
            config_dict['api']['llm']['model'] = model
        
        self.save_config(config_dict)
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            'api': {
                'llm': {
                    'api_key': self.config.llm.api_key,
                    'api_url': self.config.llm.api_url,
                    'model': self.config.llm.model,
                    'timeout': self.config.llm.timeout,
                    'max_retries': self.config.llm.max_retries,
                    'temperature': self.config.llm.temperature,
                    'max_tokens': self.config.llm.max_tokens
                }
            },
            'engine': {
                'queue': {
                    'max_size': self.config.engine.max_queue_size,
                    'check_interval': self.config.engine.check_interval
                },
                'retry': {
                    'max_attempts': self.config.engine.max_retry_attempts,
                    'base_delay': self.config.engine.base_delay,
                    'max_delay': self.config.engine.max_delay
                },
                'concurrency': {
                    'max_workers': self.config.engine.max_workers,
                    'timeout': self.config.engine.task_timeout
                }
            },
            'former': {
                'conversation': {
                    'max_history': self.config.former.max_history,
                    'session_timeout': self.config.former.session_timeout
                },
                'form_detection': {
                    'use_llm': self.config.former.use_llm_detection,
                    'fallback_to_keywords': self.config.former.fallback_to_keywords
                },
                'xml_generation': {
                    'auto_generate': self.config.former.auto_generate_xml,
                    'validate_on_generate': self.config.former.validate_xml
                }
            },
            'development': {
                'debug_mode': self.config.debug_mode,
                'mock_llm_responses': self.config.mock_llm
            },
            'logging': {
                'level': self.config.log_level
            }
        }

# 全局配置管理器实例
config_manager = ConfigManager()

def get_config() -> EventEngineConfig:
    """获取全局配置"""
    return config_manager.config

def get_llm_config() -> LLMConfig:
    """获取LLM配置"""
    return config_manager.get_llm_config()

def get_engine_config() -> EngineConfig:
    """获取引擎配置"""
    return config_manager.get_engine_config()

def get_former_config() -> FormerConfig:
    """获取Former配置"""
    return config_manager.get_former_config()

def reload_config():
    """重新加载配置"""
    config_manager.reload_config()
