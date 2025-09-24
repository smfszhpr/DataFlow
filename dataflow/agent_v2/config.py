"""
Agent V2 配置管理模块
独立于旧版本agent的配置管理
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
class AgentV2Config:
    """Agent V2完整配置"""
    llm: LLMConfig
    debug_mode: bool = False
    log_level: str = "INFO"
    max_steps: int = 20
    websocket_port: int = 8765

class ConfigManager:
    """Agent V2配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # 默认从agent_v2目录下的config.yaml加载
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> AgentV2Config:
        """加载配置文件"""
        try:
            if not self.config_path.exists():
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                return self._get_default_config()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            # 解析LLM配置（兼容旧格式）
            llm_data = config_data.get('llm', {})
            # 如果是旧格式，从 api.llm 路径读取
            if not llm_data and config_data.get('api', {}).get('llm'):
                llm_data = config_data['api']['llm']
            
            llm_config = LLMConfig(
                api_key=llm_data.get('api_key', ''),
                api_url=llm_data.get('api_url', 'https://api.deepseek.com/v1/chat/completions'),
                model=llm_data.get('model', 'deepseek-v3'),
                timeout=llm_data.get('timeout', 60),
                max_retries=llm_data.get('max_retries', 3),
                temperature=llm_data.get('temperature', 0.7),
                max_tokens=llm_data.get('max_tokens', 2000)
            )
            
            # 解析其他配置（兼容旧格式）
            debug_mode = config_data.get('debug_mode', False)
            # 尝试从development.debug_mode读取
            if not debug_mode and config_data.get('development', {}).get('debug_mode'):
                debug_mode = config_data['development']['debug_mode']
            
            log_level = config_data.get('log_level', 'INFO')
            # 尝试从logging.level读取
            if config_data.get('logging', {}).get('level'):
                log_level = config_data['logging']['level']
            
            max_steps = config_data.get('max_steps', 20)
            websocket_port = config_data.get('websocket_port', 8765)
            
            return AgentV2Config(
                llm=llm_config,
                debug_mode=debug_mode,
                log_level=log_level,
                max_steps=max_steps,
                websocket_port=websocket_port
            )
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> AgentV2Config:
        """获取默认配置"""
        return AgentV2Config(
            llm=LLMConfig()
        )
    
    def get_llm_config(self) -> LLMConfig:
        """获取LLM配置"""
        return self.config.llm
    
    def is_debug_mode(self) -> bool:
        """是否调试模式"""
        return self.config.debug_mode
    
    def get_max_steps(self) -> int:
        """获取最大步骤数"""
        return self.config.max_steps
    
    def get_websocket_port(self) -> int:
        """获取WebSocket端口"""
        return self.config.websocket_port
    
    def reload_config(self):
        """重新加载配置"""
        self.config = self._load_config()
        logger.info("Agent V2配置已重新加载")

# 全局配置管理器实例
config_manager = ConfigManager()

def get_config() -> AgentV2Config:
    """获取全局配置"""
    return config_manager.config

def get_llm_config() -> LLMConfig:
    """获取LLM配置"""
    return config_manager.get_llm_config()

def reload_config():
    """重新加载配置"""
    config_manager.reload_config()
