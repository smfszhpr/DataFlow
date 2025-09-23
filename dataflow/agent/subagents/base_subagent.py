#!/usr/bin/env python3
"""
Base SubAgent 基类
所有 SubAgent 都继承这个基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BaseSubAgent(ABC):
    """
    SubAgent 基类，定义通用接口
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.state = {}
        
    @abstractmethod
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """
        处理输入数据，返回结果
        
        Args:
            input_data: 输入数据
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return self.state.copy()
    
    def set_state(self, state: Dict[str, Any]):
        """设置状态"""
        self.state.update(state)
    
    def reset(self):
        """重置状态"""
        self.state = {}
        
    def log_info(self, message: str):
        """记录信息日志"""
        logger.info(f"[{self.name}] {message}")
        
    def log_error(self, message: str):
        """记录错误日志"""
        logger.error(f"[{self.name}] {message}")
        
    def log_debug(self, message: str):
        """记录调试日志"""
        logger.debug(f"[{self.name}] {message}")
