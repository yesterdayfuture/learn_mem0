"""
Mem0 - 智能记忆存储系统

一个支持向量检索和图数据库的记忆存储服务，
可以自动根据对话历史进行总结、更新和存储。

支持动态加载不同的数据库和模型厂商。
"""

__version__ = "2.0.0"

from .client import Mem0Client, create_client
from .config import Mem0Config, registry, PluginRegistry
from .core.memory import MemoryManager
from .core.models import Conversation, Memory, MemoryType

__all__ = [
    "MemoryManager",
    "Conversation",
    "Memory",
    "MemoryType",
    "Mem0Client",
    "create_client",
    "Mem0Config",
    "registry",
    "PluginRegistry",
]
