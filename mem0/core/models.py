"""
Mem0 核心数据模型模块

本模块定义记忆系统中使用的核心数据模型：
1. Conversation - 对话记录模型
2. Memory - 记忆模型（核心）
3. MemoryType - 记忆类型枚举
4. MemorySearchResult - 记忆搜索结果模型

这些模型用于：
- 数据序列化和反序列化
- 类型检查和验证
- 数据库存储格式定义

使用示例:
    from mem0.core.models import Conversation, Memory, MemoryType

    # 创建对话
    conversation = Conversation(role="user", content="我叫张三")

    # 创建记忆
    memory = Memory(
        content="用户名字叫张三",
        memory_type=MemoryType.SEMANTIC,
        user_id="user_001",
        entities=[{"name": "张三", "type": "Person"}]
    )
"""

import json
from datetime import datetime
from typing import Any, List
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class MemoryType(Enum):
    """
    记忆类型枚举

    定义不同类型的记忆，用于分类和检索：
    - EPISODIC: 情景记忆 - 特定事件或经历
    - SEMANTIC: 语义记忆 - 事实性知识
    - PROCEDURAL: 程序记忆 - 技能和操作
    """

    EPISODIC = "episodic"  # 情景记忆：特定事件或经历
    SEMANTIC = "semantic"  # 语义记忆：事实性知识
    PROCEDURAL = "procedural"  # 程序记忆：技能和操作


class Conversation:
    """
    对话记录模型

    表示单条对话记录，用于临时缓冲和记忆来源追溯。

    Attributes:
        id: 唯一标识符（UUID）
        role: 对话角色（"user" 或 "assistant"）
        content: 对话内容
        timestamp: 创建时间戳
        metadata: 额外元数据字典
    """

    def __init__(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        初始化对话记录

        Args:
            role: 对话角色（"user" 或 "assistant"）
            content: 对话内容
            metadata: 额外元数据（可选）
            id: 唯一标识符（可选，默认自动生成 UUID）
            timestamp: 创建时间（可选，默认当前时间）
        """
        self.id = id or str(uuid4())  # 生成唯一ID
        self.role = role  # 对话角色
        self.content = content  # 对话内容
        self.timestamp = timestamp or datetime.now()  # 创建时间
        self.metadata = metadata or {}  # 元数据字典

    def to_dict(self) -> Dict[str, Any]:
        """
        将对话对象转换为字典

        Returns:
            包含所有字段的字典，时间戳转换为 ISO 格式字符串
        """
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """
        从字典创建对话对象

        Args:
            data: 包含对话数据的字典

        Returns:
            Conversation 实例
        """
        return cls(
            id=data.get("id"),
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


class Memory:
    """
    记忆模型 - Mem0 系统的核心数据结构

    表示一条完整的记忆，包含内容、实体、关系等所有信息。
    记忆同时存储在向量库（用于语义搜索）和图数据库（用于实体关系查询）中。

    Attributes:
        id: 唯一标识符（UUID）
        content: 记忆内容文本
        memory_type: 记忆类型（EPISODIC/SEMANTIC/PROCEDURAL）
        user_id: 所属用户ID
        session_id: 所属会话ID（可选）
        vector: 向量表示（用于相似度搜索）
        entities: 实体列表，每个实体包含 name, type, description
        relations: 关系列表，每个关系包含 source, target, type, description
        source_conversations: 来源对话ID列表
        importance: 重要性分数（0-1）
        access_count: 访问次数
        last_accessed: 最后访问时间
        created_at: 创建时间
        updated_at: 更新时间
    """

    def __init__(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        user_id: str = "default",
        session_id: Optional[str] = None,
        vector: Optional[List[float]] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        relations: Optional[List[Dict[str, Any]]] = None,
        source_conversations: Optional[List[str]] = None,
        importance: float = 0.5,
        access_count: int = 0,
        last_accessed: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        id: Optional[str] = None,
    ):
        """
        初始化记忆对象

        Args:
            content: 记忆内容文本
            memory_type: 记忆类型（默认 EPISODIC）
            user_id: 用户ID（默认 "default"）
            session_id: 会话ID（可选）
            vector: 向量表示（可选，通常由嵌入模型生成）
            entities: 实体列表（可选）
            relations: 关系列表（可选）
            source_conversations: 来源对话ID列表（可选）
            importance: 重要性分数（默认 0.5）
            access_count: 访问次数（默认 0）
            last_accessed: 最后访问时间（可选）
            created_at: 创建时间（可选，默认当前时间）
            updated_at: 更新时间（可选，默认当前时间）
            id: 唯一标识符（可选，默认自动生成 UUID）
        """
        self.id = id or str(uuid4())  # 唯一标识符
        self.content = content  # 记忆内容
        self.memory_type = memory_type  # 记忆类型
        self.user_id = user_id  # 用户ID
        self.session_id = session_id  # 会话ID
        self.vector = vector  # 向量表示
        self.entities = entities or []  # 实体列表
        self.relations = relations or []  # 关系列表
        self.source_conversations = source_conversations or []  # 来源对话
        self.importance = importance  # 重要性分数
        self.access_count = access_count  # 访问次数
        self.last_accessed = last_accessed  # 最后访问时间
        self.created_at = created_at or datetime.now()  # 创建时间
        self.updated_at = updated_at or datetime.now()  # 更新时间

    def to_dict(self) -> Dict[str, Any]:
        """
        将记忆对象转换为字典

        用于：
        - 存储到向量库（作为 metadata）
        - JSON 序列化
        - API 响应

        Returns:
            包含所有字段的字典
        """
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "vector": self.vector,
            "entities": self.entities,
            "relations": self.relations,
            "source_conversations": self.source_conversations,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """
        从字典创建记忆对象

        用于：
        - 从向量库读取（metadata 转对象）
        - JSON 反序列化
        - API 请求解析

        注意：当从向量库读取时，entities、relations、source_conversations 可能是 JSON 字符串，
        需要解析为 Python 对象。

        Args:
            data: 包含记忆数据的字典

        Returns:
            Memory 实例
        """
        # 辅助函数：解析可能为 JSON 字符串的字段
        def parse_json_field(value: Any, default: List = None) -> List:
            if default is None:
                default = []
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    return parsed if isinstance(parsed, list) else default
                except (json.JSONDecodeError, TypeError):
                    return default
            elif isinstance(value, list):
                return value
            return default

        return cls(
            id=data.get("id"),
            content=data["content"],
            memory_type=MemoryType(data.get("memory_type", "episodic")),
            user_id=data.get("user_id", "default"),
            session_id=data.get("session_id"),
            vector=data.get("vector"),
            entities=parse_json_field(data.get("entities"), []),
            relations=parse_json_field(data.get("relations"), []),
            source_conversations=parse_json_field(data.get("source_conversations"), []),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"])
                if data.get("last_accessed")
                else None
            ),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.now().isoformat())
            ),
            updated_at=datetime.fromisoformat(
                data.get("updated_at", datetime.now().isoformat())
            ),
        )


class MemorySearchResult:
    """
    记忆搜索结果模型

    表示一次记忆搜索的结果，包含记忆对象、匹配分数和匹配类型。

    Attributes:
        memory: 记忆对象
        score: 匹配分数（0-1，越高越相关）
        match_type: 匹配类型（"vector" 向量匹配 / "graph" 图匹配 / "hybrid" 混合匹配）
    """

    def __init__(
        self,
        memory: Memory,
        score: float,
        match_type: str = "vector",
    ):
        """
        初始化搜索结果

        Args:
            memory: 记忆对象
            score: 匹配分数（0-1）
            match_type: 匹配类型（默认 "vector"）
        """
        self.memory = memory  # 记忆对象
        self.score = score  # 匹配分数
        self.match_type = match_type  # 匹配类型

    def to_dict(self) -> Dict[str, Any]:
        """
        将搜索结果转换为字典

        Returns:
            包含记忆、分数和匹配类型的字典
        """
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "match_type": self.match_type,
        }
