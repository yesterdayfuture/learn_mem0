from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """记忆类型"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class AddMessageRequest(BaseModel):
    """添加消息请求"""
    user_id: str = Field(default="default", description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    role: str = Field(..., description="角色: user, assistant, system")
    content: str = Field(..., description="消息内容")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="额外元数据")


class AddMessageResponse(BaseModel):
    """添加消息响应"""
    success: bool
    message_id: str
    memory_created: bool = False
    memory_id: Optional[str] = None


class SearchRequest(BaseModel):
    """搜索记忆请求"""
    query: str = Field(..., description="搜索查询")
    user_id: str = Field(default="default", description="用户ID")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    memory_type: Optional[MemoryType] = Field(default=None, description="记忆类型过滤")


class MemoryResult(BaseModel):
    """记忆结果"""
    id: str
    content: str
    memory_type: str
    importance: float
    score: float
    match_type: str
    created_at: str
    entities: List[Dict[str, Any]] = []


class SearchResponse(BaseModel):
    """搜索响应"""
    memories: List[MemoryResult]
    total: int


class GetContextRequest(BaseModel):
    """获取上下文请求"""
    query: str = Field(..., description="查询内容")
    user_id: str = Field(default="default", description="用户ID")
    max_tokens: int = Field(default=2000, description="最大token数")


class GetContextResponse(BaseModel):
    """获取上下文响应"""
    context: str
    memory_count: int


class ChatRequest(BaseModel):
    """聊天请求（带记忆）"""
    user_id: str = Field(default="default", description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    message: str = Field(..., description="用户消息")
    include_memories: bool = Field(default=True, description="是否包含相关记忆")
    stream: bool = Field(default=False, description="是否流式返回")


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    memories_used: List[str] = []


class MemoryDetail(BaseModel):
    """记忆详情"""
    id: str
    content: str
    memory_type: str
    user_id: str
    session_id: Optional[str]
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    importance: float
    access_count: int
    created_at: str
    updated_at: str


class DeleteMemoryResponse(BaseModel):
    """删除记忆响应"""
    success: bool
    message: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    vector_store: bool
    graph_db: bool
    embedding_model: bool
    llm_model: bool


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    summary_threshold: Optional[int] = None
    max_conversation_history: Optional[int] = None
    similarity_threshold: Optional[float] = None
    decay_factor: Optional[float] = None
    max_memories_per_query: Optional[int] = None
