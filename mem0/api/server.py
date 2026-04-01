"""
Mem0 FastAPI 服务模块

本模块提供基于 FastAPI 的 RESTful API 服务，用于记忆系统的 HTTP 接口暴露。

提供的 API 端点：
- POST /api/v1/conversations - 添加对话
- POST /api/v1/conversations/batch - 批量添加对话
- POST /api/v1/memories/search - 搜索记忆
- GET /api/v1/memories/{memory_id} - 获取单个记忆
- DELETE /api/v1/memories/{memory_id} - 删除记忆
- GET /api/v1/users/{user_id}/memories - 获取用户所有记忆
- POST /api/v1/memories/consolidate - 整合用户记忆
- POST /api/v1/context - 获取相关上下文
- GET /health - 健康检查

依赖安装:
    pip install fastapi uvicorn

启动服务:
    python main.py
    # 或
    uvicorn mem0.api.server:app --host 0.0.0.0 --port 8000

使用示例:
    import requests

    # 添加对话
    response = requests.post("http://localhost:8000/api/v1/conversations", json={
        "user_id": "user_001",
        "role": "user",
        "content": "我叫张三"
    })

    # 搜索记忆
    response = requests.post("http://localhost:8000/api/v1/memories/search", json={
        "query": "用户的名字",
        "user_id": "user_001"
    })
    memories = response.json()["memories"]
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..client import Mem0Client

# 创建 FastAPI 应用实例
app = FastAPI(
    title="Mem0 API",
    description="记忆存储系统 API - 支持自动总结、向量存储和图数据库存储",
    version="1.0.0",
)

# 全局客户端实例（在启动时初始化）
client: Optional[Mem0Client] = None


# ==================== Pydantic 模型 ====================

class ConversationRequest(BaseModel):
    """
    对话请求模型

    Attributes:
        user_id: 用户唯一标识
        role: 对话角色（user/assistant）
        content: 对话内容
        session_id: 会话ID（可选）
    """
    user_id: str = Field(..., description="用户唯一标识")
    role: str = Field(..., description="对话角色：user 或 assistant")
    content: str = Field(..., description="对话内容")
    session_id: Optional[str] = Field(None, description="会话ID")


class BatchConversationRequest(BaseModel):
    """
    批量对话请求模型

    Attributes:
        user_id: 用户唯一标识
        conversations: 对话列表
        session_id: 会话ID（可选）
    """
    user_id: str = Field(..., description="用户唯一标识")
    conversations: List[Dict[str, Any]] = Field(..., description="对话列表")
    session_id: Optional[str] = Field(None, description="会话ID")


class MemorySearchRequest(BaseModel):
    """
    记忆搜索请求模型

    Attributes:
        query: 搜索查询文本
        user_id: 用户ID（默认 default）
        top_k: 返回结果数量（默认 5）
        memory_type: 记忆类型过滤（可选）
    """
    query: str = Field(..., description="搜索查询文本")
    user_id: str = Field("default", description="用户ID")
    top_k: int = Field(5, description="返回结果数量", ge=1, le=50)
    memory_type: Optional[str] = Field(None, description="记忆类型过滤")


class ContextRequest(BaseModel):
    """
    上下文请求模型

    Attributes:
        query: 查询文本
        user_id: 用户ID（默认 default）
        max_tokens: 最大 token 数（默认 2000）
    """
    query: str = Field(..., description="查询文本")
    user_id: str = Field("default", description="用户ID")
    max_tokens: int = Field(2000, description="最大 token 数")


class MemoryResponse(BaseModel):
    """
    记忆响应模型

    Attributes:
        id: 记忆ID
        content: 记忆内容
        memory_type: 记忆类型
        user_id: 用户ID
        entities: 实体列表
        relations: 关系列表
        importance: 重要性分数
        created_at: 创建时间
    """
    id: str
    content: str
    memory_type: str
    user_id: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    importance: float
    created_at: str


class SearchResponse(BaseModel):
    """
    搜索响应模型

    Attributes:
        memories: 记忆列表
        total: 总数
    """
    memories: List[Dict[str, Any]]
    total: int


# ==================== 生命周期事件 ====================

@app.on_event("startup")
async def startup_event():
    """
    应用启动事件

    初始化 Mem0Client 客户端，建立数据库连接。
    """
    global client
    try:
        client = await Mem0Client.create()
        print("✅ Mem0 client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Mem0 client: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭事件

    关闭数据库连接，释放资源。
    """
    global client
    if client:
        await client.close()
        print("✅ Mem0 client closed")


# ==================== API 端点 ====================

@app.get("/health")
async def health_check():
    """
    健康检查端点

    检查服务是否正常运行。

    Returns:
        健康状态信息
    """
    if not client:
        raise HTTPException(status_code=503, detail="Client not initialized")

    # 检查各组件健康状态
    vector_healthy = await client._vector_store.health_check() if client._vector_store else False
    graph_healthy = await client._graph_db.health_check() if client._graph_db else False

    return {
        "status": "healthy" if (vector_healthy and graph_healthy) else "degraded",
        "vector_store": "healthy" if vector_healthy else "unhealthy",
        "graph_db": "healthy" if graph_healthy else "unhealthy",
    }


@app.post("/api/v1/conversations")
async def add_conversation(request: ConversationRequest):
    """
    添加对话

    添加单条对话记录，触发自动总结。

    Args:
        request: 对话请求

    Returns:
        创建的记忆信息
    """
    if not client:
        raise HTTPException(status_code=503, detail="Client not initialized")

    try:
        memory = await client.add_conversation(
            user_id=request.user_id,
            role=request.role,
            content=request.content,
            session_id=request.session_id,
        )

        if memory:
            return {
                "success": True,
                "memory_created": True,
                "memory": memory.to_dict(),
            }
        else:
            return {
                "success": True,
                "memory_created": False,
                "message": "对话已添加，等待总结",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/conversations/batch")
async def add_conversations_batch(request: BatchConversationRequest):
    """
    批量添加对话

    批量添加对话记录，并强制触发总结。

    Args:
        request: 批量对话请求

    Returns:
        创建的记忆信息
    """
    if not client:
        raise HTTPException(status_code=503, detail="Client not initialized")

    try:
        memory = await client.add_conversations(
            user_id=request.user_id,
            conversations=request.conversations,
            session_id=request.session_id,
        )

        if memory:
            return {
                "success": True,
                "memory": memory.to_dict(),
            }
        else:
            return {
                "success": True,
                "message": "没有生成记忆",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/memories/search")
async def search_memories(request: MemorySearchRequest):
    """
    搜索记忆

    根据查询文本搜索相关记忆。

    Args:
        request: 搜索请求

    Returns:
        搜索结果列表
    """
    if not client:
        raise HTTPException(status_code=503, detail="Client not initialized")

    try:
        results = await client.search_memories(
            query=request.query,
            user_id=request.user_id,
            top_k=request.top_k,
        )

        return {
            "memories": [
                {
                    "memory": r.memory.to_dict(),
                    "score": r.score,
                    "match_type": r.match_type,
                }
                for r in results
            ],
            "total": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memories/{memory_id}")
async def get_memory(memory_id: str):
    """
    获取单个记忆

    根据记忆ID获取详细信息。

    Args:
        memory_id: 记忆ID

    Returns:
        记忆详情
    """
    if not client:
        raise HTTPException(status_code=503, detail="Client not initialized")

    try:
        memory = await client.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        return memory.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """
    删除记忆

    根据记忆ID删除记忆。

    Args:
        memory_id: 记忆ID

    Returns:
        删除结果
    """
    if not client:
        raise HTTPException(status_code=503, detail="Client not initialized")

    try:
        success = await client.delete_memory(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")

        return {"success": True, "message": "Memory deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/users/{user_id}/memories")
async def get_user_memories(
    user_id: str,
    memory_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """
    获取用户记忆

    获取指定用户的所有记忆。

    Args:
        user_id: 用户ID
        memory_type: 记忆类型过滤（可选）
        limit: 返回数量限制

    Returns:
        记忆列表
    """
    if not client:
        raise HTTPException(status_code=503, detail="Client not initialized")

    try:
        memories = await client.get_user_memories(
            user_id=user_id,
            memory_type=memory_type,
            limit=limit,
        )

        return {
            "memories": [m.to_dict() for m in memories],
            "total": len(memories),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/memories/consolidate")
async def consolidate_memories(user_id: str):
    """
    整合记忆

    整合用户的记忆，合并相似记忆。

    Args:
        user_id: 用户ID

    Returns:
        整合结果
    """
    if not client:
        raise HTTPException(status_code=503, detail="Client not initialized")

    try:
        await client.consolidate_memories(user_id=user_id)
        return {"success": True, "message": "Memories consolidated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/context")
async def get_context(request: ContextRequest):
    """
    获取上下文

    获取与查询相关的记忆上下文。

    Args:
        request: 上下文请求

    Returns:
        上下文文本
    """
    if not client:
        raise HTTPException(status_code=503, detail="Client not initialized")

    try:
        context = await client.get_relevant_context(
            query=request.query,
            user_id=request.user_id,
            max_tokens=request.max_tokens,
        )

        return {"context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat")
async def chat_with_memory(request: ContextRequest):
    """
    带记忆的对话

    获取相关记忆上下文，用于增强 LLM 回答。

    Args:
        request: 包含查询文本的请求

    Returns:
        包含上下文的响应
    """
    if not client:
        raise HTTPException(status_code=503, detail="Client not initialized")

    try:
        # 搜索相关记忆
        memories = await client.search_memories(
            query=request.query,
            user_id=request.user_id,
            top_k=5,
        )

        # 获取上下文
        context = await client.get_relevant_context(
            query=request.query,
            user_id=request.user_id,
            max_tokens=request.max_tokens,
        )

        return {
            "query": request.query,
            "context": context,
            "memories": [
                {
                    "content": r.memory.content,
                    "score": r.score,
                }
                for r in memories
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
