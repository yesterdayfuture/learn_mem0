"""
向量存储插件基础接口模块

本模块定义向量存储插件的通用接口，所有向量数据库适配器都需要实现这个接口。

支持的向量数据库：
- ChromaDB（已实现）
- Milvus（待实现）
- Pinecone（待实现）
- Weaviate（待实现）
- Qdrant（待实现）

向量存储的核心功能：
1. 向量存储 - 将文本和向量一起存储
2. 相似度搜索 - 基于向量相似度查找相关内容
3. 元数据过滤 - 支持按条件过滤搜索结果
4. CRUD 操作 - 增删改查向量记录

使用示例:
    # 使用 ChromaDB 向量存储
    from mem0.plugins.vector_stores.chroma import ChromaDBStore

    store = ChromaDBStore({
        "collection_name": "my_collection",
        "persist_directory": "./data"
    })

    # 添加向量记录
    records = [VectorRecord(id="1", vector=[0.1, 0.2], text="hello", metadata={})]
    await store.add(records)

    # 搜索相似向量
    results = await store.search(query_vector=[0.1, 0.2], top_k=5)
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..base import PluginInterface


@dataclass
class VectorRecord:
    """
    向量记录数据类

    表示一条向量记录，包含向量、文本和元数据。
    这是向量存储的基本数据单元。

    Attributes:
        id: 记录唯一标识符
        vector: 向量数据（浮点数列表）
        text: 原始文本内容
        metadata: 元数据字典，可包含 user_id、memory_type 等
    """

    id: str  # 唯一标识符
    vector: List[float]  # 向量数据
    text: str  # 原始文本
    metadata: Dict[str, Any]  # 元数据字典


@dataclass
class SearchResult:
    """
    搜索结果数据类

    表示一次向量搜索的结果。

    Attributes:
        id: 记录ID
        score: 相似度分数（0-1，越高越相似）
        text: 文本内容
        metadata: 元数据
    """

    id: str  # 记录ID
    score: float  # 相似度分数
    text: str  # 文本内容
    metadata: Dict[str, Any]  # 元数据


class VectorStoreInterface(PluginInterface):
    """
    向量存储接口 - 所有向量数据库插件的基类

    定义向量存储的标准操作：
    - add: 添加向量记录
    - search: 相似度搜索
    - get: 获取单条记录
    - update: 更新记录
    - delete: 删除记录
    - list: 列出所有记录

    所有向量数据库适配器都必须实现这些方法。

    Attributes:
        config: 配置字典，包含连接参数
    """

    @abstractmethod
    async def add(self, records: List[VectorRecord]) -> List[str]:
        """
        添加向量记录

        将向量记录批量添加到向量库中。

        Args:
            records: 向量记录列表，每个记录包含 id、vector、text、metadata

        Returns:
            添加成功的记录ID列表

        Example:
            records = [
                VectorRecord(
                    id="1",
                    vector=[0.1, 0.2, 0.3],
                    text="hello world",
                    metadata={"user_id": "user_001"}
                )
            ]
            ids = await store.add(records)
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        相似度搜索

        根据查询向量搜索最相似的记录，支持元数据过滤。

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量（默认 5）
            filters: 过滤条件字典（可选），如 {"user_id": "user_001"}

        Returns:
            搜索结果列表，按相似度降序排列

        Example:
            results = await store.search(
                query_vector=[0.1, 0.2],
                top_k=10,
                filters={"user_id": "user_001"}
            )
            for result in results:
                print(f"ID: {result.id}, Score: {result.score}")
        """
        pass

    @abstractmethod
    async def get(self, id: str) -> Optional[VectorRecord]:
        """
        获取单条记录

        根据ID获取向量记录。

        Args:
            id: 记录ID

        Returns:
            向量记录，如果不存在则返回 None

        Example:
            record = await store.get("record_001")
            if record:
                print(record.text)
        """
        pass

    @abstractmethod
    async def update(self, records: List[VectorRecord]) -> List[str]:
        """
        更新向量记录

        更新已存在的向量记录，如果不存在则创建。

        Args:
            records: 要更新的向量记录列表

        Returns:
            更新成功的记录ID列表

        Example:
            record = VectorRecord(id="1", vector=[0.2, 0.3], text="updated", metadata={})
            await store.update([record])
        """
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        """
        删除向量记录

        根据ID列表删除记录。

        Args:
            ids: 要删除的记录ID列表

        Returns:
            是否删除成功

        Example:
            success = await store.delete(["id1", "id2"])
        """
        pass

    @abstractmethod
    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[VectorRecord]:
        """
        列出向量记录

        支持分页和过滤。

        Args:
            filters: 过滤条件（可选）
            limit: 返回数量限制（默认 100）
            offset: 偏移量（默认 0）

        Returns:
            向量记录列表

        Example:
            records = await store.list(filters={"user_id": "user_001"}, limit=50)
        """
        pass

    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        统计记录数量

        Args:
            filters: 过滤条件（可选）

        Returns:
            记录数量

        Example:
            count = await store.count(filters={"user_id": "user_001"})
            print(f"User has {count} memories")
        """
        pass

    async def stats(self) -> Dict[str, Any]:
        """
        获取向量库统计信息

        Returns:
            包含统计信息的字典，如记录数量、集合信息等

        Example:
            stats = await store.stats()
            print(f"Total records: {stats.get('total_count', 0)}")
        """
        return {}
