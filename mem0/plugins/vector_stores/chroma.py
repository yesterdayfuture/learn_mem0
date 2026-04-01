"""
ChromaDB 向量存储适配器模块

本模块实现 ChromaDB 向量数据库的适配器，提供向量存储和相似度搜索功能。

ChromaDB 特点：
- 轻量级，支持本地持久化
- 简单易用的 API
- 支持元数据过滤
- 适合中小型项目

依赖安装:
    pip install chromadb

使用示例:
    from mem0.plugins.vector_stores.chroma import ChromaDBStore

    store = ChromaDBStore({
        "collection_name": "mem0_memories",
        "persist_directory": "./data/chroma"
    })

    # 添加记录
    records = [VectorRecord(id="1", vector=[0.1, 0.2], text="hello", metadata={})]
    await store.add(records)

    # 搜索
    results = await store.search(query_vector=[0.1, 0.2], top_k=5)
"""

import os
from typing import Any, Dict, List, Optional

from .base import SearchResult, VectorRecord, VectorStoreInterface


class ChromaDBStore(VectorStoreInterface):
    """
    ChromaDB 向量存储适配器

    实现 VectorStoreInterface 接口，提供基于 ChromaDB 的向量存储功能。

    Attributes:
        config: 配置字典
        _client: ChromaDB 客户端实例
        _collection: ChromaDB 集合实例

    配置参数:
        - collection_name: 集合名称（默认 "mem0_memories"）
        - persist_directory: 持久化目录（默认 "./data/chroma"）
        - distance_function: 距离函数（默认 "cosine"，可选 "l2"、"ip"）
    """

    def __init__(self):
        """
        初始化 ChromaDB 存储

        注意：实际初始化通过 initialize() 方法完成
        """
        super().__init__()
        self._client = None  # ChromaDB 客户端
        self._collection = None  # ChromaDB 集合

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        异步初始化 ChromaDB 存储

        Args:
            config: 配置字典，包含：
                - collection_name: 集合名称
                - persist_directory: 持久化目录路径
                - distance_function: 距离函数类型
        """
        self.config = config
        self._init_client()

    def _init_client(self) -> None:
        """
        初始化 ChromaDB 客户端（内部方法）

        根据配置创建持久化客户端，获取或创建指定集合。
        如果集合不存在则自动创建。
        """
        import chromadb
        from chromadb.config import Settings

        # 获取配置参数
        persist_dir = self.config.get("persist_directory", "./data/chroma")
        collection_name = self.config.get("collection_name", "mem0_memories")
        distance_fn = self.config.get("distance_function", "cosine")

        # 确保持久化目录存在
        os.makedirs(persist_dir, exist_ok=True)

        # 创建持久化客户端
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,  # 禁用遥测
            ),
        )

        # 获取或创建集合
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_fn},  # 设置距离函数
        )

    async def health_check(self) -> bool:
        """
        健康检查

        检查 ChromaDB 是否正常工作。

        Returns:
            True 表示健康，False 表示异常
        """
        try:
            # 尝试获取集合信息
            self._collection.count()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """
        关闭连接

        ChromaDB 是本地数据库，无需显式关闭连接。
        此方法用于接口兼容性。
        """
        # ChromaDB 无需显式关闭
        pass

    async def add(self, records: List[VectorRecord]) -> List[str]:
        """
        添加向量记录

        将向量记录批量添加到 ChromaDB 集合中。

        Args:
            records: 向量记录列表

        Returns:
            添加成功的记录ID列表

        Example:
            records = [
                VectorRecord(
                    id="1",
                    vector=[0.1, 0.2, 0.3],
                    text="hello",
                    metadata={"user_id": "user_001"}
                )
            ]
            ids = await store.add(records)
        """
        if not records:
            return []

        # 准备批量添加的数据
        ids = [r.id for r in records]
        vectors = [r.vector for r in records]
        documents = [r.text for r in records]
        metadatas = [r.metadata for r in records]

        # 批量添加到 ChromaDB
        self._collection.add(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )

        return ids

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        相似度搜索

        根据查询向量搜索最相似的记录。

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            filters: 元数据过滤条件

        Returns:
            搜索结果列表，按相似度降序排列

        Example:
            results = await store.search(
                query_vector=[0.1, 0.2],
                top_k=10,
                filters={"user_id": "user_001"}
            )
        """
        # 执行搜索
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filters,  # 元数据过滤
            include=["metadatas", "documents", "distances"],
        )

        # 解析结果
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB 返回的是距离，需要转换为相似度分数
                # cosine 距离范围 [0, 2]，转换为相似度 [0, 1]
                distance = results["distances"][0][i]
                similarity = 1 - (distance / 2)  # 转换为相似度分数

                search_results.append(
                    SearchResult(
                        id=doc_id,
                        score=similarity,
                        text=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                    )
                )

        return search_results

    async def get(self, id: str) -> Optional[VectorRecord]:
        """
        获取单条记录

        Args:
            id: 记录ID

        Returns:
            向量记录，如果不存在则返回 None
        """
        try:
            result = self._collection.get(
                ids=[id],
                include=["embeddings", "documents", "metadatas"],
            )

            if result["ids"]:
                return VectorRecord(
                    id=result["ids"][0],
                    vector=result["embeddings"][0],
                    text=result["documents"][0],
                    metadata=result["metadatas"][0],
                )
        except Exception:
            pass

        return None

    async def update(self, records: List[VectorRecord]) -> List[str]:
        """
        更新向量记录

        ChromaDB 使用 add 方法更新（相同 ID 会覆盖）。

        Args:
            records: 要更新的向量记录列表

        Returns:
            更新成功的记录ID列表
        """
        # ChromaDB 的 add 方法会自动覆盖相同 ID 的记录
        return await self.add(records)

    async def delete(self, ids: List[str]) -> bool:
        """
        删除向量记录

        Args:
            ids: 要删除的记录ID列表

        Returns:
            是否删除成功
        """
        try:
            self._collection.delete(ids=ids)
            return True
        except Exception:
            return False

    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[VectorRecord]:
        """
        列出向量记录

        Args:
            filters: 过滤条件
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            向量记录列表
        """
        try:
            # 获取所有记录
            result = self._collection.get(
                where=filters,
                limit=limit,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            )

            records = []
            for i, doc_id in enumerate(result["ids"]):
                records.append(
                    VectorRecord(
                        id=doc_id,
                        vector=result["embeddings"][i],
                        text=result["documents"][i],
                        metadata=result["metadatas"][i],
                    )
                )
            return records
        except Exception:
            return []

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        统计记录数量

        Args:
            filters: 过滤条件

        Returns:
            记录数量
        """
        try:
            return self._collection.count(where=filters)
        except Exception:
            return 0

    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息

        Returns:
            集合信息字典，包含名称、记录数等
        """
        return {
            "name": self._collection.name,
            "count": self._collection.count(),
            "metadata": self._collection.metadata,
        }

    async def stats(self) -> Dict[str, Any]:
        """
        获取向量库统计信息

        Returns:
            包含统计信息的字典
        """
        try:
            count = self._collection.count()
            return {
                "total_count": count,
                "collection_name": self._collection.name,
            }
        except Exception as e:
            return {"error": str(e)}
