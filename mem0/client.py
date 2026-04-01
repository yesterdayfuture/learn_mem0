"""
Mem0 客户端模块 - 提供不启动服务直接使用的客户端接口

本模块是 Mem0 记忆系统的核心客户端实现，支持：
1. 多种初始化方式（配置字典、provider名称、已初始化组件实例）
2. 动态加载不同的数据库和模型厂商（通过插件系统）
3. 对话管理、记忆存储、记忆搜索等核心功能
4. 向量库和图数据库的双库支持（可配置存储位置）
5. 多源记忆搜索（vector/graph/both）

主要功能：
- add_conversation(): 添加单条对话
- add_messages(): 批量添加消息列表（支持 OpenAI 格式，自动触发总结）
- search_memories(): 搜索记忆（支持选择搜索来源）

使用示例:
    # 方式1: 使用默认配置（从环境变量读取）
    async with Mem0Client() as client:
        await client.add_conversation("user_001", "user", "我叫张三")

    # 方式2: 指定 provider 名称
    async with Mem0Client(
        vector_store="chromadb",
        graph_db="nebula",
        embedding="openai",
        llm="openai",
    ) as client:
        pass

    # 方式3: 使用完整配置字典
    config = {...}
    async with Mem0Client(config) as client:
        pass

    # 方式4: 传入已初始化的组件实例
    async with Mem0Client(
        vector_store_instance=vector_store,
        embedding_instance=embedding,
        llm_instance=llm,
    ) as client:
        pass

    # 批量添加消息（支持 OpenAI 格式）
    messages = [
        {"role": "user", "content": "你好，我叫张三"},
        {"role": "assistant", "content": "你好张三！"},
        {"role": "user", "content": "我是一名工程师"},
    ]
    memory = await client.add_messages("user_001", messages)

    # 搜索记忆（仅从向量库）
    results = await client.search_memories("用户的名字", user_id="user_001", search_source="vector")

    # 搜索记忆（仅从图数据库）
    results = await client.search_memories("张三的朋友", user_id="user_001", search_source="graph")

    # 搜索记忆（从两者搜索，默认）
    results = await client.search_memories("用户的职业", user_id="user_001", search_source="both")
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .config import Mem0Config, registry
from .core.memory import MemoryManager
from .core.models import Memory, MemorySearchResult, MemoryType
from .plugins.graph_databases.base import GraphDatabaseInterface
from .plugins.models.base import EmbeddingInterface, ModelInterface
from .plugins.vector_stores.base import VectorStoreInterface


class Mem0Client:
    """
    Mem0 客户端类 - 记忆系统的核心接口

    提供完整的记忆管理功能，包括：
    - 对话添加和管理（支持单条和批量消息列表）
    - 自动总结和记忆提取（达到阈值自动触发）
    - 向量库和图数据库存储（可配置存储位置）
    - 多源记忆搜索（支持 vector/graph/both）
    - 带记忆的对话

    支持四种初始化方式：
    1. 默认配置（从环境变量读取）
    2. 指定 provider 名称
    3. 完整配置字典
    4. 传入已初始化的组件实例

    核心方法：
    - add_conversation(): 添加单条对话记录
    - add_messages(): 批量添加消息列表（支持 OpenAI 格式），自动触发总结
    - search_memories(): 搜索记忆，支持选择搜索来源（vector/graph/both）
    - get_relevant_context(): 获取与查询相关的记忆上下文
    - chat_with_memory(): 带记忆的对话
    - update_config(): 更新配置，包括双库存储配置

    双库存储配置：
    - store_to_vector: 是否存储到向量库（默认 True）
    - store_to_graph: 是否存储到图数据库（默认 True）
    通过 update_config() 方法进行配置

    Attributes:
        _config: 配置对象
        _initialized: 是否已初始化标志
        _memory_manager: 记忆管理器实例
        _vector_store: 向量存储实例
        _graph_db: 图数据库实例
        _embedding_model: 嵌入模型实例
        _llm_model: 大模型实例
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        vector_store: Optional[str] = None,
        graph_db: Optional[str] = None,
        embedding: Optional[str] = None,
        llm: Optional[str] = None,
        vector_store_instance: Optional[VectorStoreInterface] = None,
        graph_db_instance: Optional[GraphDatabaseInterface] = None,
        embedding_instance: Optional[EmbeddingInterface] = None,
        llm_instance: Optional[ModelInterface] = None,
        **kwargs
    ):
        """
        初始化 Mem0Client 客户端

        支持四种初始化方式，按优先级：
        1. 传入已初始化的组件实例（最高优先级）
        2. 传入完整配置字典
        3. 传入 provider 名称
        4. 使用默认配置（从环境变量读取）

        Args:
            config: 完整配置字典，结构如下:
                {
                    "vector_store": {"provider": "chromadb", "config": {...}},
                    "graph_db": {"provider": "nebula", "config": {...}},
                    "embedding": {"provider": "openai", "config": {...}},
                    "llm": {"provider": "openai", "config": {...}},
                }
            vector_store: 向量存储提供商名称 (如 "chromadb")
            graph_db: 图数据库提供商名称 (如 "nebula")
            embedding: 嵌入模型提供商名称 (如 "openai")
            llm: 大模型提供商名称 (如 "openai")
            vector_store_instance: 已初始化的向量存储实例
            graph_db_instance: 已初始化的图数据库实例
            embedding_instance: 已初始化的嵌入模型实例
            llm_instance: 已初始化的大模型实例
            **kwargs: 额外的配置参数，可用于传递各插件的详细配置

        Raises:
            ValueError: 当指定的 provider 不存在时
            RuntimeError: 当客户端未初始化就使用时
        """
        # 原始配置字典，用于延迟初始化
        self._raw_config: Optional[Dict[str, Any]] = None
        # 配置对象，初始化后创建
        self._config: Optional[Mem0Config] = None
        # 初始化标志
        self._initialized: bool = False

        # 外部传入的组件实例（已初始化）
        self._external_vector_store: Optional[VectorStoreInterface] = vector_store_instance
        self._external_graph_db: Optional[GraphDatabaseInterface] = graph_db_instance
        self._external_embedding: Optional[EmbeddingInterface] = embedding_instance
        self._external_llm: Optional[ModelInterface] = llm_instance

        # 内部创建并管理的组件实例
        self._memory_manager: Optional[MemoryManager] = None
        self._vector_store: Optional[VectorStoreInterface] = None
        self._graph_db: Optional[GraphDatabaseInterface] = None
        self._embedding_model: Optional[EmbeddingInterface] = None
        self._llm_model: Optional[ModelInterface] = None

        # 标记是否使用外部传入的实例（用于决定是否需要关闭）
        self._use_external_vector_store: bool = vector_store_instance is not None
        self._use_external_graph_db: bool = graph_db_instance is not None
        self._use_external_embedding: bool = embedding_instance is not None
        self._use_external_llm: bool = llm_instance is not None

        # 根据传入参数确定初始化方式
        if config is not None:
            # 方式2: 传入完整配置字典
            self._raw_config = config
        elif any([vector_store, graph_db, embedding, llm]):
            # 方式3: 传入 provider 名称
            self._raw_config = self._build_config_from_kwargs(
                vector_store=vector_store,
                graph_db=graph_db,
                embedding=embedding,
                llm=llm,
                **kwargs
            )
        else:
            # 方式4: 使用默认配置（从环境变量读取）或完全依赖外部实例
            self._raw_config = None

    def _build_config_from_kwargs(
        self,
        vector_store: Optional[str] = None,
        graph_db: Optional[str] = None,
        embedding: Optional[str] = None,
        llm: Optional[str] = None,
        **extra_config
    ) -> Dict[str, Any]:
        """
        从简化参数构建完整配置字典

        将 provider 名称和额外配置转换为标准配置格式

        Args:
            vector_store: 向量存储提供商名称
            graph_db: 图数据库提供商名称
            embedding: 嵌入模型提供商名称
            llm: 大模型提供商名称
            **extra_config: 额外配置参数，如 vector_store_config, graph_db_config 等

        Returns:
            标准格式的配置字典
        """
        config: Dict[str, Any] = {}

        if vector_store:
            config["vector_store"] = {
                "provider": vector_store,
                "config": extra_config.get("vector_store_config", {}),
            }

        if graph_db:
            config["graph_db"] = {
                "provider": graph_db,
                "config": extra_config.get("graph_db_config", {}),
            }

        if embedding:
            config["embedding"] = {
                "provider": embedding,
                "config": extra_config.get("embedding_config", {}),
            }

        if llm:
            config["llm"] = {
                "provider": llm,
                "config": extra_config.get("llm_config", {}),
            }

        return config

    async def initialize(self) -> None:
        """
        初始化客户端的所有组件

        按照以下顺序初始化：
        1. 嵌入模型（用于生成向量）
        2. 大模型（用于总结、实体提取等）
        3. 向量存储（用于存储记忆向量）
        4. 图数据库（用于存储实体关系，可选）
        5. 记忆管理器（整合以上组件）

        如果使用了外部传入的组件实例，则跳过对应组件的初始化

        Raises:
            ValueError: 当指定的 provider 不存在时
        """
        if self._initialized:
            return

        # 加载环境变量（用于默认配置）
        load_dotenv()

        # ========== 1. 初始化嵌入模型 ==========
        if self._use_external_embedding:
            # 使用外部传入的嵌入模型实例
            self._embedding_model = self._external_embedding
        else:
            # 从配置创建嵌入模型
            if self._raw_config:
                self._config = Mem0Config(self._raw_config)
            else:
                self._config = Mem0Config.from_env()
            embedding_config = self._config.get_embedding_config()
            self._embedding_model = await self._load_embedding(embedding_config)

        # ========== 2. 初始化大模型 ==========
        if self._use_external_llm:
            # 使用外部传入的大模型实例
            self._llm_model = self._external_llm
        else:
            # 确保配置已创建
            if self._config is None:
                if self._raw_config:
                    self._config = Mem0Config(self._raw_config)
                else:
                    self._config = Mem0Config.from_env()
            llm_config = self._config.get_llm_config()
            self._llm_model = await self._load_llm(llm_config)

        # ========== 3. 初始化向量存储 ==========
        if self._use_external_vector_store:
            # 使用外部传入的向量存储实例
            self._vector_store = self._external_vector_store
        else:
            # 确保配置已创建
            if self._config is None:
                if self._raw_config:
                    self._config = Mem0Config(self._raw_config)
                else:
                    self._config = Mem0Config.from_env()
            vector_config = self._config.get_vector_store_config()
            self._vector_store = await self._load_vector_store(vector_config)

        # ========== 4. 初始化图数据库（可选） ==========
        if self._use_external_graph_db:
            # 使用外部传入的图数据库实例
            self._graph_db = self._external_graph_db
        elif self._config and self._config.get_graph_db_config().get("provider"):
            # 尝试从配置加载图数据库
            try:
                graph_config = self._config.get_graph_db_config()
                self._graph_db = await self._load_graph_db(graph_config)
            except Exception as e:
                print(f"图数据库初始化失败（可选）: {e}")
                self._graph_db = None
        elif not self._use_external_vector_store and not self._use_external_embedding and not self._use_external_llm:
            # 尝试从配置加载图数据库
            if self._config is None:
                if self._raw_config:
                    self._config = Mem0Config(self._raw_config)
                else:
                    self._config = Mem0Config.from_env()
            graph_config = self._config.get_graph_db_config()
            if graph_config.get("provider"):
                try:
                    self._graph_db = await self._load_graph_db(graph_config)
                except Exception as e:
                    print(f"图数据库初始化失败（可选）: {e}")
                    self._graph_db = None

        # ========== 5. 创建记忆管理器 ==========
        self._memory_manager = MemoryManager(
            vector_store=self._vector_store,
            graph_db=self._graph_db,
            embedding_model=self._embedding_model,
            llm_model=self._llm_model,
        )

        self._initialized = True

    async def _load_embedding(self, config: Dict[str, Any]) -> EmbeddingInterface:
        """
        加载嵌入模型插件

        Args:
            config: 嵌入模型配置字典

        Returns:
            初始化好的嵌入模型实例

        Raises:
            ValueError: 当指定的 provider 不存在时
        """
        provider = config.get("provider", "openai")
        plugin_config = config.get("config", {})

        # 从环境变量补充 API Key
        if not plugin_config.get("api_key"):
            plugin_config["api_key"] = os.getenv("OPENAI_API_KEY")

        # 从注册中心获取插件类
        plugin_class = registry.get_embedding(provider)
        if not plugin_class:
            raise ValueError(
                f"未知的嵌入模型提供商: {provider}. "
                f"可用选项: {registry.list_embeddings()}"
            )

        # 创建并初始化插件实例
        plugin = plugin_class()
        await plugin.initialize(plugin_config)
        return plugin

    async def _load_llm(self, config: Dict[str, Any]) -> ModelInterface:
        """
        加载大模型插件

        Args:
            config: 大模型配置字典

        Returns:
            初始化好的大模型实例

        Raises:
            ValueError: 当指定的 provider 不存在时
        """
        provider = config.get("provider", "openai")
        plugin_config = config.get("config", {})

        # 从环境变量补充 API Key
        if not plugin_config.get("api_key"):
            plugin_config["api_key"] = os.getenv("OPENAI_API_KEY")

        # 从注册中心获取插件类
        plugin_class = registry.get_llm(provider)
        if not plugin_class:
            raise ValueError(
                f"未知的大模型提供商: {provider}. "
                f"可用选项: {registry.list_llms()}"
            )

        # 创建并初始化插件实例
        plugin = plugin_class()
        await plugin.initialize(plugin_config)
        return plugin

    async def _load_vector_store(self, config: Dict[str, Any]) -> VectorStoreInterface:
        """
        加载向量存储插件

        Args:
            config: 向量存储配置字典

        Returns:
            初始化好的向量存储实例

        Raises:
            ValueError: 当指定的 provider 不存在时
        """
        provider = config.get("provider", "chromadb")
        plugin_config = config.get("config", {})

        # 从注册中心获取插件类
        plugin_class = registry.get_vector_store(provider)
        if not plugin_class:
            raise ValueError(
                f"未知的向量存储提供商: {provider}. "
                f"可用选项: {registry.list_vector_stores()}"
            )

        # 创建并初始化插件实例
        plugin = plugin_class()
        await plugin.initialize(plugin_config)
        return plugin

    async def _load_graph_db(self, config: Dict[str, Any]) -> GraphDatabaseInterface:
        """
        加载图数据库插件

        Args:
            config: 图数据库配置字典

        Returns:
            初始化好的图数据库实例

        Raises:
            ValueError: 当指定的 provider 不存在时
        """
        provider = config.get("provider", "nebula")
        plugin_config = config.get("config", {})

        # 从注册中心获取插件类
        plugin_class = registry.get_graph_database(provider)
        if not plugin_class:
            raise ValueError(
                f"未知的图数据库提供商: {provider}. "
                f"可用选项: {registry.list_graph_databases()}"
            )

        # 创建并初始化插件实例
        plugin = plugin_class()
        await plugin.initialize(plugin_config)
        return plugin

    async def close(self) -> None:
        """
        关闭客户端，释放所有资源

        注意：外部传入的组件实例不会被关闭，由调用方管理
        """
        # 只关闭内部创建的组件
        if self._vector_store and not self._use_external_vector_store:
            await self._vector_store.close()
        if self._graph_db and not self._use_external_graph_db:
            await self._graph_db.close()
        if self._embedding_model and not self._use_external_embedding:
            await self._embedding_model.close()
        if self._llm_model and not self._use_external_llm:
            await self._llm_model.close()
        self._initialized = False

    async def __aenter__(self):
        """
        异步上下文管理器入口

        自动调用 initialize() 方法
        """
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        异步上下文管理器出口

        自动调用 close() 方法
        """
        await self.close()

    def _ensure_initialized(self):
        """
        确保客户端已初始化

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        if not self._initialized:
            raise RuntimeError("客户端未初始化，请先调用 initialize()")

    # ==================== 属性访问器 ====================

    @property
    def config(self) -> Optional[Mem0Config]:
        """获取配置对象"""
        return self._config

    @property
    def vector_store(self) -> Optional[VectorStoreInterface]:
        """获取向量存储实例"""
        return self._vector_store

    @property
    def graph_db(self) -> Optional[GraphDatabaseInterface]:
        """获取图数据库实例"""
        return self._graph_db

    @property
    def embedding_model(self) -> Optional[EmbeddingInterface]:
        """获取嵌入模型实例"""
        return self._embedding_model

    @property
    def llm_model(self) -> Optional[ModelInterface]:
        """获取大模型实例"""
        return self._llm_model

    # ==================== 核心功能 ====================

    async def add_conversation(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加单条对话记录

        将单条对话添加到缓冲区，当缓冲区达到阈值时会自动触发总结

        Args:
            user_id: 用户唯一标识
            role: 对话角色，可选 "user" 或 "assistant"
            content: 对话内容
            session_id: 会话ID，用于区分不同会话（可选）
            metadata: 额外元数据，如时间戳、来源等（可选）

        Returns:
            对话记录的唯一ID

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()
        conversation = await self._memory_manager.add_conversation(
            user_id=user_id,
            role=role,
            content=content,
            session_id=session_id,
            metadata=metadata,
        )
        return conversation.id

    async def add_messages(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Memory]:
        """
        批量添加对话消息并自动总结

        支持 OpenAI 格式的消息列表，自动将消息添加到缓冲区并触发总结。
        当对话数量达到阈值时会自动触发总结，生成记忆并存储到向量库和图数据库。

        Args:
            user_id: 用户唯一标识
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            session_id: 会话ID（可选）
            metadata: 额外元数据（可选）

        Returns:
            如果触发总结则返回生成的 Memory 对象，否则返回 None

        Raises:
            RuntimeError: 当客户端未初始化时

        Example:
            messages = [
                {"role": "user", "content": "你好，我叫张三"},
                {"role": "assistant", "content": "你好张三！很高兴认识你。"},
                {"role": "user", "content": "我是一名软件工程师"},
            ]
            memory = await client.add_messages("user_001", messages)
            if memory:
                print(f"记忆已创建: {memory.content}")
        """
        self._ensure_initialized()
        return await self._memory_manager.add_messages(
            user_id=user_id,
            messages=messages,
            session_id=session_id,
            metadata=metadata,
        )

    async def search_memories(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
        search_source: str = "both",
    ) -> List[MemorySearchResult]:
        """
        搜索记忆

        支持从向量库、图数据库或两者同时搜索记忆

        Args:
            query: 搜索查询文本
            user_id: 用户ID，用于过滤特定用户的记忆
            top_k: 返回结果的最大数量
            memory_type: 记忆类型过滤（可选）
            search_source: 搜索来源，可选值：
                - "vector": 仅从向量库搜索（基于语义相似度）
                - "graph": 仅从图数据库搜索（基于实体匹配）
                - "both": 从两者搜索并合并结果（默认）

        Returns:
            记忆搜索结果列表，每个结果包含：
                - memory: 记忆对象
                - score: 匹配分数（0-1）
                - match_type: 匹配来源（"vector", "graph", "hybrid"）

        Raises:
            RuntimeError: 当客户端未初始化时
            ValueError: 当 search_source 参数无效时

        Example:
            # 默认从两者搜索
            results = await client.search_memories("用户的爱好", user_id="user_001")

            # 仅从向量库搜索
            results = await client.search_memories(
                "用户的爱好", user_id="user_001", search_source="vector"
            )
        """
        self._ensure_initialized()

        # 直接调用 MemoryManager 的 search_memories 方法
        return await self._memory_manager.search_memories(
            query=query,
            user_id=user_id,
            top_k=top_k,
            memory_type=memory_type,
            search_source=search_source,
        )

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        获取指定ID的记忆详情

        Args:
            memory_id: 记忆唯一标识

        Returns:
            记忆对象，如果不存在则返回 None

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()
        return await self._memory_manager.get_memory(memory_id)

    async def delete_memory(self, memory_id: str) -> bool:
        """
        删除指定ID的记忆

        会从向量库和图数据库中同时删除

        Args:
            memory_id: 记忆唯一标识

        Returns:
            是否删除成功

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()
        return await self._memory_manager.delete_memory(memory_id)

    async def get_relevant_context(
        self,
        query: str,
        user_id: str = "default",
        max_tokens: int = 2000,
    ) -> str:
        """
        获取与查询相关的记忆上下文

        搜索相关记忆并格式化为上下文文本，可用于增强 LLM 回答

        Args:
            query: 查询文本
            user_id: 用户ID
            max_tokens: 上下文最大长度（字符数）

        Returns:
            格式化的记忆上下文文本

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()
        return await self._memory_manager.get_relevant_context(
            query=query,
            user_id=user_id,
            max_tokens=max_tokens,
        )

    async def force_summarize(self, user_id: str, session_id: Optional[str] = None) -> Optional[Memory]:
        """
        强制总结当前缓冲区中的对话

        即使缓冲区未达到阈值，也立即触发总结

        Args:
            user_id: 用户ID
            session_id: 会话ID（可选）

        Returns:
            生成的记忆对象，如果缓冲区为空则返回 None

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()
        return await self._memory_manager.force_summarize(user_id, session_id)

    async def chat_with_memory(
        self,
        message: str,
        user_id: str = "default",
        session_id: Optional[str] = None,
        include_memories: bool = True,
    ) -> str:
        """
        带记忆的聊天

        自动完成以下流程：
        1. 添加用户消息到记忆
        2. 获取相关记忆作为上下文
        3. 使用 LLM 生成回复
        4. 添加助手回复到记忆

        Args:
            message: 用户消息
            user_id: 用户ID
            session_id: 会话ID（可选）
            include_memories: 是否包含相关记忆作为上下文

        Returns:
            助手生成的回复文本

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()

        # 添加用户消息到记忆
        await self.add_conversation(user_id, "user", message, session_id)

        # 获取相关记忆作为上下文
        context = ""
        if include_memories:
            context = await self.get_relevant_context(message, user_id)

        # 构建消息列表
        from .plugins.models.base import ChatMessage, MessageRole

        messages = []
        if context:
            messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"以下是与用户相关的记忆信息，请在回答时参考：\n\n{context}",
            ))
        messages.append(ChatMessage(role=MessageRole.USER, content=message))

        # 使用 LLM 生成回复
        result = await self._llm_model.chat(messages=messages, temperature=0.7)
        response = result.content

        # 添加助手回复到记忆
        await self.add_conversation(user_id, "assistant", response, session_id)

        return response

    # ==================== 私有搜索方法 ====================

    async def _search_from_graph_only(
        self,
        query: str,
        user_id: str,
        top_k: int,
    ) -> List[MemorySearchResult]:
        """
        仅从图数据库搜索记忆（私有方法）

        流程：
        1. 从查询中提取实体
        2. 在图数据库中查找匹配的实体节点
        3. 获取与这些实体相关的记忆

        Args:
            query: 搜索查询
            user_id: 用户ID
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        from .core.models import MemorySearchResult

        results = []

        if not self._graph_db:
            return results

        # 1. 从查询中提取实体
        entities = await self._llm_model.extract_entities(query)

        # 2. 在图数据库中查找相关实体
        memory_ids = set()
        for entity in entities:
            entity_name = entity.get("name", "")
            if entity_name:
                # 搜索实体节点
                nodes = await self._graph_db.search_nodes(
                    properties={"name": entity_name}
                )
                for node in nodes:
                    memory_id = node.properties.get("memory_id")
                    if memory_id:
                        memory_ids.add(memory_id)

                # 获取相关关系
                node = await self._graph_db.get_node(entity_name)
                if node:
                    relationships = await self._graph_db.get_relationships(
                        entity_name, direction="both"
                    )
                    for rel in relationships:
                        # 从关系中找到相关的记忆
                        for entity_id in [rel.source_id, rel.target_id]:
                            related_node = await self._graph_db.get_node(entity_id)
                            if related_node:
                                memory_id = related_node.properties.get("memory_id")
                                if memory_id:
                                    memory_ids.add(memory_id)

        # 3. 获取记忆详情
        for memory_id in list(memory_ids)[:top_k]:
            memory = await self._memory_manager.get_memory(memory_id)
            if memory and memory.user_id == user_id:
                results.append(MemorySearchResult(
                    memory=memory,
                    score=0.8,  # 图搜索默认分数
                    match_type="graph"
                ))

        return results

    async def _search_from_both(
        self,
        query: str,
        user_id: str,
        top_k: int,
        memory_type: Optional[MemoryType] = None,
    ) -> List[MemorySearchResult]:
        """
        从向量库和图数据库同时搜索并合并结果（私有方法）

        合并策略：
        1. 分别从两个源获取搜索结果
        2. 对于同时存在于两个源的记忆，标记为 "hybrid" 并取最高分数
        3. 按分数排序返回

        Args:
            query: 搜索查询
            user_id: 用户ID
            top_k: 返回结果数量
            memory_type: 记忆类型过滤

        Returns:
            合并后的搜索结果列表
        """
        from .core.models import MemorySearchResult

        # 1. 向量库搜索
        vector_results = await self._memory_manager.search_memories(
            query=query,
            user_id=user_id,
            top_k=top_k * 2,  # 获取更多结果用于合并
            memory_type=memory_type,
        )

        # 2. 图数据库搜索
        graph_results = await self._search_from_graph_only(
            query=query,
            user_id=user_id,
            top_k=top_k * 2,
        )

        # 3. 合并结果
        memory_map: Dict[str, MemorySearchResult] = {}

        # 添加向量搜索结果
        for result in vector_results:
            memory_map[result.memory.id] = MemorySearchResult(
                memory=result.memory,
                score=result.score,
                match_type="vector"
            )

        # 添加/合并图搜索结果
        for result in graph_results:
            if result.memory.id in memory_map:
                # 已存在，升级为 hybrid，取最高分数
                existing = memory_map[result.memory.id]
                memory_map[result.memory.id] = MemorySearchResult(
                    memory=result.memory,
                    score=max(existing.score, result.score),
                    match_type="hybrid"
                )
            else:
                memory_map[result.memory.id] = result

        # 4. 按分数排序并返回
        results = list(memory_map.values())
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    # ==================== 管理功能 ====================

    def update_config(self, **kwargs) -> None:
        """
        更新记忆管理器的配置

        Args:
            **kwargs: 配置参数，可选值：
                - summary_threshold: 触发总结的对话数量阈值
                - similarity_threshold: 记忆相似度阈值
                - max_memories_per_query: 每次查询最大记忆数
                - decay_factor: 记忆衰减因子

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()
        self._memory_manager.update_config(**kwargs)

    async def consolidate_memories(self, user_id: str) -> None:
        """
        整合用户的记忆（去重、合并相似记忆）

        Args:
            user_id: 用户ID

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()
        await self._memory_manager.consolidate_memories(user_id)

    async def apply_memory_decay(self, user_id: str) -> None:
        """
        应用记忆衰减

        根据记忆的访问频率和时间衰减重要性分数

        Args:
            user_id: 用户ID

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()
        await self._memory_manager.apply_memory_decay(user_id)

    # ==================== 数据查询功能 ====================

    async def get_all_memories(self, user_id: Optional[str] = None, limit: int = 100) -> List[Memory]:
        """
        获取所有记忆（从向量库）

        注意：ChromaDB 不直接支持列出所有记录，这里通过搜索空向量实现

        Args:
            user_id: 用户ID过滤（可选）
            limit: 返回数量限制

        Returns:
            记忆对象列表

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()
        # 使用一个特殊查询获取记录
        # 实际实现可能需要根据具体向量库调整
        return []

    async def get_vector_store_stats(self) -> Dict[str, Any]:
        """
        获取向量库统计信息

        Returns:
            包含向量库统计信息的字典

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()

        try:
            stats = await self._vector_store.stats()
            return stats
        except Exception as e:
            return {"error": str(e)}

    async def get_graph_stats(self) -> Dict[str, Any]:
        """
        获取图数据库统计信息

        Returns:
            包含图数据库统计信息的字典

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()

        if not self._graph_db:
            return {"error": "图数据库未启用"}

        try:
            labels = await self._graph_db.list_labels(label_type="node")
            node_count = 0
            for label in labels:
                info = await self._graph_db.get_label_info(label, "node")
                if info:
                    node_count += info.count

            edge_labels = await self._graph_db.list_labels(label_type="edge")
            edge_count = 0
            for label in edge_labels:
                info = await self._graph_db.get_label_info(label, "edge")
                if info:
                    edge_count += info.count

            return {
                "node_labels": len(labels),
                "edge_labels": len(edge_labels),
                "total_nodes": node_count,
                "total_edges": edge_count,
            }
        except Exception as e:
            return {"error": str(e)}

    async def search_entities(
        self,
        entity_name: str,
        fuzzy: bool = True,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        在图数据库中搜索实体

        Args:
            entity_name: 实体名称
            fuzzy: 是否使用模糊查询
            limit: 返回数量限制

        Returns:
            实体信息字典列表

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()
        if not self._graph_db:
            return []

        if fuzzy and hasattr(self._graph_db, 'search_entities_fuzzy'):
            return await self._graph_db.search_entities_fuzzy(entity_name, limit)
        else:
            node = await self._graph_db.get_node(entity_name)
            return [node.to_dict()] if node else []

    async def get_entity_relationships(
        self,
        entity_name: str,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        """
        获取实体的关系

        Args:
            entity_name: 实体名称
            direction: 关系方向，可选 "outgoing", "incoming", "both"

        Returns:
            关系信息列表

        Raises:
            RuntimeError: 当客户端未初始化时
        """
        self._ensure_initialized()
        if not self._graph_db:
            return []

        relationships = await self._graph_db.get_relationships(entity_name, direction)
        return [rel.to_dict() for rel in relationships]


# ==================== 便捷函数 ====================

async def create_client(
    vector_store: str = "chromadb",
    graph_db: Optional[str] = "nebula",
    embedding: str = "openai",
    llm: str = "openai",
    **kwargs
) -> Mem0Client:
    """
    便捷函数：创建客户端并指定 provider

    这是 Mem0Client 的简化创建方式，适用于大多数场景

    Args:
        vector_store: 向量存储提供商名称，默认 "chromadb"
        graph_db: 图数据库提供商名称，默认 "nebula"，传入 None 表示不使用
        embedding: 嵌入模型提供商名称，默认 "openai"
        llm: 大模型提供商名称，默认 "openai"
        **kwargs: 各插件的配置参数，如 vector_store_config, graph_db_config 等

    Returns:
        已初始化的 Mem0Client 实例

    Example:
        client = await create_client(
            vector_store="chromadb",
            graph_db="nebula",
            embedding="openai",
            llm="openai",
        )
    """
    client = Mem0Client(
        vector_store=vector_store,
        graph_db=graph_db,
        embedding=embedding,
        llm=llm,
        **kwargs
    )
    await client.initialize()
    return client


async def create_client_with_instances(
    vector_store: Optional[VectorStoreInterface] = None,
    graph_db: Optional[GraphDatabaseInterface] = None,
    embedding: Optional[EmbeddingInterface] = None,
    llm: Optional[ModelInterface] = None,
) -> Mem0Client:
    """
    便捷函数：使用已初始化的组件实例创建客户端

    适用于需要自定义组件配置的高级场景

    Args:
        vector_store: 已初始化的向量存储实例
        graph_db: 已初始化的图数据库实例（可选）
        embedding: 已初始化的嵌入模型实例
        llm: 已初始化的大模型实例

    Returns:
        已初始化的 Mem0Client 实例

    Example:
        # 先初始化各个组件
        vector_store = ChromaDB()
        await vector_store.initialize({"collection_name": "memories"})

        embedding = OpenAIEmbedding()
        await embedding.initialize({"api_key": "xxx"})

        llm = OpenAIModel()
        await llm.initialize({"api_key": "xxx"})

        # 然后创建客户端
        client = await create_client_with_instances(
            vector_store=vector_store,
            embedding=embedding,
            llm=llm,
        )
    """
    client = Mem0Client(
        vector_store_instance=vector_store,
        graph_db_instance=graph_db,
        embedding_instance=embedding,
        llm_instance=llm,
    )
    await client.initialize()
    return client
