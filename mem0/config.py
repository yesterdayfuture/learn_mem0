"""
Mem0 全局配置和插件注册中心模块

本模块提供：
1. PluginRegistry - 插件注册中心（单例模式），管理所有可用插件
2. Mem0Config - 配置类，支持从字典或环境变量加载配置
3. 默认插件注册函数

插件系统支持动态加载不同的数据库和模型厂商，无需修改核心代码即可扩展功能。

使用示例:
    # 获取全局注册中心
    from mem0.config import registry

    # 注册自定义插件
    registry.register_vector_store("my_store", MyVectorStore)
    registry.register_graph_database("my_graph", MyGraphDB)
    registry.register_embedding("my_embedding", MyEmbedding)
    registry.register_llm("my_llm", MyLLM)

    # 创建配置
    config = Mem0Config({
        "vector_store": {"provider": "chromadb", "config": {...}},
        "graph_db": {"provider": "nebula", "config": {...}},
        "embedding": {"provider": "openai", "config": {...}},
        "llm": {"provider": "openai", "config": {...}},
    })

    # 从环境变量创建配置
    config = Mem0Config.from_env()
"""

from typing import Any, Dict, List, Optional, Type

from .plugins.base import PluginInterface
from .plugins.graph_databases.base import GraphDatabaseInterface
from .plugins.models.base import EmbeddingInterface, ModelInterface
from .plugins.vector_stores.base import VectorStoreInterface


class PluginRegistry:
    """
    插件注册中心 - 管理所有可用插件（单例模式）

    使用单例模式确保全局只有一个注册中心实例，所有插件都注册到同一个中心。

    管理的插件类型：
    - 向量存储插件（如 ChromaDB）
    - 图数据库插件（如 NebulaGraph）
    - 嵌入模型插件（如 OpenAI Embedding）
    - 大模型插件（如 OpenAI GPT）

    Attributes:
        _instance: 单例实例
        _initialized: 是否已初始化标志
        _vector_stores: 向量存储插件字典 {name: plugin_class}
        _graph_databases: 图数据库插件字典 {name: plugin_class}
        _embeddings: 嵌入模型插件字典 {name: plugin_class}
        _llms: 大模型插件字典 {name: plugin_class}
    """

    _instance = None  # 单例实例引用

    def __new__(cls):
        """
        创建单例实例

        如果实例不存在则创建，否则返回已有实例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        初始化注册中心

        只在第一次创建实例时执行初始化
        """
        if self._initialized:
            return

        # 存储插件类的字典，键为插件名称，值为插件类
        self._vector_stores: Dict[str, Type[VectorStoreInterface]] = {}
        self._graph_databases: Dict[str, Type[GraphDatabaseInterface]] = {}
        self._embeddings: Dict[str, Type[EmbeddingInterface]] = {}
        self._llms: Dict[str, Type[ModelInterface]] = {}

        self._initialized = True

    # ==================== 注册方法 ====================

    def register_vector_store(
        self,
        name: str,
        plugin_class: Type[VectorStoreInterface],
    ) -> None:
        """
        注册向量存储插件

        Args:
            name: 插件名称，用于后续获取
            plugin_class: 向量存储插件类，必须实现 VectorStoreInterface

        Example:
            registry.register_vector_store("chromadb", ChromaDB)
        """
        self._vector_stores[name] = plugin_class

    def register_graph_database(
        self,
        name: str,
        plugin_class: Type[GraphDatabaseInterface],
    ) -> None:
        """
        注册图数据库插件

        Args:
            name: 插件名称，用于后续获取
            plugin_class: 图数据库插件类，必须实现 GraphDatabaseInterface

        Example:
            registry.register_graph_database("nebula", NebulaGraphDB)
        """
        self._graph_databases[name] = plugin_class

    def register_embedding(
        self,
        name: str,
        plugin_class: Type[EmbeddingInterface],
    ) -> None:
        """
        注册嵌入模型插件

        Args:
            name: 插件名称，用于后续获取
            plugin_class: 嵌入模型插件类，必须实现 EmbeddingInterface

        Example:
            registry.register_embedding("openai", OpenAIEmbedding)
        """
        self._embeddings[name] = plugin_class

    def register_llm(
        self,
        name: str,
        plugin_class: Type[ModelInterface],
    ) -> None:
        """
        注册大模型插件

        Args:
            name: 插件名称，用于后续获取
            plugin_class: 大模型插件类，必须实现 ModelInterface

        Example:
            registry.register_llm("openai", OpenAIModel)
        """
        self._llms[name] = plugin_class

    # ==================== 获取方法 ====================

    def get_vector_store(self, name: str) -> Optional[Type[VectorStoreInterface]]:
        """
        获取向量存储插件类

        Args:
            name: 插件名称

        Returns:
            插件类，如果不存在则返回 None
        """
        return self._vector_stores.get(name)

    def get_graph_database(self, name: str) -> Optional[Type[GraphDatabaseInterface]]:
        """
        获取图数据库插件类

        Args:
            name: 插件名称

        Returns:
            插件类，如果不存在则返回 None
        """
        return self._graph_databases.get(name)

    def get_embedding(self, name: str) -> Optional[Type[EmbeddingInterface]]:
        """
        获取嵌入模型插件类

        Args:
            name: 插件名称

        Returns:
            插件类，如果不存在则返回 None
        """
        return self._embeddings.get(name)

    def get_llm(self, name: str) -> Optional[Type[ModelInterface]]:
        """
        获取大模型插件类

        Args:
            name: 插件名称

        Returns:
            插件类，如果不存在则返回 None
        """
        return self._llms.get(name)

    # ==================== 列表方法 ====================

    def list_vector_stores(self) -> List[str]:
        """
        列出所有已注册的向量存储插件名称

        Returns:
            插件名称列表
        """
        return list(self._vector_stores.keys())

    def list_graph_databases(self) -> List[str]:
        """
        列出所有已注册的图数据库插件名称

        Returns:
            插件名称列表
        """
        return list(self._graph_databases.keys())

    def list_embeddings(self) -> List[str]:
        """
        列出所有已注册的嵌入模型插件名称

        Returns:
            插件名称列表
        """
        return list(self._embeddings.keys())

    def list_llms(self) -> List[str]:
        """
        列出所有已注册的大模型插件名称

        Returns:
            插件名称列表
        """
        return list(self._llms.keys())


# 全局插件注册中心实例
# 所有模块都通过导入这个实例来访问注册中心
registry = PluginRegistry()


class Mem0Config:
    """
    Mem0 全局配置类

    支持根据配置动态加载不同的数据库和模型。
    可以从字典或环境变量创建配置。

    配置结构:
        {
            "vector_store": {
                "provider": "chromadb",  # 或自定义插件名
                "config": {...}  # 插件特定配置
            },
            "graph_db": {
                "provider": "nebula",
                "config": {...}
            },
            "embedding": {
                "provider": "openai",
                "config": {...}
            },
            "llm": {
                "provider": "openai",
                "config": {...}
            }
        }

    Attributes:
        _config: 配置字典
        _registry: 插件注册中心引用
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化配置

        Args:
            config: 配置字典，如果为 None 则使用空配置
        """
        self._config = config or {}
        self._registry = registry

    def get_vector_store_config(self) -> Dict[str, Any]:
        """
        获取向量存储配置

        Returns:
            向量存储配置字典，包含 provider 和 config
            如果未配置，返回默认配置（chromadb）
        """
        return self._config.get("vector_store", {
            "provider": "chromadb",
            "config": {},
        })

    def get_graph_db_config(self) -> Dict[str, Any]:
        """
        获取图数据库配置

        Returns:
            图数据库配置字典，包含 provider 和 config
            如果未配置，返回默认配置（nebula）
        """
        return self._config.get("graph_db", {
            "provider": "nebula",
            "config": {},
        })

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        获取嵌入模型配置

        Returns:
            嵌入模型配置字典，包含 provider 和 config
            如果未配置，返回默认配置（openai）
        """
        return self._config.get("embedding", {
            "provider": "openai",
            "config": {},
        })

    def get_llm_config(self) -> Dict[str, Any]:
        """
        获取大模型配置

        Returns:
            大模型配置字典，包含 provider 和 config
            如果未配置，返回默认配置（openai）
        """
        return self._config.get("llm", {
            "provider": "openai",
            "config": {},
        })

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典

        Returns:
            配置字典的副本
        """
        return self._config.copy()

    @classmethod
    def from_env(cls) -> "Mem0Config":
        """
        从环境变量创建配置

        支持的环境变量：
        - VECTOR_STORE_PROVIDER: 向量存储提供商（默认 chromadb）
        - CHROMA_COLLECTION: ChromaDB 集合名（默认 mem0_memories）
        - CHROMA_PERSIST_DIR: ChromaDB 持久化目录（默认 ./data/chroma）
        - GRAPH_DB_PROVIDER: 图数据库提供商（默认 nebula）
        - NEBULA_HOST: NebulaGraph 主机（默认 127.0.0.1）
        - NEBULA_PORT: NebulaGraph 端口（默认 9669）
        - NEBULA_USER: NebulaGraph 用户名（默认 root）
        - NEBULA_PASSWORD: NebulaGraph 密码（默认 nebula）
        - NEBULA_SPACE: NebulaGraph 空间名（默认 mem0）
        - EMBEDDING_PROVIDER: 嵌入模型提供商（默认 openai）
        - OPENAI_API_KEY: OpenAI API 密钥
        - OPENAI_BASE_URL: OpenAI API 基础 URL
        - EMBEDDING_MODEL: 嵌入模型名称（默认 text-embedding-3-small）
        - EMBEDDING_DIMENSION: 嵌入维度（默认 1536）
        - LLM_PROVIDER: 大模型提供商（默认 openai）
        - LLM_MODEL: 大模型名称（默认 gpt-4o-mini）

        Returns:
            Mem0Config 实例
        """
        import os

        config = {
            "vector_store": {
                "provider": os.getenv("VECTOR_STORE_PROVIDER", "chromadb"),
                "config": {
                    "collection_name": os.getenv("CHROMA_COLLECTION", "mem0_memories"),
                    "persist_directory": os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
                },
            },
            "graph_db": {
                "provider": os.getenv("GRAPH_DB_PROVIDER", "nebula"),
                "config": {
                    "host": os.getenv("NEBULA_HOST", "127.0.0.1"),
                    "port": int(os.getenv("NEBULA_PORT", "9669")),
                    "username": os.getenv("NEBULA_USER", "root"),
                    "password": os.getenv("NEBULA_PASSWORD", "nebula"),
                    "space_name": os.getenv("NEBULA_SPACE", "mem0"),
                },
            },
            "embedding": {
                "provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
                "config": {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    "model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                    "dimension": int(os.getenv("EMBEDDING_DIMENSION", "1536")),
                },
            },
            "llm": {
                "provider": os.getenv("LLM_PROVIDER", "openai"),
                "config": {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                },
            },
        }

        return cls(config)


def register_default_plugins():
    """
    注册默认插件

    在模块导入时自动调用，注册系统内置的插件：
    - ChromaDB 向量存储
    - NebulaGraph 图数据库
    - OpenAI 嵌入模型
    - OpenAI 大模型
    """
    from .plugins.graph_databases.nebula import NebulaGraphStore
    from .plugins.models.openai_adapter import OpenAIEmbedding, OpenAIModel
    from .plugins.vector_stores.chroma import ChromaDBStore

    # 注册向量存储
    registry.register_vector_store("chromadb", ChromaDBStore)

    # 注册图数据库
    registry.register_graph_database("nebula", NebulaGraphStore)

    # 注册嵌入模型
    registry.register_embedding("openai", OpenAIEmbedding)

    # 注册大模型
    registry.register_llm("openai", OpenAIModel)


# 初始化时注册默认插件
# 这确保了在导入 config 模块时，默认插件就已经注册到注册中心
register_default_plugins()
