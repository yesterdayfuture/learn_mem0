# Mem0 - 智能记忆存储系统

一个从零实现的 mem0 记忆存储系统，支持自动根据对话历史进行总结、更新，并存储到向量库和图数据库中。

## 特性

- **自动记忆总结**: 根据对话历史自动提取关键信息
- **向量检索**: 基于语义相似度的记忆检索（ChromaDB）
- **图数据存储**: 实体关系图谱构建（NebulaGraph）
- **插件化架构**: 支持通过插件适配不同的向量库、图数据库和模型厂商
- **OpenAI 兼容**: 大模型和向量模型均支持 OpenAI 格式
- **可视化界面**: Web 界面可视化查看和管理数据
- **灵活使用**: 支持作为服务启动或直接作为客户端使用

## 整体运行流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Mem0 智能记忆存储系统                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   用户输入        │────▶│   对话处理        │────▶│   自动总结        │
│  (文本/消息列表)   │     │  add_messages    │     │  summarize       │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                              ┌────────────────────────────┼────────────────────────────┐
                              │                            │                            │
                              ▼                            ▼                            ▼
                    ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
                    │   向量库存储      │      │   图数据库存储    │      │   记忆合并/更新   │
                    │  (ChromaDB)      │      │  (NebulaGraph)   │      │  merge/update    │
                    │                  │      │                  │      │                  │
                    │  • 记忆向量       │      │  • 实体节点       │      │  • 相似度检查     │
                    │  • 语义搜索       │      │  • 关系边         │      │  • 内容合并       │
                    │  • 元数据         │      │  • 图谱查询       │      │  • 去重处理       │
                    └──────────────────┘      └──────────────────┘      └──────────────────┘
                              │                            │                            │
                              └────────────────────────────┼────────────────────────────┘
                                                           │
                                                           ▼
                                              ┌──────────────────┐
                                              │   记忆搜索        │
                                              │  search_memories │
                                              └────────┬─────────┘
                                                       │
                              ┌────────────────────────┼────────────────────────┐
                              │                        │                        │
                              ▼                        ▼                        ▼
                    ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
                    │   向量库搜索      │      │   图数据库搜索    │      │   混合排序        │
                    │  (语义相似度)     │      │  (实体匹配)      │      │  (综合评分)      │
                    └──────────────────┘      └──────────────────┘      └──────────────────┘
                                                                                │
                                                                                ▼
                                                                     ┌──────────────────┐
                                                                     │   返回记忆结果    │
                                                                     │  MemorySearchResult
                                                                     └──────────────────┘
```

### 详细数据流

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              对话处理流程 (add_messages)                              │
└─────────────────────────────────────────────────────────────────────────────────────┘

用户输入消息列表
       │
       ▼
┌──────────────┐
│ 消息列表格式  │  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 添加到缓冲区  │────▶│ 检查阈值     │────▶│ 触发总结     │
│ (按user_id)  │     │ (>=3条对话)  │     │              │
└──────────────┘     └──────┬───────┘     └──────┬───────┘
                            │                      │
                    未达到阈值│                      │达到阈值
                            │                      ▼
                            │             ┌──────────────┐
                            │             │ LLM生成总结  │
                            │             │              │
                            │             │ • 提取关键信息│
                            │             │ • 识别实体   │
                            │             │ • 发现关系   │
                            │             └──────┬───────┘
                            │                    │
                            │       ┌────────────┼────────────┐
                            │       │            │            │
                            │       ▼            ▼            ▼
                            │ ┌──────────┐ ┌──────────┐ ┌──────────┐
                            │ │向量库存储 │ │图数据库存│ │记忆合并  │
                            │ │          │ │储        │ │          │
                            │ │• 记忆向量 │ │• 实体节点 │ │• 相似度检查│
                            │ │• 元数据   │ │• 关系边   │ │• 内容合并  │
                            │ └──────────┘ └──────────┘ └──────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ 返回对话ID列表  │
                   └────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              记忆搜索流程 (search_memories)                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

搜索查询
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         双库并行搜索 (默认)                               │
│                    search_source: "both" | "vector" | "graph"            │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ├────────────────────────────┬────────────────────────────┐
    │                            │                            │
    ▼                            ▼                            ▼
┌──────────┐            ┌──────────┐                  ┌──────────┐
│向量库搜索 │            │图数据库搜索│                  │混合搜索  │
│          │            │          │                  │          │
│• 语义向量 │            │• 实体提取  │                  │• 两者结合 │
│• 相似度计算│            │• 实体匹配  │                  │• 综合评分 │
│• Top-K   │            │• 关系遍历  │                  │• 去重排序 │
└────┬─────┘            └────┬─────┘                  └────┬─────┘
     │                       │                            │
     └───────────────────────┼────────────────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │ 结果合并与排序    │
                   │                  │
                   │ • 按分数排序      │
                   │ • 去重处理        │
                   │ • 更新访问统计    │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ 返回搜索结果列表  │
                   │ MemorySearchResult│
                   └──────────────────┘
```

## 项目结构

```
zero_mem0/
│
├── main.py                          # 服务启动入口
├── visualize.py                     # 可视化工具启动入口
├── requirements.txt                 # 项目依赖
├── .env.example                     # 环境变量模板
├── README.md                        # 项目文档
│
├── mem0/                            # 核心包
│   ├── __init__.py                  # 包初始化
│   ├── client.py                    # Mem0Client 客户端（核心入口）
│   │                                 #   - 支持多种初始化方式
│   │                                 #   - 封装记忆管理功能
│   │                                 #   - 提供对话、搜索、聊天等接口
│   │
│   ├── config.py                    # 配置管理
│   │                                 #   - PluginRegistry 插件注册中心（单例）
│   │                                 #   - Mem0Config 配置类
│   │                                 #   - 支持动态加载插件
│   │
│   ├── core/                        # 核心逻辑
│   │   ├── __init__.py
│   │   ├── memory.py                # MemoryManager 记忆管理器（核心类）
│   │   │                                 #   - 对话缓冲区管理
│   │   │                                 #   - 自动总结逻辑
│   │   │                                 #   - 双库存储（向量库+图数据库）
│   │   │                                 #   - 记忆搜索与合并
│   │   │                                 #   - 记忆衰减与整合
│   │   │
│   │   └── models.py                # 数据模型
│   │                                 #   - Conversation 对话
│   │                                 #   - Memory 记忆
│   │                                 #   - MemoryType 记忆类型枚举
│   │                                 #   - MemorySearchResult 搜索结果
│   │
│   ├── plugins/                     # 插件系统
│   │   ├── __init__.py              # 导出插件接口
│   │   ├── base.py                  # PluginInterface 插件基础接口
│   │   │
│   │   ├── vector_stores/           # 向量存储插件
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # VectorStoreInterface 接口
│   │   │   │                                 #   - VectorRecord 向量记录
│   │   │   │                                 #   - SearchResult 搜索结果
│   │   │   │
│   │   │   └── chroma.py            # ChromaDB 适配器
│   │   │                                 #   - 向量存储与检索
│   │   │                                 #   - 元数据过滤
│   │   │
│   │   ├── graph_databases/         # 图数据库插件
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # GraphDatabaseInterface 接口
│   │   │   │                                 #   - Node 节点
│   │   │   │                                 #   - Relationship 关系
│   │   │   │                                 #   - Path 路径
│   │   │   │
│   │   │   └── nebula.py            # NebulaGraph 适配器
│   │   │                                 #   - 节点增删改查
│   │   │                                 #   - 关系管理
│   │   │                                 #   - 标签/边类型管理
│   │   │                                 #   - 模糊搜索
│   │   │
│   │   └── models/                  # 模型插件
│   │       ├── __init__.py
│   │       ├── base.py              # ModelInterface & EmbeddingInterface
│   │       │                                 #   - ChatMessage 消息
│   │       │                                 #   - ChatCompletion 完成结果
│   │       │                                 #   - 总结、实体提取、关系提取
│   │       │
│   │       └── openai_adapter.py    # OpenAI 适配器
│   │                                 #   - OpenAIModel 大模型
│   │                                 #   - OpenAIEmbedding 嵌入模型
│   │
│   ├── api/                         # FastAPI 服务
│   │   ├── __init__.py
│   │   ├── server.py                # FastAPI 服务入口
│   │   │                                 #   - REST API 端点
│   │   │                                 #   - 健康检查
│   │   │                                 #   - 对话、记忆、搜索接口
│   │   │
│   │   └── models.py                # API 数据模型
│   │                                 #   - Pydantic 模型定义
│   │
│   └── web/                         # 可视化界面
│       ├── __init__.py
│       └── visualization.py         # Web 可视化工具
│                                     #   - ChromaDB 数据查看
│                                     #   - NebulaGraph 图谱查看
│                                     #   - 实体搜索与关系展示
│
├── examples/                        # 使用示例
│   ├── __init__.py
│   ├── basic_usage.py               # 基础使用示例
│   ├── advanced_usage.py            # 高级功能示例
│   ├── client_usage.py              # 客户端使用示例（8种初始化方式）
│   └── graph_schema_management.py   # 图数据库 Schema 管理示例
│
└── tests/                           # 测试目录
    └── ...
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的配置
```

### 3. 使用方式

#### 方式一：作为服务启动（REST API）

```bash
python main.py
```

服务将在 http://localhost:8000 启动，API 文档访问 http://localhost:8000/docs

#### 方式二：直接使用客户端（无需启动服务）

**方式 2.1: 使用默认配置（从环境变量读取）**

```python
import asyncio
from mem0.client import Mem0Client

async def main():
    # 使用上下文管理器
    async with Mem0Client() as client:
        # 添加单条对话
        await client.add_conversation("user_001", "user", "我叫张三")

        # 添加对话列表（自动总结和存储）
        messages = [
            {"role": "user", "content": "我喜欢打篮球"},
            {"role": "assistant", "content": "篮球是很好的运动！"},
            {"role": "user", "content": "是的，我每周都打"}
        ]
        await client.add_messages("user_001", messages)

        # 搜索记忆（默认从向量库和图数据库搜索）
        results = await client.search_memories("用户的爱好", user_id="user_001")
        for result in results:
            print(f"{result.memory.content} (score: {result.score}, source: {result.match_type})")

asyncio.run(main())
```

**方式 2.2: 指定 provider 名称**

```python
from mem0.client import Mem0Client

async with Mem0Client(
    vector_store="chromadb",
    graph_db="nebula",
    embedding="openai",
    llm="openai",
) as client:
    # 使用客户端...
    pass
```

**方式 2.3: 使用完整配置字典**

```python
from mem0.client import Mem0Client

config = {
    "vector_store": {
        "provider": "chromadb",
        "config": {"collection_name": "my_memories", "persist_directory": "./data"}
    },
    "graph_db": {
        "provider": "nebula",
        "config": {
            "host": "127.0.0.1",
            "port": 9669,
            "user": "root",
            "password": "nebula",
            "space": "mem0"
        }
    },
    "embedding": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small", "api_key": "xxx"}
    },
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini", "api_key": "xxx"}
    }
}

async with Mem0Client(config) as client:
    # 使用客户端...
    pass
```

**方式 2.4: 传入已初始化的组件实例**

```python
from mem0.client import Mem0Client
from mem0.plugins.vector_stores.chroma import ChromaDB
from mem0.plugins.models.openai_adapter import OpenAIEmbedding, OpenAIModel

# 先初始化各个组件
vector_store = ChromaDB()
await vector_store.initialize({"collection_name": "memories"})

embedding = OpenAIEmbedding()
await embedding.initialize({"api_key": "xxx"})

llm = OpenAIModel()
await llm.initialize({"api_key": "xxx"})

# 然后创建客户端
async with Mem0Client(
    vector_store_instance=vector_store,
    embedding_instance=embedding,
    llm_instance=llm,
) as client:
    # 使用客户端...
    pass
```

#### 方式三：使用可视化界面

```bash
python visualize.py
```

Web 界面将在 http://localhost:8080 启动，可以可视化查看 ChromaDB 和 NebulaGraph 中的数据。

## 核心功能详解

### 对话处理

```python
# 添加单条对话
await client.add_conversation(
    user_id="user_001",
    role="user",
    content="我叫张三",
    session_id="session_001"
)

# 添加对话列表（自动总结）
messages = [
    {"role": "user", "content": "我叫张三"},
    {"role": "assistant", "content": "你好张三！很高兴认识你。"},
    {"role": "user", "content": "我喜欢打篮球"},
    {"role": "assistant", "content": "篮球是很好的运动！你打什么位置？"},
    {"role": "user", "content": "我打控球后卫"}
]
await client.add_messages("user_001", messages)
```

### 记忆搜索

```python
# 默认从向量库和图数据库同时搜索
results = await client.search_memories("用户的爱好", user_id="user_001")

# 仅从向量库搜索
results = await client.search_memories(
    "用户的爱好",
    user_id="user_001",
    search_source="vector"
)

# 仅从图数据库搜索
results = await client.search_memories(
    "用户的爱好",
    user_id="user_001",
    search_source="graph"
)

# 处理结果
for result in results:
    print(f"内容: {result.memory.content}")
    print(f"分数: {result.score}")
    print(f"来源: {result.match_type}")  # "vector", "graph", "hybrid"
```

### 带记忆的聊天

```python
# 自动添加消息、获取记忆、生成回复
response = await client.chat_with_memory(
    message="我最近在学习编程",
    user_id="user_001",
    include_memories=True
)
print(response)
```

### 图数据库 Schema 管理

```python
from mem0.client import Mem0Client

async with Mem0Client() as client:
    # 创建节点标签（实体类型）
    await client.graph_db.create_node_label(
        label="Person",
        properties={
            "name": "string",
            "age": "int",
            "description": "string"
        }
    )

    # 创建边标签（关系类型）
    await client.graph_db.create_edge_label(
        label="KNOWS",
        properties={
            "since": "date",
            "description": "string"
        }
    )

    # 查看所有标签详情
    node_labels = await client.graph_db.get_label_details("node")
    edge_labels = await client.graph_db.get_label_details("edge")

    for label, info in node_labels.items():
        print(f"标签: {label}")
        print(f"  属性: {info['properties']}")
        print(f"  实体数: {info['count']}")
```

## 文件详细说明

### 核心文件

| 文件 | 作用 | 关键类/函数 |
|------|------|------------|
| `mem0/client.py` | 客户端主入口，封装所有功能 | `Mem0Client`, `create_client`, `create_client_with_instances` |
| `mem0/config.py` | 配置管理和插件注册 | `PluginRegistry`, `Mem0Config`, `registry` |
| `mem0/core/memory.py` | 记忆管理核心逻辑 | `MemoryManager` |
| `mem0/core/models.py` | 数据模型定义 | `Conversation`, `Memory`, `MemoryType`, `MemorySearchResult` |

### 插件接口

| 文件 | 作用 | 关键类 |
|------|------|--------|
| `mem0/plugins/base.py` | 插件基础接口 | `PluginInterface` |
| `mem0/plugins/vector_stores/base.py` | 向量存储接口 | `VectorStoreInterface`, `VectorRecord`, `SearchResult` |
| `mem0/plugins/graph_databases/base.py` | 图数据库接口 | `GraphDatabaseInterface`, `Node`, `Relationship` |
| `mem0/plugins/models/base.py` | 模型接口 | `ModelInterface`, `EmbeddingInterface`, `ChatMessage` |

### 插件实现

| 文件 | 作用 | 关键类 |
|------|------|--------|
| `mem0/plugins/vector_stores/chroma.py` | ChromaDB 适配器 | `ChromaDB` |
| `mem0/plugins/graph_databases/nebula.py` | NebulaGraph 适配器 | `NebulaGraphDB` |
| `mem0/plugins/models/openai_adapter.py` | OpenAI 适配器 | `OpenAIModel`, `OpenAIEmbedding` |

### API 和 Web

| 文件 | 作用 | 关键类/函数 |
|------|------|------------|
| `mem0/api/server.py` | FastAPI 服务 | `create_app`, `app` |
| `mem0/api/models.py` | API 数据模型 | Pydantic 模型 |
| `mem0/web/visualization.py` | 可视化界面 | `create_visualization_app` |

## 插件开发

### 1. 创建向量存储插件

```python
from mem0.plugins.vector_stores.base import VectorStoreInterface, VectorRecord, SearchResult

class MyVectorStore(VectorStoreInterface):
    async def initialize(self, config: dict) -> None:
        # 初始化连接
        pass

    async def add(self, records: List[VectorRecord]) -> None:
        # 添加向量记录
        pass

    async def search(self, query_vector: List[float], top_k: int = 5, filters: dict = None) -> List[SearchResult]:
        # 搜索相似向量
        pass

    async def close(self) -> None:
        # 关闭连接
        pass

# 注册插件
from mem0.config import registry

registry.register_vector_store("my_store", MyVectorStore)
```

### 2. 创建图数据库插件

```python
from mem0.plugins.graph_databases.base import GraphDatabaseInterface, Node, Relationship

class MyGraphDB(GraphDatabaseInterface):
    async def initialize(self, config: dict) -> None:
        pass

    async def create_node(self, id: str, labels: List[str], properties: dict) -> Node:
        pass

    async def create_relationship(self, source_id: str, target_id: str, rel_type: str, properties: dict) -> Relationship:
        pass

    async def search_nodes(self, properties: dict) -> List[Node]:
        pass

    async def close(self) -> None:
        pass

# 注册插件
registry.register_graph_database("my_graph", MyGraphDB)
```

### 3. 创建模型插件

```python
from mem0.plugins.models.base import ModelInterface, ChatMessage, ChatCompletion

class MyModel(ModelInterface):
    async def initialize(self, config: dict) -> None:
        pass

    async def chat(self, messages: List[ChatMessage], **kwargs) -> ChatCompletion:
        # 实现聊天功能
        pass

    async def summarize(self, text: str, instruction: str = None) -> str:
        # 实现总结功能
        pass

    async def extract_entities(self, text: str) -> List[dict]:
        # 实现实体提取
        pass

    async def extract_relations(self, text: str, entities: List[dict]) -> List[dict]:
        # 实现关系提取
        pass

# 注册插件
registry.register_llm("my_model", MyModel)
```

## 搜索来源

- **vector**: 仅从向量库搜索（基于语义相似度）
- **graph**: 仅从图数据库搜索（基于实体匹配）
- **both**: 从两者搜索并合并结果（默认）

## 配置参数

```python
client.update_config(
    summary_threshold=3,        # 多少条对话后触发总结
    similarity_threshold=0.75,  # 相似度阈值
    max_memories_per_query=5,   # 每次查询最大记忆数
    decay_factor=0.95,          # 记忆衰减因子
)
```

## License

MIT
