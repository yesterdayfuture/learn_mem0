"""
记忆管理器模块 - Mem0 系统的核心逻辑

本模块实现记忆系统的核心功能：
1. 对话缓冲区管理 - 临时存储对话，支持单条和批量消息列表
2. 自动总结 - 使用 LLM 从对话中提取关键信息，达到阈值自动触发
3. 双库存储 - 可配置存储到向量库（语义搜索）和/或图数据库（实体关系）
4. 记忆搜索 - 支持从向量库、图数据库或两者同时搜索
5. 记忆合并 - 合并相似记忆，避免重复
6. 记忆衰减 - 根据时间和访问频率调整记忆重要性

数据流：
    对话输入 -> 缓冲区 -> 触发总结 -> 实体/关系提取 -> 向量库存储（可选）
                                                      -> 图数据库存储（可选）
                                                      -> 记忆合并（如有相似）

主要特性：
- add_messages(): 支持 OpenAI 格式的消息列表 [{"role": "user", "content": "..."}, ...]
- 自动总结: 当对话数量达到阈值（默认3条）时自动触发
- 双库存储配置: 通过 store_to_vector 和 store_to_graph 控制存储位置
- 多源搜索: search_memories() 支持 search_source 参数 ("vector"/"graph"/"both")

使用示例:
    memory_manager = MemoryManager(
        vector_store=vector_store,
        graph_db=graph_db,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )

    # 添加单条对话
    conversation = await memory_manager.add_conversation(
        user_id="user_001",
        role="user",
        content="我叫张三"
    )

    # 批量添加消息（支持 OpenAI 格式）
    messages = [
        {"role": "user", "content": "你好，我叫张三"},
        {"role": "assistant", "content": "你好张三！"},
        {"role": "user", "content": "我是一名工程师"},
    ]
    memory = await memory_manager.add_messages("user_001", messages)

    # 搜索记忆（仅从向量库）
    results = await memory_manager.search_memories(
        "用户的名字",
        user_id="user_001",
        search_source="vector"
    )

    # 搜索记忆（仅从图数据库）
    results = await memory_manager.search_memories(
        "张三的朋友",
        user_id="user_001",
        search_source="graph"
    )

    # 搜索记忆（从两者搜索，默认）
    results = await memory_manager.search_memories(
        "用户的职业",
        user_id="user_001",
        search_source="both"
    )

    # 配置只存储到向量库
    memory_manager.update_config(store_to_vector=True, store_to_graph=False)
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..plugins.graph_databases.base import GraphDatabaseInterface
from ..plugins.models.base import ChatMessage, EmbeddingInterface, MessageRole, ModelInterface
from ..plugins.vector_stores.base import SearchResult, VectorRecord, VectorStoreInterface
from .models import Conversation, Memory, MemorySearchResult, MemoryType


class MemoryManager:
    """
    记忆管理器 - Mem0 系统的核心类

    负责管理记忆的完整生命周期：
    - 接收和缓冲对话（支持单条和批量消息列表）
    - 触发自动总结（达到阈值自动触发）
    - 提取实体和关系
    - 存储到向量库和/或图数据库（可配置）
    - 搜索和检索记忆（支持 vector/graph/both 三种来源）
    - 合并相似记忆
    - 应用记忆衰减

    主要功能：
    1. add_conversation(): 添加单条对话
    2. add_messages(): 批量添加消息列表（支持 OpenAI 格式）
    3. search_memories(): 搜索记忆（支持选择搜索来源）

    存储配置：
    - store_to_vector: 是否存储到向量库（默认 True）
    - store_to_graph: 是否存储到图数据库（默认 True）

    Attributes:
        vector_store: 向量存储接口实例
        graph_db: 图数据库接口实例
        embedding_model: 嵌入模型接口实例
        llm_model: 大模型接口实例
        config: 配置参数字典
        _conversation_buffer: 对话缓冲区，按 user_id 分组
    """

    def __init__(
        self,
        vector_store: VectorStoreInterface,
        graph_db: GraphDatabaseInterface,
        embedding_model: EmbeddingInterface,
        llm_model: ModelInterface,
    ):
        """
        初始化记忆管理器

        Args:
            vector_store: 向量存储接口实例，用于存储记忆向量
            graph_db: 图数据库接口实例，用于存储实体关系
            embedding_model: 嵌入模型接口实例，用于生成向量
            llm_model: 大模型接口实例，用于总结和实体提取
        """
        self.vector_store = vector_store
        self.graph_db = graph_db
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # 配置参数 - 可通过 update_config 方法修改
        self.config = {
            "summary_threshold": 10,  # 多少条对话后触发总结
            "max_conversation_history": 20,  # 最大保留的对话历史数
            "similarity_threshold": 0.75,  # 相似度阈值，用于判断是否为同一记忆
            "decay_factor": 0.95,  # 记忆衰减因子
            "max_memories_per_query": 5,  # 每次查询返回的最大记忆数
            "store_to_vector": True,  # 是否存储到向量库
            "store_to_graph": True,  # 是否存储到图数据库
        }

        # 临时对话缓存 (user_id -> List[Conversation])
        # 按用户ID分组存储，达到阈值后触发总结
        self._conversation_buffer: Dict[str, List[Conversation]] = {}

    def update_config(self, **kwargs) -> None:
        """
        更新配置参数

        Args:
            **kwargs: 要更新的配置项
                - summary_threshold: 触发总结的对话数量阈值（默认 3）
                - max_conversation_history: 最大保留的对话历史数（默认 10）
                - similarity_threshold: 记忆相似度阈值（0-1，默认 0.75）
                - decay_factor: 记忆衰减因子（0-1，默认 0.95）
                - max_memories_per_query: 每次查询最大记忆数（默认 5）
                - store_to_vector: 是否存储到向量库（默认 True）
                - store_to_graph: 是否存储到图数据库（默认 True）

        Example:
            memory_manager.update_config(
                summary_threshold=5,
                similarity_threshold=0.8,
                store_to_vector=True,
                store_to_graph=False,  # 只存储到向量库
            )
        """
        self.config.update(kwargs)

    async def add_conversation(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """
        添加对话记录

        将对话添加到缓冲区，当缓冲区达到阈值时自动触发总结

        Args:
            user_id: 用户唯一标识
            role: 对话角色，可选 "user" 或 "assistant"
            content: 对话内容
            session_id: 会话ID（可选）
            metadata: 额外元数据（可选）

        Returns:
            创建的对话对象
        """
        # 创建对话对象
        conversation = Conversation(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        # 添加到缓冲区
        if user_id not in self._conversation_buffer:
            self._conversation_buffer[user_id] = []

        self._conversation_buffer[user_id].append(conversation)

        # 检查是否需要触发总结
        if len(self._conversation_buffer[user_id]) >= self.config["summary_threshold"]:
            await self._process_conversations(user_id, session_id)

        return conversation

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

        Example:
            messages = [
                {"role": "user", "content": "你好，我叫张三"},
                {"role": "assistant", "content": "你好张三！很高兴认识你。"},
                {"role": "user", "content": "我是一名软件工程师"},
            ]
            memory = await memory_manager.add_messages("user_001", messages)
        """
        # 将消息转换为 Conversation 对象并添加到缓冲区
        for msg in messages:
            conversation = Conversation(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                metadata=metadata or {},
            )

            if user_id not in self._conversation_buffer:
                self._conversation_buffer[user_id] = []

            self._conversation_buffer[user_id].append(conversation)

        # 检查是否需要触发总结
        if len(self._conversation_buffer[user_id]) >= self.config["summary_threshold"]:
            return await self._process_conversations(user_id, session_id)

        return None

    async def _process_conversations(
        self, user_id: str, session_id: Optional[str] = None
    ) -> Optional[Memory]:
        """
        处理对话缓冲区，生成记忆（私有方法）

        流程：
        1. 格式化对话文本
        2. 使用 LLM 生成总结
        3. 提取实体和关系
        4. 创建记忆对象
        5. 检查是否需要合并到现有记忆
        6. 保存到向量库和图数据库
        7. 清空缓冲区

        Args:
            user_id: 用户ID
            session_id: 会话ID（可选）

        Returns:
            生成的记忆对象，如果缓冲区为空则返回 None
        """
        # 获取当前用户的对话缓冲区
        conversations = self._conversation_buffer.get(user_id, [])
        if not conversations:
            return None

        # 1. 构建对话文本
        conversation_text = self._format_conversations(conversations)

        # 2. 生成总结
        summary = await self.llm_model.summarize(
            conversation_text,
            instruction="请总结以下对话中的关键信息，包括用户提到的个人信息、偏好、重要事件等。保持简洁但信息完整。",
        )

        # 3. 提取实体和关系
        entities = await self.llm_model.extract_entities(summary)
        relations = await self.llm_model.extract_relations(summary, entities)

        print(f"  总结: {summary}")
        print(f"  实体: {entities}")
        print(f"  关系: {relations}")

        # 4. 创建记忆对象
        memory = Memory(
            content=summary,
            memory_type=MemoryType.EPISODIC,  # 默认为情景记忆
            user_id=user_id,
            session_id=session_id,
            entities=entities,
            relations=relations,
            source_conversations=[c.id for c in conversations],  # 记录来源对话
            importance=self._calculate_importance(summary, entities),  # 计算重要性
        )

        # 5. 检查是否需要合并到现有记忆
        existing_memory = await self._find_similar_memory(user_id, summary)

        if existing_memory:
            # 合并记忆
            merged_memory = await self._merge_memories(existing_memory, memory)
            await self._update_memory(merged_memory)
        else:
            # 保存新记忆
            await self._save_memory(memory)

        # 6. 清空缓冲区
        self._conversation_buffer[user_id] = []

        return memory

    def _format_conversations(self, conversations: List[Conversation]) -> str:
        """
        格式化对话列表为文本（私有方法）

        将对话列表转换为 LLM 可理解的文本格式

        Args:
            conversations: 对话列表

        Returns:
            格式化后的对话文本
        """
        lines = []
        for conv in conversations:
            prefix = "用户" if conv.role == "user" else "助手"
            lines.append(f"{prefix}: {conv.content}")
        return "\n".join(lines)

    def _calculate_importance(self, summary: str, entities: List[Dict]) -> float:
        """
        计算记忆重要性分数（私有方法）

        基于以下因素计算：
        - 实体数量：实体越多，重要性越高
        - 文本长度：适中长度（100-500字）得分更高
        - 关键词：包含关键信息词加分

        Args:
            summary: 总结文本
            entities: 实体列表

        Returns:
            重要性分数（0-1）
        """
        # 基础分数
        score = 0.5

        # 实体越多，重要性越高（最多加0.2）
        score += min(len(entities) * 0.05, 0.2)

        # 文本长度适中较好（100-500字）
        length = len(summary)
        if 100 <= length <= 500:
            score += 0.1
        elif length > 500:
            score += 0.05

        # 包含关键信息词加分
        keywords = ["喜欢", "讨厌", "重要", "必须", "总是", "从不", "计划", "目标"]
        for keyword in keywords:
            if keyword in summary:
                score += 0.02

        # 确保分数在 0-1 范围内
        return min(score, 1.0)

    async def _find_similar_memory(self, user_id: str, content: str) -> Optional[Memory]:
        """
        查找相似的记忆（私有方法）

        使用向量相似度搜索，找到与当前内容最相似的记忆

        Args:
            user_id: 用户ID
            content: 要比较的内容

        Returns:
            相似的记忆对象，如果未找到则返回 None
        """
        # 获取查询向量
        query_vector = await self.embedding_model.embed_query(content)

        # 在向量库中搜索
        results = await self.vector_store.search(
            query_vector=query_vector,
            top_k=1,
            filters={"user_id": user_id},
        )

        # 检查相似度是否超过阈值
        if results and results[0].score >= self.config["similarity_threshold"]:
            # 找到相似记忆，从元数据中重建 Memory 对象
            metadata = results[0].metadata
            return Memory.from_dict(metadata)

        return None

    async def _merge_memories(self, existing: Memory, new: Memory) -> Memory:
        """
        合并两个记忆（私有方法）

        使用 LLM 合并两段记忆的内容，去除重复信息

        Args:
            existing: 已有记忆
            new: 新记忆

        Returns:
            合并后的记忆对象
        """
        # 使用 LLM 合并内容
        merge_prompt = f"""请合并以下两段关于同一主题的记忆，去除重复信息，保留所有重要细节：

现有记忆：
{existing.content}

新记忆：
{new.content}

请输出合并后的记忆内容："""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="你是一个专业的信息整合助手。"),
            ChatMessage(role=MessageRole.USER, content=merge_prompt),
        ]

        result = await self.llm_model.chat(messages, temperature=0.3)
        merged_content = result.content.strip()

        # 合并实体和关系
        all_entities = existing.entities + new.entities
        all_relations = existing.relations + new.relations

        # 去重实体
        seen_entities = set()
        unique_entities = []
        for e in all_entities:
            key = (e.get("name"), e.get("type"))
            if key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(e)

        # 去重关系
        seen_relations = set()
        unique_relations = []
        for r in all_relations:
            key = (r.get("source"), r.get("target"), r.get("type"))
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(r)

        # 更新记忆
        existing.content = merged_content
        existing.entities = unique_entities
        existing.relations = unique_relations
        existing.updated_at = datetime.now()
        existing.importance = max(existing.importance, new.importance)
        existing.source_conversations.extend(new.source_conversations)

        return existing

    def _memory_to_vector_metadata(self, memory: Memory) -> Dict[str, Any]:
        """
        将 Memory 对象转换为适合向量库存储的扁平化元数据

        ChromaDB 要求元数据值必须是简单的标量类型（str, int, float, bool），
        不支持嵌套的字典或列表。因此需要将复杂类型转换为 JSON 字符串。

        Args:
            memory: 记忆对象

        Returns:
            扁平化的元数据字典
        """
        metadata = {
            "id": memory.id,
            "content": memory.content,
            "memory_type": memory.memory_type.value,
            "user_id": memory.user_id,
            "session_id": memory.session_id or "",
            "importance": memory.importance,
            "access_count": memory.access_count,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
        }

        # 将复杂类型转换为 JSON 字符串
        if memory.entities:
            metadata["entities"] = json.dumps(memory.entities, ensure_ascii=False)
        else:
            metadata["entities"] = "[]"

        if memory.relations:
            metadata["relations"] = json.dumps(memory.relations, ensure_ascii=False)
        else:
            metadata["relations"] = "[]"

        if memory.source_conversations:
            metadata["source_conversations"] = json.dumps(memory.source_conversations, ensure_ascii=False)
        else:
            metadata["source_conversations"] = "[]"

        if memory.last_accessed:
            metadata["last_accessed"] = memory.last_accessed.isoformat()
        else:
            metadata["last_accessed"] = ""

        return metadata

    async def _save_memory(self, memory: Memory) -> None:
        """
        保存记忆到存储（私有方法）

        根据配置同时或分别保存到：
        1. 向量库 - 用于语义搜索
        2. 图数据库 - 用于实体关系查询

        可通过配置控制存储位置：
        - store_to_vector: 是否存储到向量库（默认 True）
        - store_to_graph: 是否存储到图数据库（默认 True）

        Args:
            memory: 要保存的记忆对象
        """
        # 1. 生成向量
        memory.vector = await self.embedding_model.embed_query(memory.content)

        # 2. 保存到向量库（如果启用）
        if self.config.get("store_to_vector", True):
            vector_record = VectorRecord(
                id=memory.id,
                vector=memory.vector,
                metadata=self._memory_to_vector_metadata(memory),
                text=memory.content,
            )
            await self.vector_store.add([vector_record])

        # 3. 保存到图数据库（如果启用且图数据库可用）
        if self.config.get("store_to_graph", True) and self.graph_db:
            await self._save_to_graph(memory)

    async def _update_memory(self, memory: Memory) -> None:
        """
        更新已有记忆（私有方法）

        根据配置重新生成向量并更新向量库和/或图数据库

        Args:
            memory: 要更新的记忆对象
        """
        # 1. 重新生成向量
        memory.vector = await self.embedding_model.embed_query(memory.content)

        # 2. 更新向量库（如果启用）
        if self.config.get("store_to_vector", True):
            vector_record = VectorRecord(
                id=memory.id,
                vector=memory.vector,
                metadata=self._memory_to_vector_metadata(memory),
                text=memory.content,
            )
            await self.vector_store.update([vector_record])

        # 3. 更新图数据库（如果启用且图数据库可用）
        if self.config.get("store_to_graph", True) and self.graph_db:
            # 删除旧实体和关系
            for entity in memory.entities:
                await self.graph_db.delete_node(entity.get("name", ""))

            # 保存新实体和关系
            await self._save_to_graph(memory)

    async def _save_to_graph(self, memory: Memory) -> None:
        """
        保存记忆到图数据库（私有方法）

        创建实体节点和关系边

        Args:
            memory: 要保存的记忆对象
        """
        # 创建实体节点
        entity_map = {}
        for entity in memory.entities:
            # 处理字符串类型的实体（转换为字典格式）
            if isinstance(entity, str):
                entity_name = entity
                entity_id = entity
                entity_type = "Concept"
            elif isinstance(entity, dict):
                entity_name = entity.get("name", "")
                entity_id = entity.get("name", str(uuid.uuid4()))
                entity_type = entity.get("type", "Concept")
            else:
                continue

            if not entity_name:
                continue

            try:
                node = await self.graph_db.create_node(
                    id=entity_id,
                    labels=[entity_type],
                    properties={
                        "name": entity_name,
                        "description": entity.get("description", "") if isinstance(entity, dict) else "",
                        "memory_id": memory.id,  # 关联记忆ID
                    },
                )
                entity_map[entity_id] = node
            except Exception as e:
                # 实体可能已存在，尝试获取
                node = await self.graph_db.get_node(entity_id)
                if node:
                    entity_map[entity_id] = node

        # 创建关系
        for relation in memory.relations:
            # 处理字符串类型的关系（跳过）
            if isinstance(relation, str):
                continue

            source = relation.get("source", "")
            target = relation.get("target", "")
            rel_type = relation.get("type", "RELATED_TO")

            if source in entity_map and target in entity_map:
                try:
                    await self.graph_db.create_relationship(
                        source_id=source,
                        target_id=target,
                        rel_type=rel_type,
                        properties={
                            "description": relation.get("description", ""),
                            "memory_id": memory.id,
                        },
                    )
                except Exception as e:
                    pass  # 忽略关系创建失败

    async def search_memories(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
        search_source: str = "both",
    ) -> List[MemorySearchResult]:
        """
        搜索相关记忆

        支持从向量库、图数据库或两者同时搜索，并合并排序结果。

        Args:
            query: 搜索查询文本
            user_id: 用户ID
            top_k: 返回结果数量
            memory_type: 记忆类型过滤（可选）
            search_source: 搜索来源，可选值：
                - "vector": 仅从向量库搜索
                - "graph": 仅从图数据库搜索
                - "both": 从两者搜索并合并结果（默认）

        Returns:
            搜索结果列表，每个结果包含：
                - memory: 记忆对象
                - score: 匹配分数（0-1）
                - match_type: 匹配类型（"vector"、"graph" 或 "hybrid"）

        Example:
            # 仅从向量库搜索
            results = await memory_manager.search_memories(
                query="用户的爱好",
                user_id="user_001",
                search_source="vector"
            )

            # 仅从图数据库搜索
            results = await memory_manager.search_memories(
                query="张三的朋友",
                user_id="user_001",
                search_source="graph"
            )

            # 从两者搜索（默认）
            results = await memory_manager.search_memories(
                query="用户的职业",
                user_id="user_001",
                search_source="both"
            )
        """
        memory_scores: Dict[str, Tuple[Memory, float, str]] = {}

        # 1. 向量搜索（如果启用）
        if search_source in ("vector", "both"):
            query_vector = await self.embedding_model.embed_query(query)

            # 构建过滤条件
            filters = {"user_id": user_id}
            if memory_type:
                filters["memory_type"] = memory_type.value

            vector_results = await self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k * 2,  # 获取更多结果用于混合排序
                filters=filters,
            )

            # 添加向量搜索结果
            for result in vector_results:
                memory = Memory.from_dict(result.metadata)
                memory_scores[memory.id] = (memory, result.score, "vector")

        # 2. 图数据库搜索（如果启用且图数据库可用）
        if search_source in ("graph", "both") and self.graph_db:
            entities = await self.llm_model.extract_entities(query)
            graph_memories = []

            for entity in entities:
                entity_name = entity.get("name", "")
                if entity_name:
                    # 在图中查找相关节点
                    nodes = await self.graph_db.search_nodes(
                        properties={"name": entity_name}
                    )
                    for node in nodes:
                        memory_id = node.properties.get("memory_id")
                        if memory_id:
                            # 从向量库获取完整记忆
                            record = await self.vector_store.get(memory_id)
                            if record:
                                graph_memories.append((record, 0.8))  # 图搜索默认分数

            # 添加/合并图搜索结果
            for record, score in graph_memories:
                memory = Memory.from_dict(record.metadata)
                if memory.id in memory_scores:
                    # 已存在，升级为 hybrid，取最高分数
                    existing = memory_scores[memory.id]
                    if score > existing[1]:
                        memory_scores[memory.id] = (memory, score, "hybrid")
                else:
                    memory_scores[memory.id] = (memory, score, "graph")

        # 3. 转换为搜索结果并排序
        results = [
            MemorySearchResult(memory=m, score=s, match_type=t)
            for m, s, t in memory_scores.values()
        ]
        results.sort(key=lambda x: x.score, reverse=True)

        # 4. 更新访问统计
        for result in results[:top_k]:
            await self._update_access_stats(result.memory)

        return results[:top_k]

    async def _update_access_stats(self, memory: Memory) -> None:
        """
        更新记忆访问统计（私有方法）

        增加访问计数并更新最后访问时间

        Args:
            memory: 要更新的记忆对象
        """
        memory.access_count += 1
        memory.last_accessed = datetime.now()

        # 如果向量为空，重新生成
        if not memory.vector:
            memory.vector = await self.embedding_model.embed_query(memory.content)

        # 更新向量库中的元数据（使用扁平化元数据格式）
        vector_record = VectorRecord(
            id=memory.id,
            vector=memory.vector,
            metadata=self._memory_to_vector_metadata(memory),
            text=memory.content,
        )
        await self.vector_store.update([vector_record])

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        获取单个记忆

        Args:
            memory_id: 记忆唯一标识

        Returns:
            记忆对象，如果不存在则返回 None
        """
        record = await self.vector_store.get(memory_id)
        if record:
            memory = Memory.from_dict(record.metadata)
            await self._update_access_stats(memory)
            return memory
        return None

    async def delete_memory(self, memory_id: str) -> bool:
        """
        删除记忆

        同时从向量库和图数据库中删除

        Args:
            memory_id: 记忆唯一标识

        Returns:
            是否删除成功
        """
        # 获取记忆信息
        memory = await self.get_memory(memory_id)
        if not memory:
            return False

        # 1. 从向量库删除
        await self.vector_store.delete([memory_id])

        # 2. 从图数据库删除相关实体和关系
        for entity in memory.entities:
            entity_id = entity.get("name", "")
            if entity_id:
                await self.graph_db.delete_node(entity_id)

        return True

    async def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """
        获取用户的所有记忆

        Args:
            user_id: 用户ID
            memory_type: 记忆类型过滤（可选）
            limit: 返回数量限制

        Returns:
            记忆对象列表

        Note:
            这里简化处理，实际应该实现向量库的列表查询
        """
        # 暂时返回空列表，可以通过图数据库查询
        return []

    async def consolidate_memories(self, user_id: str) -> None:
        """
        整合用户的记忆（去重、合并相似记忆）

        流程：
        1. 获取用户所有记忆
        2. 按相似度分组
        3. 合并每组记忆

        Args:
            user_id: 用户ID
        """
        # 获取用户所有记忆
        memories = await self.get_user_memories(user_id)

        if len(memories) < 2:
            return

        # 按相似度分组
        groups = []
        used = set()

        for i, mem1 in enumerate(memories):
            if mem1.id in used:
                continue

            group = [mem1]
            used.add(mem1.id)

            for mem2 in memories[i + 1 :]:
                if mem2.id in used:
                    continue

                # 计算相似度
                similarity = await self._calculate_similarity(mem1, mem2)
                if similarity >= self.config["similarity_threshold"]:
                    group.append(mem2)
                    used.add(mem2.id)

            groups.append(group)

        # 合并每组记忆
        for group in groups:
            if len(group) > 1:
                merged = group[0]
                for mem in group[1:]:
                    merged = await self._merge_memories(merged, mem)
                    await self.delete_memory(mem.id)
                await self._update_memory(merged)

    async def _calculate_similarity(self, mem1: Memory, mem2: Memory) -> float:
        """
        计算两个记忆的相似度（私有方法）

        使用向量余弦相似度

        Args:
            mem1: 第一个记忆
            mem2: 第二个记忆

        Returns:
            相似度分数（0-1）
        """
        if mem1.vector and mem2.vector:
            import math

            # 计算点积
            dot_product = sum(a * b for a, b in zip(mem1.vector, mem2.vector))
            # 计算模长
            norm1 = math.sqrt(sum(a * a for a in mem1.vector))
            norm2 = math.sqrt(sum(b * b for b in mem2.vector))

            if norm1 > 0 and norm2 > 0:
                return dot_product / (norm1 * norm2)

        return 0.0

    async def apply_memory_decay(self, user_id: str) -> None:
        """
        应用记忆衰减

        根据时间和访问频率调整记忆重要性：
        - 时间衰减：越久未访问的记忆重要性越低
        - 访问提升：访问次数多的记忆重要性提升

        Args:
            user_id: 用户ID
        """
        memories = await self.get_user_memories(user_id)
        now = datetime.now()

        for memory in memories:
            # 基于时间衰减
            days_since_access = (
                (now - memory.last_accessed).days if memory.last_accessed else 30
            )
            time_decay = self.config["decay_factor"] ** days_since_access

            # 基于访问频率调整
            access_boost = min(memory.access_count * 0.05, 0.2)

            # 计算新的重要性
            new_importance = memory.importance * time_decay + access_boost
            memory.importance = max(0.1, min(new_importance, 1.0))

            # 如果重要性太低，可以考虑删除或归档
            if memory.importance < 0.2:
                # 可以选择删除或归档
                pass

            # 更新存储
            await self._update_memory(memory)

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
        """
        memories = await self.search_memories(query, user_id, top_k=5)

        if not memories:
            return ""

        # 按重要性排序
        memories.sort(key=lambda x: x.memory.importance, reverse=True)

        # 构建上下文
        context_parts = []
        total_length = 0

        for result in memories:
            memory_text = f"- {result.memory.content}"
            if total_length + len(memory_text) > max_tokens:
                break
            context_parts.append(memory_text)
            total_length += len(memory_text)

        if context_parts:
            return "相关记忆：\n" + "\n".join(context_parts)

        return ""

    async def force_summarize(self, user_id: str, session_id: Optional[str] = None) -> Optional[Memory]:
        """
        强制总结当前缓冲区中的对话

        即使缓冲区未达到阈值，也立即触发总结

        Args:
            user_id: 用户ID
            session_id: 会话ID（可选）

        Returns:
            生成的记忆对象，如果缓冲区为空则返回 None
        """
        return await self._process_conversations(user_id, session_id)
