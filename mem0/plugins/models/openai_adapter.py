"""
OpenAI 模型适配器模块

本模块实现 OpenAI API 的适配器，提供大模型对话和嵌入向量功能。

支持的功能：
- 对话生成（chat completions）
- 文本总结
- 命名实体识别（NER）
- 关系提取
- 文本嵌入（embeddings）

依赖安装:
    pip install openai

环境变量:
    OPENAI_API_KEY: OpenAI API 密钥
    OPENAI_BASE_URL: API 基础 URL（可选，用于代理或第三方兼容服务）

使用示例:
    from mem0.plugins.models.openai_adapter import OpenAIModel, OpenAIEmbedding

    # 初始化大模型
    llm = OpenAIModel({
        "api_key": "sk-xxx",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1"
    })

    # 对话
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="你是助手"),
        ChatMessage(role=MessageRole.USER, content="你好")
    ]
    response = await llm.chat(messages)
    print(response.content)

    # 初始化嵌入模型
    embedding = OpenAIEmbedding({
        "api_key": "sk-xxx",
        "model": "text-embedding-3-small"
    })

    # 生成向量
    vector = await embedding.embed_query("hello world")
"""

import json
from typing import Any, Dict, List, Optional

import openai

from .base import ChatMessage, ChatResponse, EmbeddingInterface, MessageRole, ModelInterface


class OpenAIModel(ModelInterface):
    """
    OpenAI 大模型适配器

    实现 ModelInterface 接口，提供基于 OpenAI API 的大模型功能。

    Attributes:
        config: 配置字典
        client: OpenAI 客户端实例
        model: 使用的模型名称

    配置参数:
        - api_key: OpenAI API 密钥（必需）
        - model: 模型名称（默认 "gpt-4o-mini"）
        - base_url: API 基础 URL（可选）
        - temperature: 默认温度参数（默认 0.7）
        - max_tokens: 默认最大 token 数（可选）
    """

    def __init__(self):
        """
        初始化 OpenAI 模型

        注意：实际初始化通过 initialize() 方法完成
        """
        super().__init__()
        self.client = None
        self.model = "gpt-4o-mini"
        self.default_temperature = 0.7
        self.default_max_tokens = None

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        异步初始化 OpenAI 模型

        Args:
            config: 配置字典，包含 api_key、model、base_url 等
        """
        self.config = config

        # 获取配置参数
        api_key = config.get("api_key")
        base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model = config.get("model", "gpt-4o-mini")
        self.default_temperature = config.get("temperature", 0.7)
        self.default_max_tokens = config.get("max_tokens")

        # 初始化 OpenAI 客户端
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def health_check(self) -> bool:
        """
        健康检查

        通过简单的 API 调用检查连接是否正常。

        Returns:
            True 表示健康，False 表示异常
        """
        try:
            # 尝试列出模型来检查连接
            await self.client.models.list()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """
        关闭连接

        关闭 OpenAI 客户端的 HTTP 连接池。
        """
        await self.client.close()

    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResponse:
        """
        对话生成

        调用 OpenAI Chat Completions API 生成回复。

        Args:
            messages: 消息列表
            temperature: 温度参数（默认 0.7）
            max_tokens: 最大 token 数（可选）
            **kwargs: 额外的 API 参数

        Returns:
            模型回复对象

        Example:
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content="你是助手"),
                ChatMessage(role=MessageRole.USER, content="你好")
            ]
            response = await llm.chat(messages, temperature=0.5)
            print(response.content)
        """
        # 转换消息格式
        openai_messages = [msg.to_dict() for msg in messages]

        # 调用 API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            **kwargs
        )

        # 检查响应是否有效
        if not response or not response.choices or len(response.choices) == 0:
            raise RuntimeError("OpenAI API 返回空响应或无选择结果")

        # 解析响应
        choice = response.choices[0]
        if not choice.message or not choice.message.content:
            raise RuntimeError("OpenAI API 返回的消息内容为空")

        return ChatResponse(
            content=choice.message.content,
            role=MessageRole(choice.message.role),
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
            raw_response=response,
        )

    async def summarize(
        self,
        text: str,
        instruction: Optional[str] = None,
        max_length: int = 200,
    ) -> str:
        """
        文本总结

        使用 LLM 将长文本压缩为简洁摘要。

        Args:
            text: 要总结的文本
            instruction: 总结指令（可选）
            max_length: 最大摘要长度（默认 200 字）

        Returns:
            摘要文本

        Example:
            summary = await llm.summarize(
                text="很长的文本...",
                instruction="请总结关键信息",
                max_length=100
            )
        """
        default_instruction = f"请总结以下文本的关键信息，限制在 {max_length} 字以内："
        prompt = f"{instruction or default_instruction}\n\n{text}"

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="你是一个专业的文本总结助手。"),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        response = await self.chat(messages, temperature=0.3)
        return response.content.strip()

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        实体提取

        使用 LLM 从文本中提取命名实体。

        Args:
            text: 输入文本

        Returns:
            实体列表，每个实体包含 name、type、description 字段

        Example:
            entities = await llm.extract_entities("张三在北京工作")
            # 返回: [{"name": "张三", "type": "Person"}, {"name": "北京", "type": "Location"}]
        """
        prompt = f"""请从以下文本中提取命名实体，并以 JSON 格式返回。

文本：{text}

要求：
1. 提取人名、地名、组织名、时间、概念等实体
2. 每个实体包含 name（名称）、type（类型）、description（描述）字段
3. 只返回 JSON 数组，不要其他内容

示例输出：
[
    {{"name": "张三", "type": "Person", "description": "一个名叫张三的人"}},
    {{"name": "北京", "type": "Location", "description": "中国的首都"}}
]"""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="你是一个专业的命名实体识别助手。"),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        response = await self.chat(messages, temperature=0.3)

        try:
            # 尝试解析 JSON
            content = response.content.strip()
            # 移除可能的 markdown 代码块标记
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            entities = json.loads(content.strip())
            return entities if isinstance(entities, list) else []
        except json.JSONDecodeError:
            # 如果解析失败，返回空列表
            return []

    async def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        关系提取

        使用 LLM 从文本中提取实体之间的关系。

        Args:
            text: 输入文本
            entities: 已提取的实体列表

        Returns:
            关系列表，每个关系包含 source、target、type、description 字段

        Example:
            entities = [{"name": "张三"}, {"name": "李四"}]
            relations = await llm.extract_relations("张三和李四是朋友", entities)
            # 返回: [{"source": "张三", "target": "李四", "type": "FRIEND"}]
        """
        entity_names = [e.get("name", "") for e in entities]
        entity_list = ", ".join(entity_names)

        prompt = f"""请从以下文本中提取实体之间的关系，并以 JSON 格式返回。

文本：{text}

已知实体：{entity_list}

要求：
1. 只提取上述实体之间的关系
2. 每个关系包含 source（源实体）、target（目标实体）、type（关系类型）、description（描述）字段
3. 关系类型使用大写英文，如 FRIEND、WORKS_AT、LOCATED_IN 等
4. 只返回 JSON 数组，不要其他内容

示例输出：
[
    {{"source": "张三", "target": "李四", "type": "FRIEND", "description": "张三和李四是朋友关系"}}
]"""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="你是一个专业的关系提取助手。"),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        response = await self.chat(messages, temperature=0.3)

        try:
            # 尝试解析 JSON
            content = response.content.strip()
            # 移除可能的 markdown 代码块标记
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            relations = json.loads(content.strip())
            return relations if isinstance(relations, list) else []
        except json.JSONDecodeError:
            # 如果解析失败，返回空列表
            return []


class OpenAIEmbedding(EmbeddingInterface):
    """
    OpenAI 嵌入模型适配器

    实现 EmbeddingInterface 接口，提供基于 OpenAI API 的文本嵌入功能。

    Attributes:
        config: 配置字典
        client: OpenAI 客户端实例
        model: 使用的嵌入模型名称
        dimension: 向量维度

    配置参数:
        - api_key: OpenAI API 密钥（必需）
        - model: 嵌入模型名称（默认 "text-embedding-3-small"）
        - base_url: API 基础 URL（可选）
        - dimension: 向量维度（可选，用于 text-embedding-3 系列）
    """

    def __init__(self):
        """
        初始化 OpenAI 嵌入模型

        注意：实际初始化通过 initialize() 方法完成
        """
        super().__init__()
        self.client = None
        self.model = "text-embedding-3-small"
        self.dimension = None

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        异步初始化 OpenAI 嵌入模型

        Args:
            config: 配置字典，包含 api_key、model、base_url 等
        """
        self.config = config

        # 获取配置参数
        api_key = config.get("api_key")
        base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model = config.get("model", "text-embedding-3-small")
        self.dimension = config.get("dimension")

        # 初始化 OpenAI 客户端
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def health_check(self) -> bool:
        """
        健康检查

        通过简单的 API 调用检查连接是否正常。

        Returns:
            True 表示健康，False 表示异常
        """
        try:
            # 尝试生成一个简单文本的嵌入
            await self.embed_query("test")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """
        关闭连接

        关闭 OpenAI 客户端的 HTTP 连接池。
        """
        await self.client.close()

    async def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本

        将单个文本转换为向量表示。

        Args:
            text: 输入文本

        Returns:
            向量（浮点数列表）

        Example:
            vector = await embedding.embed_query("hello world")
            print(len(vector))  # 输出向量维度
        """
        # 调用 Embeddings API
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimension,
        )

        # 返回向量
        return response.data[0].embedding

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档

        将多个文本批量转换为向量表示。

        Args:
            texts: 文本列表

        Returns:
            向量列表

        Example:
            texts = ["文档1", "文档2", "文档3"]
            vectors = await embedding.embed_documents(texts)
            print(len(vectors))  # 输出: 3
        """
        # 调用 Embeddings API（支持批量）
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimension,
        )

        # 返回向量列表
        return [item.embedding for item in response.data]

    def get_dimension(self) -> int:
        """
        获取向量维度

        根据模型名称返回对应的向量维度。

        Returns:
            向量维度数

        Note:
            如果配置了 dimension 参数，则返回配置值。
            否则根据模型名称返回默认维度。
        """
        # 如果配置了维度，直接返回
        if self.dimension:
            return self.dimension

        # 根据模型名称返回默认维度
        dimension_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        return dimension_map.get(self.model, 1536)
