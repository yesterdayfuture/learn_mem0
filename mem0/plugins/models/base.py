"""
模型插件基础接口模块

本模块定义大模型和嵌入模型插件的通用接口，所有模型适配器都需要实现这些接口。

支持的模型提供商：
- OpenAI（已实现）
- Azure OpenAI（待实现）
- Anthropic Claude（待实现）
- Google Gemini（待实现）
- 本地模型（待实现）

模型接口的核心功能：
1. 对话生成 - 基于消息列表生成回复
2. 文本总结 - 将长文本压缩为简洁摘要
3. 实体提取 - 从文本中提取命名实体
4. 关系提取 - 从文本中提取实体间关系
5. 向量嵌入 - 将文本转换为向量表示

使用示例:
    # 使用 OpenAI 模型
    from mem0.plugins.models.openai_adapter import OpenAIModel, OpenAIEmbedding

    # 初始化大模型
    llm = OpenAIModel({
        "api_key": "sk-xxx",
        "model": "gpt-4o-mini"
    })

    # 对话
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="你是助手"),
        ChatMessage(role=MessageRole.USER, content="你好")
    ]
    response = await llm.chat(messages)

    # 初始化嵌入模型
    embedding = OpenAIEmbedding({
        "api_key": "sk-xxx",
        "model": "text-embedding-3-small"
    })

    # 生成向量
    vector = await embedding.embed_query("hello world")
"""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ..base import PluginInterface


class MessageRole(Enum):
    """
    消息角色枚举

    定义对话中不同角色的类型：
    - SYSTEM: 系统消息，设定助手行为
    - USER: 用户消息
    - ASSISTANT: 助手回复
    """

    SYSTEM = "system"  # 系统消息
    USER = "user"  # 用户消息
    ASSISTANT = "assistant"  # 助手消息


@dataclass
class ChatMessage:
    """
    聊天消息数据类

    表示对话中的一条消息。

    Attributes:
        role: 消息角色（system/user/assistant）
        content: 消息内容
        name: 发送者名称（可选，用于区分多个用户）
    """

    role: MessageRole  # 消息角色
    content: str  # 消息内容
    name: Optional[str] = None  # 发送者名称（可选）

    def to_dict(self) -> Dict[str, str]:
        """
        转换为字典格式（用于 API 调用）

        Returns:
            符合 OpenAI 格式的字典
        """
        result = {"role": self.role.value, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class ChatResponse:
    """
    聊天响应数据类

    表示模型生成的回复。

    Attributes:
        content: 回复内容
        role: 回复角色（通常是 assistant）
        model: 使用的模型名称
        usage: Token 使用统计
        raw_response: 原始响应数据
    """

    content: str  # 回复内容
    role: MessageRole = MessageRole.ASSISTANT  # 角色
    model: Optional[str] = None  # 模型名称
    usage: Optional[Dict[str, int]] = None  # Token 使用统计
    raw_response: Optional[Any] = None  # 原始响应


class ModelInterface(PluginInterface):
    """
    大模型接口 - 所有大模型插件的基类

    定义大模型的标准操作：
    - chat: 对话生成
    - summarize: 文本总结
    - extract_entities: 实体提取
    - extract_relations: 关系提取

    所有大模型适配器都必须实现这些方法。

    Attributes:
        config: 配置字典，包含 API 密钥、模型名称等
    """

    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResponse:
        """
        对话生成

        基于消息列表生成模型回复。

        Args:
            messages: 消息列表，包含对话历史
            temperature: 温度参数，控制随机性（0-2，默认 0.7）
            max_tokens: 最大生成 token 数（可选）
            **kwargs: 额外的模型参数

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
        pass

    @abstractmethod
    async def summarize(
        self,
        text: str,
        instruction: Optional[str] = None,
        max_length: int = 200,
    ) -> str:
        """
        文本总结

        将长文本压缩为简洁的摘要。

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
        pass

    @abstractmethod
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        实体提取

        从文本中提取命名实体（人名、地名、组织等）。

        Args:
            text: 输入文本

        Returns:
            实体列表，每个实体包含 name、type、description 等字段

        Example:
            entities = await llm.extract_entities("张三在北京工作")
            # 返回: [{"name": "张三", "type": "Person"}, {"name": "北京", "type": "Location"}]
        """
        pass

    @abstractmethod
    async def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        关系提取

        从文本中提取实体之间的关系。

        Args:
            text: 输入文本
            entities: 已提取的实体列表

        Returns:
            关系列表，每个关系包含 source、target、type、description 等字段

        Example:
            entities = [{"name": "张三"}, {"name": "李四"}]
            relations = await llm.extract_relations("张三和李四是朋友", entities)
            # 返回: [{"source": "张三", "target": "李四", "type": "FRIEND"}]
        """
        pass


class EmbeddingInterface(PluginInterface):
    """
    嵌入模型接口 - 所有嵌入模型插件的基类

    定义嵌入模型的标准操作：
    - embed_query: 将查询文本转换为向量
    - embed_documents: 将文档列表批量转换为向量

    所有嵌入模型适配器都必须实现这些方法。

    Attributes:
        config: 配置字典，包含 API 密钥、模型名称等
    """

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本

        将单个查询文本转换为向量表示。

        Args:
            text: 输入文本

        Returns:
            向量（浮点数列表）

        Example:
            vector = await embedding.embed_query("hello world")
            print(len(vector))  # 输出向量维度，如 1536
        """
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档

        将多个文档文本批量转换为向量表示。

        Args:
            texts: 文本列表

        Returns:
            向量列表，每个文本对应一个向量

        Example:
            texts = ["文档1", "文档2", "文档3"]
            vectors = await embedding.embed_documents(texts)
            print(len(vectors))  # 输出: 3
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        获取向量维度

        Returns:
            向量的维度数

        Example:
            dim = embedding.get_dimension()
            print(dim)  # 输出: 1536
        """
        pass
