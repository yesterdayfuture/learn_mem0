"""
图数据库插件基础接口模块

本模块定义图数据库插件的通用接口，所有图数据库适配器都需要实现这个接口。

支持的图数据库：
- NebulaGraph（已实现）
- Neo4j（待实现）
- JanusGraph（待实现）
- TigerGraph（待实现）

图数据库的核心功能：
1. 节点管理 - 创建、查询、更新、删除节点
2. 关系管理 - 创建、查询、删除关系（边）
3. 标签管理 - 管理节点标签和边类型
4. 图遍历 - 基于关系的查询
5. 属性搜索 - 基于节点/边属性的查询

使用示例:
    # 使用 NebulaGraph
    from mem0.plugins.graph_databases.nebula import NebulaGraphStore

    graph = NebulaGraphStore({
        "host": "127.0.0.1",
        "port": 9669,
        "username": "root",
        "password": "nebula",
        "space_name": "mem0"
    })

    # 创建节点
    node = await graph.create_node(
        id="张三",
        labels=["Person"],
        properties={"name": "张三", "age": 25}
    )

    # 创建关系
    await graph.create_relationship(
        source_id="张三",
        target_id="李四",
        rel_type="FRIEND",
        properties={"since": "2020"}
    )
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import PluginInterface


@dataclass
class GraphNode:
    """
    图节点数据类

    表示图数据库中的一个节点（顶点）。

    Attributes:
        id: 节点唯一标识符
        labels: 节点标签列表（类型标签）
        properties: 节点属性字典
    """

    id: str  # 节点ID
    labels: List[str] = field(default_factory=list)  # 标签列表
    properties: Dict[str, Any] = field(default_factory=dict)  # 属性字典


@dataclass
class GraphEdge:
    """
    图边数据类

    表示图数据库中的一条边（关系）。

    Attributes:
        id: 边唯一标识符
        source_id: 源节点ID
        target_id: 目标节点ID
        rel_type: 关系类型
        properties: 边属性字典
    """

    id: str  # 边ID
    source_id: str  # 源节点ID
    target_id: str  # 目标节点ID
    rel_type: str  # 关系类型
    properties: Dict[str, Any] = field(default_factory=dict)  # 属性字典


@dataclass
class LabelInfo:
    """
    标签信息数据类

    表示节点标签或边类型的详细信息。

    Attributes:
        name: 标签名称
        type: 类型（"node" 或 "edge"）
        count: 实体数量
        properties: 属性定义列表
    """

    name: str  # 标签名称
    type: str  # 类型："node" 或 "edge"
    count: int = 0  # 实体数量
    properties: List[Dict[str, Any]] = field(default_factory=list)  # 属性定义


class GraphDatabaseInterface(PluginInterface):
    """
    图数据库接口 - 所有图数据库插件的基类

    定义图数据库的标准操作：
    - 节点操作：创建、获取、更新、删除、搜索
    - 关系操作：创建、获取、删除
    - 标签管理：列出、获取详情

    所有图数据库适配器都必须实现这些方法。

    Attributes:
        config: 配置字典，包含连接参数
    """

    # ==================== 节点操作 ====================

    @abstractmethod
    async def create_node(
        self,
        id: str,
        labels: List[str],
        properties: Dict[str, Any],
    ) -> GraphNode:
        """
        创建节点

        在图中创建一个新节点，指定ID、标签和属性。

        Args:
            id: 节点唯一标识符
            labels: 节点标签列表（如 ["Person", "User"]）
            properties: 节点属性字典

        Returns:
            创建的节点对象

        Raises:
            ValueError: 如果节点ID已存在

        Example:
            node = await graph.create_node(
                id="user_001",
                labels=["Person"],
                properties={"name": "张三", "age": 25}
            )
        """
        pass

    @abstractmethod
    async def get_node(self, id: str) -> Optional[GraphNode]:
        """
        获取节点

        根据ID获取节点信息。

        Args:
            id: 节点ID

        Returns:
            节点对象，如果不存在则返回 None

        Example:
            node = await graph.get_node("user_001")
            if node:
                print(f"Name: {node.properties.get('name')}")
        """
        pass

    @abstractmethod
    async def update_node(
        self,
        id: str,
        properties: Dict[str, Any],
    ) -> Optional[GraphNode]:
        """
        更新节点

        更新节点的属性。

        Args:
            id: 节点ID
            properties: 要更新的属性字典

        Returns:
            更新后的节点对象，如果不存在则返回 None

        Example:
            await graph.update_node("user_001", {"age": 26, "city": "北京"})
        """
        pass

    @abstractmethod
    async def delete_node(self, id: str) -> bool:
        """
        删除节点

        删除指定ID的节点及其所有关系。

        Args:
            id: 节点ID

        Returns:
            是否删除成功

        Example:
            success = await graph.delete_node("user_001")
        """
        pass

    @abstractmethod
    async def search_nodes(
        self,
        properties: Dict[str, Any],
        labels: Optional[List[str]] = None,
        fuzzy: bool = False,
    ) -> List[GraphNode]:
        """
        搜索节点

        根据属性和标签搜索节点。

        Args:
            properties: 属性过滤条件
            labels: 标签过滤条件（可选）
            fuzzy: 是否使用模糊匹配（默认 False）

        Returns:
            匹配的节点列表

        Example:
            # 精确匹配
            nodes = await graph.search_nodes({"name": "张三"})

            # 模糊匹配
            nodes = await graph.search_nodes({"name": "张"}, fuzzy=True)

            # 带标签过滤
            nodes = await graph.search_nodes(
                {"age": 25},
                labels=["Person"]
            )
        """
        pass

    # ==================== 关系操作 ====================

    @abstractmethod
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> GraphEdge:
        """
        创建关系

        在两个节点之间创建一条边。

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            rel_type: 关系类型（如 "FRIEND", "WORKS_AT"）
            properties: 关系属性（可选）

        Returns:
            创建的边对象

        Raises:
            ValueError: 如果源节点或目标节点不存在

        Example:
            edge = await graph.create_relationship(
                source_id="user_001",
                target_id="company_001",
                rel_type="WORKS_AT",
                properties={"since": "2020", "position": "工程师"}
            )
        """
        pass

    @abstractmethod
    async def get_relationships(
        self,
        node_id: str,
        direction: str = "both",
        rel_type: Optional[str] = None,
    ) -> List[GraphEdge]:
        """
        获取节点的关系

        获取与指定节点相关的所有边。

        Args:
            node_id: 节点ID
            direction: 方向（"in" 入边 / "out" 出边 / "both" 双向，默认 "both"）
            rel_type: 关系类型过滤（可选）

        Returns:
            边对象列表

        Example:
            # 获取所有关系
            edges = await graph.get_relationships("user_001")

            # 只获取出边
            edges = await graph.get_relationships("user_001", direction="out")

            # 只获取特定类型的关系
            edges = await graph.get_relationships("user_001", rel_type="FRIEND")
        """
        pass

    @abstractmethod
    async def delete_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: Optional[str] = None,
    ) -> bool:
        """
        删除关系

        删除两个节点之间的边。

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            rel_type: 关系类型（可选，不指定则删除所有类型）

        Returns:
            是否删除成功

        Example:
            # 删除特定类型的关系
            await graph.delete_relationship("user_001", "user_002", "FRIEND")

            # 删除所有关系
            await graph.delete_relationship("user_001", "user_002")
        """
        pass

    # ==================== 标签管理 ====================

    @abstractmethod
    async def list_labels(self, label_type: str = "node") -> List[str]:
        """
        列出所有标签

        Args:
            label_type: 标签类型（"node" 节点标签 / "edge" 边类型，默认 "node"）

        Returns:
            标签名称列表

        Example:
            # 列出所有节点标签
            labels = await graph.list_labels("node")

            # 列出所有边类型
            edge_types = await graph.list_labels("edge")
        """
        pass

    @abstractmethod
    async def get_label_info(self, name: str, label_type: str = "node") -> Optional[LabelInfo]:
        """
        获取标签详细信息

        获取标签的属性定义和实体数量。

        Args:
            name: 标签名称
            label_type: 标签类型（"node" 或 "edge"，默认 "node"）

        Returns:
            标签信息对象，如果不存在则返回 None

        Example:
            info = await graph.get_label_info("Person", "node")
            print(f"Count: {info.count}")
            print(f"Properties: {info.properties}")
        """
        pass

    @abstractmethod
    async def create_label(
        self,
        name: str,
        label_type: str = "node",
        properties: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        创建标签

        创建新的节点标签或边类型。

        Args:
            name: 标签名称
            label_type: 标签类型（"node" 或 "edge"，默认 "node"）
            properties: 属性定义列表（可选）
                每个属性定义包含：name（属性名）、type（数据类型）、nullable（是否可空）

        Returns:
            是否创建成功

        Example:
            # 创建节点标签
            await graph.create_label(
                name="Person",
                label_type="node",
                properties=[
                    {"name": "name", "type": "string", "nullable": False},
                    {"name": "age", "type": "int", "nullable": True}
                ]
            )

            # 创建边类型
            await graph.create_label(
                name="FRIEND",
                label_type="edge",
                properties=[
                    {"name": "since", "type": "string", "nullable": True}
                ]
            )
        """
        pass
