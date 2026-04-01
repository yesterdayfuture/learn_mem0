"""
NebulaGraph 图数据库适配器模块

本模块实现 NebulaGraph 图数据库的适配器，提供图存储和查询功能。

NebulaGraph 特点：
- 分布式架构，支持大规模数据
- 高性能图遍历
- 类 SQL 的查询语言（nGQL）
- 支持多种存储后端

依赖安装:
    pip install nebula3-python

使用示例:
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
        id="user_001",
        labels=["Person"],
        properties={"name": "张三", "age": 25}
    )

    # 创建关系
    await graph.create_relationship(
        source_id="user_001",
        target_id="user_002",
        rel_type="FRIEND"
    )
"""

import json
import re
from typing import Any, Dict, List, Optional

from .base import GraphDatabaseInterface, GraphEdge, GraphNode, LabelInfo


class NebulaGraphStore(GraphDatabaseInterface):
    """
    NebulaGraph 图数据库适配器

    实现 GraphDatabaseInterface 接口，提供基于 NebulaGraph 的图存储功能。

    Attributes:
        config: 配置字典
        _pool: 连接池
        _session: 会话对象
        _space_name: 图空间名称

    配置参数:
        - host: 服务器地址（默认 "127.0.0.1"）
        - port: 服务器端口（默认 9669）
        - username: 用户名（默认 "root"）
        - password: 密码（默认 "nebula"）
        - space_name: 图空间名称（默认 "mem0"）
    """

    def _get_value_as_string(self, value) -> str:
        """
        从 Value 对象获取字符串值（兼容方法）

        处理不同版本的 nebula3-python API 差异

        Args:
            value: Value 对象

        Returns:
            字符串值
        """
        import re

        # 尝试不同的方法获取字符串值
        try:
            if callable(value.get_string):
                return value.get_string()
        except Exception:
            pass

        try:
            if callable(value.as_string):
                return value.as_string()
        except Exception:
            pass

        try:
            if callable(value.get_value):
                raw_val = value.get_value()
                if isinstance(raw_val, bytes):
                    return raw_val.decode('utf-8')
                return str(raw_val)
        except Exception:
            pass

        # 尝试直接访问 sVal 或 s_val 属性
        str_val = str(value)

        # 从 str(value) 中提取值，如 "Value(sVal=b'Concept')"
        match = re.search(r"sVal\s*=\s*b['\"]([^'\"]+)['\"]", str_val)
        if match:
            return match.group(1)

        match = re.search(r"b['\"]([^'\"]+)['\"]", str_val)
        if match:
            return match.group(1)

        return str_val

    def _get_value_as_int(self, value) -> int:
        """
        从 Value 对象获取整数值（兼容方法）

        处理不同版本的 nebula3-python API 差异

        Args:
            value: Value 对象

        Returns:
            整数值
        """
        try:
            if callable(value.get_int):
                return value.get_int()
        except Exception:
            pass

        try:
            if callable(value.as_int):
                return value.as_int()
        except Exception:
            pass

        try:
            if hasattr(value, 'iVal'):
                return value.iVal
            elif hasattr(value, 'i_val'):
                return value.i_val
            elif hasattr(value, 'value'):
                return value.value
        except Exception:
            pass

        return int(str(value))

    # 从 ValueWrapper 数据类型中获取顶点 ✅
    def _get_vertex(self, value) -> Any:
        """
        从 Value 对象获取顶点（兼容方法）

        Args:
            value: Value 对象

        Returns:
            顶点对象
        """
        if hasattr(value, 'get_vertex'):
            return value.get_vertex()
        elif hasattr(value, 'as_node'):
            return value.as_node()
        else:
            raise AttributeError(f"Value 对象没有获取顶点的方法: {type(value)}")

    # 从 ValueWrapper 数据类型中获取relation ✅
    def _get_edge(self, value) -> Any:
        """
        从 Value 对象获取边（兼容方法）

        Args:
            value: Value 对象

        Returns:
            边对象
        """
        if hasattr(value, 'get_edge'):
            return value.get_edge()
        elif hasattr(value, 'as_relationship'):
            return value.as_relationship()
        else:
            raise AttributeError(f"Value 对象没有获取边的方法: {type(value)}")

    def __init__(self):
        """
        初始化 NebulaGraph 存储

        注意：实际初始化通过 initialize() 方法完成
        """
        super().__init__()
        self._pool = None  # 连接池
        self._session = None  # 会话
        self._space_name = "mem0"  # 图空间名

    # 通过字典配置初始化客户端 ✅
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        异步初始化 NebulaGraph 存储

        Args:
            config: 配置字典，包含连接参数
        """
        self.config = config
        self._space_name = config.get("space_name", "mem0")
        self._init_connection()

    def _init_connection(self) -> None:
        """
        初始化数据库连接（内部方法）

        创建连接池，建立会话，并初始化图空间。
        """
        from nebula3.gclient.net import ConnectionPool
        from nebula3.Config import Config

        nebula_config = Config()

        # 获取连接配置
        host = self.config.get("host", "127.0.0.1")
        port = self.config.get("port", 9669)
        username = self.config.get("username", "root")
        password = self.config.get("password", "nebula")

        # 创建连接池配置
        nebula_config.max_connection_pool_size = 10

        # 初始化连接池
        self._pool = ConnectionPool()
        self._pool.init([(host, port)], nebula_config)

        # 创建会话
        self._session = self._pool.get_session(username, password)

        # 初始化图空间
        self._init_space()

    # 初始化图空间 ✅
    def _init_space(self) -> None:
        """
        初始化图空间（内部方法）

        如果图空间不存在则创建，并切换到该空间。
        使用轮询方式等待图空间就绪。
        """
        import time

        # 创建图空间（如果不存在）
        create_space_sql = f"""
        CREATE SPACE IF NOT EXISTS {self._space_name} (
            vid_type = FIXED_STRING(256),
            partition_num = 1,
            replica_factor = 1
        )
        """
        result = self._session.execute(create_space_sql)
        if not result.is_succeeded():
            error_msg = result.error_msg()
            print(f"创建图空间失败: {error_msg}")
            raise RuntimeError(f"创建图空间失败: {error_msg}")

        # 等待图空间就绪（轮询检查）
        max_retries = 30
        retry_interval = 0.5
        for i in range(max_retries):
            result = self._session.execute(f"USE {self._space_name}")
            if result.is_succeeded():
                break
            time.sleep(retry_interval)
        else:
            raise RuntimeError(f"图空间 {self._space_name} 未能在 {max_retries * retry_interval} 秒内就绪")

        # 初始化标签和边类型
        self._init_schema()

    # 初始化 Schema（节点标签和关系标签） ✅
    def _init_schema(self) -> None:
        """
        初始化 Schema（内部方法）

        创建默认的节点标签和边类型，并检查执行结果。
        """
        # 创建默认标签
        tags = ["Entity", "Person", "Organization", "Location", "Concept"]
        for tag in tags:
            sql = f"""
            CREATE TAG IF NOT EXISTS {tag} (
                name string,
                description string,
                memory_id string,
                created_at timestamp
            )
            """
            result = self._session.execute(sql)
            if not result.is_succeeded():
                error_msg = result.error_msg()
                # 忽略 "已存在" 错误
                if "Existed" not in error_msg:
                    print(f"创建标签 {tag} 失败: {error_msg}")

        # 创建默认边类型
        edge_types = ["RELATED_TO", "FRIEND", "WORKS_AT", "LOCATED_IN", "KNOWS"]
        for edge_type in edge_types:
            sql = f"""
            CREATE EDGE IF NOT EXISTS {edge_type} (
                description string,
                memory_id string,
                created_at timestamp
            )
            """
            result = self._session.execute(sql)
            if not result.is_succeeded():
                error_msg = result.error_msg()
                # 忽略 "已存在" 错误
                if "Existed" not in error_msg:
                    print(f"创建边类型 {edge_type} 失败: {error_msg}")

    # 健康检测 ✅
    async def health_check(self) -> bool:
        """
        健康检查

        检查 NebulaGraph 连接是否正常。

        Returns:
            True 表示健康，False 表示异常
        """
        try:
            result = self._session.execute("SHOW SPACES")
            return result.is_succeeded()
        except Exception:
            return False

    # 关闭连接 ✅
    async def close(self) -> None:
        """
        关闭连接

        释放会话和连接池资源。
        """
        if self._session:
            self._session.release()
        if self._pool:
            self._pool.close()

    def _escape_string(self, value: str) -> str:
        """
        转义字符串（内部方法）

        处理特殊字符，防止 nGQL 注入。

        Args:
            value: 原始字符串

        Returns:
            转义后的字符串
        """
        # 转义单引号和反斜杠
        return value.replace("\\", "\\\\").replace("'", "\\'")

    # 创建节点 ✅
    async def create_node(
        self,
        id: str,
        labels: List[str],
        properties: Dict[str, Any],
    ) -> GraphNode:
        """
        创建节点

        Args:
            id: 节点ID
            labels: 标签列表
            properties: 属性字典

        Returns:
            创建的节点对象
        """
        import time

        # 使用第一个标签作为主标签
        primary_label = labels[0] if labels else "Entity"

        # 构建属性字符串
        props = {
            "name": properties.get("name", ""),
            "description": properties.get("description", ""),
            "memory_id": properties.get("memory_id", ""),
            "created_at": int(time.time()),
        }

        # 构建 INSERT 语句
        # nGQL 语法: INSERT VERTEX tag(prop1, prop2) VALUES "vid":("value1", "value2")
        prop_names = ", ".join(props.keys())
        prop_values = []
        for v in props.values():
            if isinstance(v, int):
                prop_values.append(str(v))
            else:
                prop_values.append(f'"{self._escape_string(str(v))}"')
        prop_values_str = ", ".join(prop_values)

        sql = f'INSERT VERTEX {primary_label}({prop_names}) VALUES "{self._escape_string(id)}":({prop_values_str})'

        result = self._session.execute(sql)

        if not result.is_succeeded():
            raise ValueError(f"Failed to create node: {result.error_msg()}")

        return GraphNode(
            id=id,
            labels=labels,
            properties=properties,
        )


    def get_properties_by_tag(self, vertex, target_tag):
        """新版：从 Vertex 对象中获取指定 tag 的属性字典"""
        tag_props_map = vertex.as_map()  # 返回 {b'Concept': {...}, ...}
        for tag_bytes, props in tag_props_map.items():
            tag = tag_bytes.decode('utf-8')
            if tag == target_tag:
                # props 是一个字典，键是属性名（bytes），值是 ValueWrapper
                result = {}
                for k_bytes, v_wrapper in props.items():
                    key = k_bytes.decode('utf-8')
                    # 根据实际类型转换值
                    if v_wrapper.is_int():
                        value = v_wrapper.as_int()
                    elif v_wrapper.is_string():
                        value = v_wrapper.as_string().decode('utf-8')
                    elif v_wrapper.is_bool():
                        value = v_wrapper.as_bool()
                    else:
                        value = v_wrapper  # 或继续递归
                    result[key] = value
                return result
        return {}

    # 根据 id 获取节点 ✅
    async def get_node(self, id: str) -> Optional[GraphNode]:
        """
        获取节点

        Args:
            id: 节点ID

        Returns:
            节点对象，如果不存在则返回 None
        """
        # 查询节点 - 使用 FETCH PROP ON * 获取所有标签的属性
        sql = f"FETCH PROP ON * \"{self._escape_string(id)}\" YIELD vertex AS v"
        result = self._session.execute(sql)

        if not result.is_succeeded() or result.row_size() == 0:
            return None

        # 解析结果
        row = result.row_values(0)
        vertex = self._get_vertex(row[0])

        # 提取标签
        labels = []
        if hasattr(vertex, 'tags'):
            for tag in vertex.tags():
                labels.append(tag)

        # 提取属性
        properties = {}
        from nebula3.data.DataObject import ValueWrapper
        for label in labels:
            entity = vertex.properties(label)
            cur_result = {}
            for k, v in entity.items():
                v_value = ""
                if isinstance(v, ValueWrapper):
                    v = v.get_value()
                    v_value = v.value
                if isinstance(v, str):
                    v_value = v

                if isinstance(v_value, bytes):
                    v_value = v_value.decode("utf-8")

                cur_result[k] = v_value

            properties[label] = cur_result

        return GraphNode(
            id=id,
            labels=labels,
            properties= properties,
        )


    async def update_node(
        self,
        id: str,
        properties: Dict[str, Any],
    ) -> Optional[GraphNode]:
        """
        更新节点

        Args:
            id: 节点ID
            properties: 要更新的属性

        Returns:
            更新后的节点对象
        """
        # 先获取节点
        node = await self.get_node(id)
        if not node:
            return None

        # 更新属性
        node.properties.update(properties)

        # 使用第一个标签更新
        primary_label = node.labels[0] if node.labels else "Entity"

        # 构建 UPDATE 语句
        set_clauses = []
        for k, v in properties.items():
            set_clauses.append(f"{k}='{self._escape_string(str(v))}'")

        if set_clauses:
            sql = f"UPDATE VERTEX ON {primary_label} '{self._escape_string(id)}' SET {', '.join(set_clauses)}"
            self._session.execute(sql)

        return node

    async def delete_node(self, id: str) -> bool:
        """
        删除节点

        Args:
            id: 节点ID

        Returns:
            是否删除成功
        """
        # 删除节点及其所有边
        sql = f"DELETE VERTEX '{self._escape_string(id)}' WITH EDGE"
        result = self._session.execute(sql)

        return result.is_succeeded()

    async def search_nodes(
        self,
        properties: Dict[str, Any],
        labels: Optional[List[str]] = None,
        fuzzy: bool = False,
    ) -> List[GraphNode]:
        """
        搜索节点

        Args:
            properties: 属性过滤条件
            labels: 标签过滤
            fuzzy: 是否模糊匹配

        Returns:
            匹配的节点列表
        """
        nodes = []

        # 获取所有标签（如果未指定）
        if not labels:
            labels_result = self._session.execute("SHOW TAGS")
            if labels_result.is_succeeded():
                labels = [self._get_value_as_string(row.values[0]) for row in labels_result.rows()]

        # 对每个标签进行搜索
        for label in (labels or ["Entity"]):
            # 构建 WHERE 条件
            conditions = []
            for k, v in properties.items():
                if fuzzy and isinstance(v, str):
                    # 模糊匹配
                    conditions.append(f"{k} =~ '.*{self._escape_string(v)}.*'")
                else:
                    conditions.append(f"{k} == '{self._escape_string(str(v))}'")

            where_clause = " AND ".join(conditions) if conditions else ""

            # 构建查询语句
            if where_clause:
                sql = f"MATCH (v:{label}) WHERE {where_clause} RETURN v LIMIT 100"
            else:
                sql = f"MATCH (v:{label}) RETURN v LIMIT 100"

            result = self._session.execute(sql)

            if result.is_succeeded():
                for row in result.rows():
                    vertex = self._get_vertex(row.values[0])
                    vid = self._get_value_as_string(vertex.vid)

                    # 提取标签
                    node_labels = [tag.tag_name.decode('utf-8') for tag in vertex.tags]

                    # 提取属性
                    node_props = {}
                    for tag in vertex.tags:
                        for prop in tag.props:
                            key = prop.key.decode('utf-8')
                            value = prop.value
                            node_props[key] = value

                    nodes.append(GraphNode(
                        id=vid,
                        labels=node_labels,
                        properties=node_props,
                    ))

        return nodes

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> GraphEdge:
        """
        创建关系

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            rel_type: 关系类型
            properties: 关系属性

        Returns:
            创建的边对象
        """
        import time

        props = properties or {}
        edge_props = {
            "description": props.get("description", ""),
            "memory_id": props.get("memory_id", ""),
            "created_at": int(time.time()),
        }

        # 构建属性字符串
        # nGQL 语法: INSERT EDGE type(prop1, prop2) VALUES "src"->"dst":("val1", "val2")
        prop_names = ", ".join(edge_props.keys())
        prop_values = []
        for v in edge_props.values():
            if isinstance(v, int):
                prop_values.append(str(v))
            else:
                prop_values.append(f'"{self._escape_string(str(v))}"')
        prop_values_str = ", ".join(prop_values)

        sql = f'INSERT EDGE {rel_type}({prop_names}) VALUES "{self._escape_string(source_id)}"->"{self._escape_string(target_id)}":({prop_values_str})'

        result = self._session.execute(sql)

        if not result.is_succeeded():
            raise ValueError(f"Failed to create relationship: {result.error_msg()}")

        return GraphEdge(
            id=f"{source_id}->{target_id}",
            source_id=source_id,
            target_id=target_id,
            rel_type=rel_type,
            properties=props,
        )
    # ✅获取某个节点的关系
    async def get_relationships(
        self,
        node_id: str,
        direction: str = "both",
        rel_type: Optional[str] = None,
    ) -> List[GraphEdge]:
        """
        获取节点的关系

        Args:
            node_id: 节点ID
            direction: 方向（in/out/both）
            rel_type: 关系类型过滤

        Returns:
            边对象列表
        """
        edges = []

        # 构建 MATCH 语句
        if direction == "out":
            if rel_type:
                sql = f"MATCH (v)-[e:{rel_type}]->() WHERE id(v) == '{self._escape_string(node_id)}' RETURN e"
            else:
                sql = f"MATCH (v)-[e]->() WHERE id(v) == '{self._escape_string(node_id)}' RETURN e"
        elif direction == "in":
            if rel_type:
                sql = f"MATCH ()-[e:{rel_type}]->(v) WHERE id(v) == '{self._escape_string(node_id)}' RETURN e"
            else:
                sql = f"MATCH ()-[e]->(v) WHERE id(v) == '{self._escape_string(node_id)}' RETURN e"
        else:  # both
            if rel_type:
                sql = f"MATCH (v)-[e:{rel_type}]-() WHERE id(v) == '{self._escape_string(node_id)}' RETURN e"
            else:
                sql = f"MATCH (v)-[e]-() WHERE id(v) == '{self._escape_string(node_id)}' RETURN e"

        result = self._session.execute(sql)

        if result.is_succeeded():
            for row in result.column_values("e"):
                edge = self._get_edge(row)

                edges.append(GraphEdge(
                    id=f"{edge.start_vertex_id()}->{edge.end_vertex_id()}",
                    source_id=edge.start_vertex_id().as_string(),
                    target_id=edge.end_vertex_id().as_string(),
                    rel_type=edge.edge_name(),
                    properties={},  # 可以进一步解析属性
                ))

        return edges

    async def delete_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: Optional[str] = None,
    ) -> bool:
        """
        删除关系

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            rel_type: 关系类型

        Returns:
            是否删除成功
        """
        if rel_type:
            sql = f"""
            DELETE EDGE {rel_type} '{self._escape_string(source_id)}'->'{self._escape_string(target_id)}'
            """
        else:
            # 删除所有类型的边
            sql = f"""
            DELETE EDGE * '{self._escape_string(source_id)}'->'{self._escape_string(target_id)}'
            """

        result = self._session.execute(sql)
        return result.is_succeeded()

    async def list_labels(self, label_type: str = "node") -> List[str]:
        """
        列出所有标签

        Args:
            label_type: 标签类型（node/edge）

        Returns:
            标签名称列表
        """
        if label_type == "node":
            sql = "SHOW TAGS"
        else:
            sql = "SHOW EDGES"

        result = self._session.execute(sql)
        labels = []

        if result.is_succeeded():
            for row in result.rows():
                labels.append(self._get_value_as_string(row.values[0]))

        return labels

    async def get_label_info(self, name: str, label_type: str = "node") -> Optional[LabelInfo]:
        """
        获取标签详细信息

        Args:
            name: 标签名称
            label_type: 标签类型

        Returns:
            标签信息对象
        """
        # 获取属性定义
        if label_type == "node":
            sql = f"DESCRIBE TAG {name}"
        else:
            sql = f"DESCRIBE EDGE {name}"

        result = self._session.execute(sql)

        properties = []
        if result.is_succeeded():
            for row in result.rows():
                properties.append({
                    "name": self._get_value_as_string(row.values[0]),
                    "type": self._get_value_as_string(row.values[1]),
                    "nullable": self._get_value_as_string(row.values[2]) == "YES",
                })

        # 获取实体数量 - 使用 FETCH 方法（不需要索引）
        if label_type == "node":
            # 使用 FETCH PROP 获取所有该标签的顶点
            fetch_sql = f"FETCH PROP ON {name} *"
            fetch_result = self._session.execute(fetch_sql)
            count = 0
            if fetch_result.is_succeeded():
                count = fetch_result.row_size()
            else:
                # 如果 FETCH 失败，尝试使用 SHOW STATS
                stats_sql = "SHOW STATS"
                stats_result = self._session.execute(stats_sql)
                if stats_result.is_succeeded():
                    # 解析 SHOW STATS 结果
                    for row in stats_result.rows():
                        if len(row.values) >= 2:
                            tag_name = self._get_value_as_string(row.values[1])
                            if tag_name == name:
                                count = self._get_value_as_int(row.values[2])
                                break
                else:
                    print(f"Count query failed for {name}: {fetch_result.error_msg() if hasattr(fetch_result, 'error_msg') else 'Unknown error'}")
        else:
            # 对于边，使用 FETCH PROP 获取所有该类型的边
            fetch_sql = f"FETCH PROP ON {name} *"
            fetch_result = self._session.execute(fetch_sql)
            count = 0
            if fetch_result.is_succeeded():
                count = fetch_result.row_size()
            else:
                # 如果 FETCH 失败，尝试使用 SHOW STATS
                stats_sql = "SHOW STATS"
                stats_result = self._session.execute(stats_sql)
                if stats_result.is_succeeded():
                    # 解析 SHOW STATS 结果
                    for row in stats_result.rows():
                        if len(row.values) >= 2:
                            edge_name = self._get_value_as_string(row.values[1])
                            if edge_name == name:
                                count = self._get_value_as_int(row.values[2])
                                break
                else:
                    print(f"Count query failed for edge {name}: {fetch_result.error_msg() if hasattr(fetch_result, 'error_msg') else 'Unknown error'}")

        return LabelInfo(
            name=name,
            type=label_type,
            count=count,
            properties=properties,
        )

    async def create_label(
        self,
        name: str,
        label_type: str = "node",
        properties: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        创建标签

        Args:
            name: 标签名称
            label_type: 标签类型
            properties: 属性定义列表

        Returns:
            是否创建成功
        """
        props = properties or []

        # 构建属性定义
        prop_defs = []
        for prop in props:
            prop_name = prop.get("name", "")
            prop_type = prop.get("type", "string")
            nullable = prop.get("nullable", True)

            # 类型映射
            type_mapping = {
                "string": "string",
                "int": "int64",
                "float": "double",
                "bool": "bool",
                "timestamp": "timestamp",
            }
            nebula_type = type_mapping.get(prop_type, "string")

            if nullable:
                prop_defs.append(f"{prop_name} {nebula_type} NULL")
            else:
                prop_defs.append(f"{prop_name} {nebula_type}")

        prop_str = f"({', '.join(prop_defs)})" if prop_defs else "()"

        # 构建创建语句
        if label_type == "node":
            sql = f"CREATE TAG IF NOT EXISTS {name}{prop_str}"
        else:
            sql = f"CREATE EDGE IF NOT EXISTS {name}{prop_str}"

        result = self._session.execute(sql)
        return result.is_succeeded()

    async def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        执行原生查询（支持 MATCH 等查询）

        Args:
            query: nGQL 查询语句

        Returns:
            查询结果列表，每行是一个字典
        """
        result = self._session.execute(query)

        if not result.is_succeeded():
            raise ValueError(f"Query failed: {result.error_msg()}")

        # 解析结果
        rows = []
        columns = result.keys()

        for row in result.rows():
            row_dict = {}
            for i, col_name in enumerate(columns):
                value = row.values[i]
                # 使用 _get_value_as_string 处理所有类型
                row_dict[col_name] = self._get_value_as_string(value)
            rows.append(row_dict)

        return rows
