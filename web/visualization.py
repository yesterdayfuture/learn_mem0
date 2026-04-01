"""
Mem0 数据可视化 Web 界面
使用 FastAPI + 原生 HTML/JS 提供向量库和图数据库的可视化查看
支持动态加载不同的数据库和模型厂商
"""

import json
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from mem0.config import Mem0Config, registry
from mem0.plugins.graph_databases.base import GraphDatabaseInterface
from mem0.plugins.vector_stores.base import VectorStoreInterface
from dotenv import load_dotenv
# 加载项目启动路径下的.env 文件
load_dotenv()


class VisualizationServer:
    """可视化服务器 - 支持动态加载插件"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.app = FastAPI(title="Mem0 Visualization", version="2.0.0")
        self._config: Optional[Mem0Config] = None
        self._raw_config = config

        # 组件
        self.vector_store: Optional[VectorStoreInterface] = None
        self.graph_db: Optional[GraphDatabaseInterface] = None

        # 插件信息
        self._vector_provider: Optional[str] = None
        self._graph_provider: Optional[str] = None

        self._setup_routes()
        self._setup_cors()

    def _setup_cors(self):
        """设置 CORS"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def _initialize_components(self):
        """初始化组件"""
        if self._config is not None:
            return

        # 创建配置
        if self._raw_config:
            self._config = Mem0Config(self._raw_config)
        else:
            self._config = Mem0Config.from_env()

        # 初始化向量存储
        vector_config = self._config.get_vector_store_config()
        self._vector_provider = vector_config.get("provider", "chromadb")
        vector_store_class = registry.get_vector_store(self._vector_provider)

        if vector_store_class:
            self.vector_store = vector_store_class()
            # await self.vector_store.initialize(vector_config.get("config", {}))
            await self.vector_store.initialize({
                "collection_name": os.getenv("CHROMA_COLLECTION"),
                "persist_directory": os.getenv("CHROMA_PERSIST_DIR")
            })



        # 初始化图数据库
        graph_config = self._config.get_graph_db_config()
        self._graph_provider = graph_config.get("provider", "nebula")
        graph_db_class = registry.get_graph_database(self._graph_provider)

        if graph_db_class:
            try:
                self.graph_db = graph_db_class()
                # await self.graph_db.initialize(graph_config.get("config", {}))
                await self.graph_db.initialize({
                        "host": os.getenv("NEBULA_HOST"),
                        "port": os.getenv("NEBULA_PORT"),
                        "username": os.getenv("NEBULA_USER"),
                        "password": os.getenv("NEBULA_PASSWORD"),
                        "space_name": os.getenv("NEBULA_SPACE")  # 使用与主程序相同的 space 名称
                    })
            except Exception as e:
                print(f"图数据库初始化失败（可选）: {e}")
                self.graph_db = None

    def _setup_routes(self):
        """设置路由"""

        @self.app.on_event("startup")
        async def startup():
            """启动时初始化"""
            await self._initialize_components()

        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            """主页面"""
            return self._get_html_template()

        @self.app.get("/api/health")
        async def health():
            """健康检查"""
            await self._initialize_components()
            return {
                "vector_store": {
                    "connected": await self.vector_store.health_check() if self.vector_store else False,
                    "provider": self._vector_provider,
                },
                "graph_db": {
                    "connected": await self.graph_db.health_check() if self.graph_db else False,
                    "provider": self._graph_provider,
                },
            }

        @self.app.get("/api/plugins")
        async def list_plugins():
            """列出所有可用插件"""
            return {
                "vector_stores": registry.list_vector_stores(),
                "graph_databases": registry.list_graph_databases(),
                "embeddings": registry.list_embeddings(),
                "llms": registry.list_llms(),
            }

        @self.app.post("/api/connect/vector")
        async def connect_vector_store(config: Dict[str, Any]):
            """连接向量存储 - 支持动态切换"""
            provider = config.get("provider", "chromadb")
            plugin_config = config.get("config", {})

            plugin_class = registry.get_vector_store(provider)
            if not plugin_class:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown vector store provider: {provider}. Available: {registry.list_vector_stores()}"
                )

            try:
                # 关闭现有连接
                if self.vector_store:
                    await self.vector_store.close()

                # 创建新连接
                self.vector_store = plugin_class()
                await self.vector_store.initialize(plugin_config)
                self._vector_provider = provider

                return {
                    "success": True,
                    "message": f"Connected to {provider}",
                    "provider": provider,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/connect/graph")
        async def connect_graph_db(config: Dict[str, Any]):
            """连接图数据库 - 支持动态切换"""
            provider = config.get("provider", "nebula")
            plugin_config = config.get("config", {})

            plugin_class = registry.get_graph_database(provider)
            if not plugin_class:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown graph database provider: {provider}. Available: {registry.list_graph_databases()}"
                )

            try:
                # 关闭现有连接
                if self.graph_db:
                    await self.graph_db.close()

                # 创建新连接
                self.graph_db = plugin_class()
                await self.graph_db.initialize(plugin_config)
                self._graph_provider = provider

                return {
                    "success": True,
                    "message": f"Connected to {provider}",
                    "provider": provider,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/vector/stats")
        async def vector_stats():
            """获取向量库统计"""
            await self._initialize_components()

            if not self.vector_store or not await self.vector_store.health_check():
                raise HTTPException(status_code=503, detail="Vector store not connected")

            try:
                # 尝试使用 stats 方法获取统计信息
                if hasattr(self.vector_store, 'stats'):
                    stats = await self.vector_store.stats()
                    return {
                        "total_records": stats.get("total_count", 0),
                        "provider": self._vector_provider,
                        "collection_name": stats.get("collection_name", "unknown"),
                    }
                else:
                    # 回退到 count 方法
                    count = await self.vector_store.count()
                    # 获取集合名称
                    collection_name = 'unknown'
                    if hasattr(self.vector_store, '_collection') and self.vector_store._collection:
                        collection_name = getattr(self.vector_store._collection, 'name', 'unknown')
                    elif hasattr(self.vector_store, '_collection_name'):
                        collection_name = self.vector_store._collection_name

                    return {
                        "total_records": count,
                        "provider": self._vector_provider,
                        "collection_name": collection_name,
                    }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/vector/memories")
        async def vector_memories(
            limit: int = Query(50, ge=1, le=200),
            offset: int = Query(0, ge=0),
            search: Optional[str] = None,
        ):
            """获取向量库中的记忆"""
            await self._initialize_components()

            if not self.vector_store or not await self.vector_store.health_check():
                raise HTTPException(status_code=503, detail="Vector store not connected")

            try:
                memories = []

                # 检查是否有 _collection 属性
                if not hasattr(self.vector_store, '_collection') or not self.vector_store._collection:
                    return {
                        "memories": [],
                        "total": 0,
                        "limit": limit,
                        "offset": offset,
                    }

                # 尝试获取所有记录
                collection = self.vector_store._collection
                if hasattr(collection, 'get'):
                    # ChromaDB 的 get 方法不支持 offset，使用 peek 或 get 获取所有数据
                    try:
                        # 尝试使用 peek 获取数据
                        result = collection.peek(limit=limit + offset)
                        # 手动处理 offset
                        if result and result.get("ids"):
                            for key in ["ids", "documents", "metadatas", "embeddings"]:
                                if (key in result) and len(result[key]) > 0:
                                    result[key] = result[key][offset:offset + limit]
                    except Exception as e:
                        print(f"/api/vector/memories -> {e}")
                        # 如果 peek 失败，尝试使用 get 不带 offset
                        result = collection.get(
                            limit=limit + offset,
                            include=["metadatas", "documents", "embeddings"]
                        )
                        # 手动处理 offset
                        if result and result.get("ids"):
                            for key in ["ids", "documents", "metadatas", "embeddings"]:
                                if key in result and result[key]:
                                    result[key] = result[key][offset:offset + limit]

                    if result and result.get("ids"):
                        for i, id_ in enumerate(result["ids"]):
                            # 安全获取 embedding 预览
                            vector_preview = []
                            embeddings = result.get("embeddings")
                            if embeddings is not None and len(embeddings) > i:
                                embedding = result["embeddings"][i]
                                # 处理 NumPy 数组或列表
                                if hasattr(embedding, '__len__') and len(embedding) > 0:
                                    vector_preview = list(embedding)[:5]

                            memory = {
                                "id": id_,
                                "content": result["documents"][i] if result.get("documents") and i < len(result["documents"]) else "",
                                "metadata": result["metadatas"][i] if result.get("metadatas") and i < len(result["metadatas"]) else {},
                                "vector_preview": vector_preview,
                            }

                            # 搜索过滤
                            if search:
                                search_lower = search.lower()
                                content_match = search_lower in memory["content"].lower()
                                metadata_str = json.dumps(memory["metadata"]).lower()
                                metadata_match = search_lower in metadata_str
                                if not (content_match or metadata_match):
                                    continue

                            memories.append(memory)

                return {
                    "memories": memories,
                    "total": len(memories),
                    "limit": limit,
                    "offset": offset,
                }
            except Exception as e:
                import traceback
                error_detail = f"{str(e)}\n{traceback.format_exc()}"
                print(f"Vector memories error: {error_detail}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/vector/memory/{memory_id}")
        async def vector_memory_detail(memory_id: str):
            """获取单个记忆详情"""
            await self._initialize_components()

            if not self.vector_store or not await self.vector_store.health_check():
                raise HTTPException(status_code=503, detail="Vector store not connected")

            try:
                record = await self.vector_store.get(memory_id)
                if not record:
                    raise HTTPException(status_code=404, detail="Memory not found")

                return {
                    "id": record.id,
                    "content": record.text,
                    "metadata": record.metadata,
                    "vector_dimension": len(record.vector) if record.vector else 0,
                    "vector_preview": record.vector[:10] if record.vector else [],
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/vector/memory/{memory_id}")
        async def delete_vector_memory(memory_id: str):
            """删除记忆"""
            await self._initialize_components()

            if not self.vector_store or not await self.vector_store.health_check():
                raise HTTPException(status_code=503, detail="Vector store not connected")

            try:
                await self.vector_store.delete([memory_id])
                return {"success": True, "message": f"Memory {memory_id} deleted"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/graph/stats")
        async def graph_stats():
            """获取图数据库统计"""
            await self._initialize_components()

            if not self.graph_db or not await self.graph_db.health_check():
                raise HTTPException(status_code=503, detail="Graph database not connected")

            try:
                # 获取标签列表和统计信息
                labels = await self.graph_db.list_labels(label_type="node")
                edge_labels = await self.graph_db.list_labels(label_type="edge")

                # 统计节点数量
                node_count = 0
                for label in labels:
                    info = await self.graph_db.get_label_info(label, "node")
                    if info:
                        node_count += info.count

                # 统计边数量
                edge_count = 0
                for label in edge_labels:
                    info = await self.graph_db.get_label_info(label, "edge")
                    if info:
                        edge_count += info.count

                return {
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "provider": self._graph_provider,
                    "space_name": getattr(self.graph_db, '_space_name', 'unknown'),
                    "node_labels": labels,
                    "edge_labels": edge_labels,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/graph/entities")
        async def graph_entities(
            search: Optional[str] = None,
            fuzzy: bool = True,
            limit: int = Query(50, ge=1, le=200),
        ):
            """获取/搜索实体"""
            await self._initialize_components()

            if not self.graph_db or not await self.graph_db.health_check():
                raise HTTPException(status_code=503, detail="Graph database not connected")

            try:
                # 获取所有实体 - 使用 list_labels 和 get_node 组合
                entities = []
                try:
                    # 获取所有标签
                    labels = await self.graph_db.list_labels(label_type="node")
                    for label in labels:
                        # 获取标签信息
                        info = await self.graph_db.get_label_info(label, "node")
                        if info:
                            from nebula3.data.ResultSet import ResultSet
                            # 获取标签下的所有实体
                            nodes = self.graph_db._session.execute(f"MATCH (v:{label}) RETURN v;")
                            nodes_values = []
                            for cur_entity in nodes.column_values("v"):
                                temp = cur_entity.as_node().properties(label)
                                temp["label"] = label
                                nodes_values.append(temp)
                            if nodes_values:
                                entities.extend(nodes_values)
                    # 由于没有索引无法使用 MATCH/SCAN，直接返回空列表
                    # 实体详情可以通过 /api/graph/entity/{entity_name} 获取
                except Exception as e:
                    print(f"Error fetching labels: {e}")

                from nebula3.data.DataObject import ValueWrapper

                results = []
                for entity in entities:
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

                    results.append(cur_result)

                entities = results

                # 搜索过滤
                if search:
                    search_lower = search.lower()
                    filtered_entities = []
                    for entity in entities:
                        name = entity.get("name", "")
                        if search_lower in name.lower():
                            filtered_entities.append(entity)
                    entities = filtered_entities



                return {
                    "entities": entities[:limit],
                    "total": len(entities),
                    "search": search,
                    "fuzzy": fuzzy,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/graph/entity/{entity_name}")
        async def graph_entity_detail(entity_name: str):
            """获取实体详情"""
            await self._initialize_components()

            if not self.graph_db or not await self.graph_db.health_check():
                raise HTTPException(status_code=503, detail="Graph database not connected")

            try:
                node = await self.graph_db.get_node(entity_name)
                if not node:
                    raise HTTPException(status_code=404, detail="Entity not found")

                # 获取关系
                relationships = await self.graph_db.get_relationships(entity_name, "both")

                return {
                    "entity": {
                        "id": node.id,
                        "labels": node.labels,
                        "properties": node.properties,
                    },
                    "relationships": [
                        {
                            "id": r.id,
                            "source": r.source_id,
                            "target": r.target_id,
                            "type": r.rel_type,
                            "properties": r.properties,
                        }
                        for r in relationships
                    ],
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/graph/relationships")
        async def graph_relationships(
            entity: Optional[str] = None,
            limit: int = Query(100, ge=1, le=500),
        ):
            """获取关系"""
            await self._initialize_components()

            if not self.graph_db or not await self.graph_db.health_check():
                raise HTTPException(status_code=503, detail="Graph database not connected")

            try:
                if entity:
                    rels = await self.graph_db.get_relationships(entity, "both")
                    relationships = [
                        {
                            "id": r.id,
                            "source": r.source_id,
                            "target": r.target_id,
                            "type": r.rel_type,
                            "properties": r.properties,
                        }
                        for r in rels[:limit]
                    ]
                else:
                    # 获取所有边 - 使用 list_labels 获取边类型，然后逐个获取
                    edge_labels = await self.graph_db.list_labels(label_type="edge")
                    relationships = []
                    for edge_label in edge_labels:
                        try:
                            edge_rels = await self.graph_db.get_relationships(
                                entity_id="",  # 空 entity_id 获取所有该类型的边
                                rel_type=edge_label,
                            )
                            for r in edge_rels:
                                relationships.append({
                                    "id": r.id,
                                    "source": r.source_id,
                                    "target": r.target_id,
                                    "type": r.rel_type,
                                    "properties": r.properties,
                                })
                                if len(relationships) >= limit:
                                    break
                        except Exception:
                            pass  # 忽略获取边类型的错误

                return {
                    "relationships": relationships[:limit],
                    "total": len(relationships),
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def _parse_vertex_to_dict(self, vertex) -> Dict[str, Any]:
        """解析顶点为字典"""
        vid = str(vertex.vid) if hasattr(vertex, 'vid') else str(vertex)

        labels = []
        properties = {}

        if hasattr(vertex, 'tags'):
            for tag in vertex.tags:
                tag_name = tag.tag_name.decode('utf-8') if isinstance(tag.tag_name, bytes) else tag.tag_name
                labels.append(tag_name)

                if hasattr(tag, 'properties'):
                    for key, value in tag.properties.items():
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        properties[key_str] = value

        return {
            "id": vid,
            "labels": labels,
            "properties": properties,
        }

    def _parse_edge_to_dict(self, edge) -> Dict[str, Any]:
        """解析边为字典"""
        src = str(edge.src) if hasattr(edge, 'src') else ""
        dst = str(edge.dst) if hasattr(edge, 'dst') else ""
        edge_type = edge.name.decode('utf-8') if hasattr(edge, 'name') and isinstance(edge.name, bytes) else getattr(edge, 'name', 'UNKNOWN')

        properties = {}
        if hasattr(edge, 'properties'):
            for key, value in edge.properties.items():
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                properties[key_str] = value

        return {
            "id": f"{src}->{dst}:{edge_type}",
            "source": src,
            "target": dst,
            "type": edge_type,
            "properties": properties,
        }

    def _get_html_template(self) -> str:
        """获取 HTML 模板"""
        return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mem0 数据可视化</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .header p {
            opacity: 0.9;
            font-size: 14px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .tab {
            padding: 10px 20px;
            border: none;
            background: #f0f0f0;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 14px;
        }

        .tab:hover {
            background: #e0e0e0;
        }

        .tab.active {
            background: #667eea;
            color: white;
        }

        .panel {
            display: none;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        .panel.active {
            display: block;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-card h3 {
            font-size: 32px;
            margin-bottom: 5px;
        }

        .stat-card p {
            opacity: 0.9;
            font-size: 14px;
        }

        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        .search-box input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
        }

        .search-box button {
            padding: 12px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.3s;
        }

        .search-box button:hover {
            background: #5568d3;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }

        .data-table th,
        .data-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }

        .data-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }

        .data-table tr:hover {
            background: #f8f9fa;
        }

        .data-table .content-preview {
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }

        .badge-primary {
            background: #e3f2fd;
            color: #1976d2;
        }

        .badge-success {
            background: #e8f5e9;
            color: #388e3c;
        }

        .badge-warning {
            background: #fff3e0;
            color: #f57c00;
        }

        .action-btn {
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s;
        }

        .action-btn.view {
            background: #e3f2fd;
            color: #1976d2;
        }

        .action-btn.delete {
            background: #ffebee;
            color: #d32f2f;
        }

        .action-btn:hover {
            opacity: 0.8;
        }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }

        .modal.active {
            display: flex;
        }

        .modal-content {
            background: white;
            border-radius: 8px;
            max-width: 800px;
            width: 90%;
            max-height: 80vh;
            overflow: auto;
            padding: 20px;
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }

        .modal-header h2 {
            font-size: 20px;
        }

        .modal-close {
            font-size: 24px;
            cursor: pointer;
            color: #999;
        }

        .modal-close:hover {
            color: #333;
        }

        .json-viewer {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #999;
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }

        .empty-state svg {
            width: 80px;
            height: 80px;
            margin-bottom: 20px;
            opacity: 0.3;
        }

        .connection-form {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .form-group {
            margin-bottom: 15px;
        }

        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #555;
        }

        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }

        .btn-primary {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }

        .btn-primary:hover {
            background: #5568d3;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }

        .status-connected {
            background: #e8f5e9;
            color: #2e7d32;
        }

        .status-disconnected {
            background: #ffebee;
            color: #c62828;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
        }

        .provider-info {
            display: inline-block;
            padding: 4px 10px;
            background: #f0f0f0;
            border-radius: 4px;
            font-size: 12px;
            color: #666;
            margin-left: 10px;
        }

        .plugin-list {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }

        .plugin-category {
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #eee;
        }

        .plugin-category h4 {
            margin-bottom: 10px;
            color: #555;
        }

        .plugin-category ul {
            list-style: none;
            padding: 0;
        }

        .plugin-category li {
            padding: 5px 0;
            color: #666;
            font-size: 14px;
        }

        .plugin-category li:before {
            content: "✓ ";
            color: #4caf50;
            margin-right: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Mem0 数据可视化</h1>
        <p>支持动态切换数据库 - 查看和管理向量库与图数据库中的数据</p>
    </div>

    <div class="container">
        <div class="tabs">
            <button class="tab active" onclick="switchTab('vector')">📦 向量库</button>
            <button class="tab" onclick="switchTab('graph')">🕸️ 图数据库</button>
            <button class="tab" onclick="switchTab('settings')">⚙️ 连接设置</button>
        </div>

        <!-- 向量库面板 -->
        <div id="vector-panel" class="panel active">
            <div class="stats-grid">
                <div class="stat-card">
                    <h3 id="vector-total">-</h3>
                    <p>总记录数</p>
                </div>
                <div class="stat-card">
                    <h3 id="vector-collection">-</h3>
                    <p>集合名称</p>
                </div>
            </div>

            <div style="margin-bottom: 15px;">
                <span class="status-indicator status-disconnected" id="vector-status">
                    <span class="status-dot"></span>
                    <span id="vector-status-text">未连接</span>
                </span>
                <span class="provider-info" id="vector-provider">Provider: -</span>
            </div>

            <div class="search-box">
                <input type="text" id="vector-search" placeholder="搜索记忆内容或元数据...">
                <button onclick="searchVectorMemories()">🔍 搜索</button>
                <button onclick="loadVectorMemories()">🔄 刷新</button>
            </div>

            <div id="vector-content">
                <div class="loading">加载中...</div>
            </div>
        </div>

        <!-- 图数据库面板 -->
        <div id="graph-panel" class="panel">
            <div class="stats-grid">
                <div class="stat-card">
                    <h3 id="graph-nodes">-</h3>
                    <p>实体节点</p>
                </div>
                <div class="stat-card">
                    <h3 id="graph-edges">-</h3>
                    <p>关系边</p>
                </div>
            </div>

            <div style="margin-bottom: 15px;">
                <span class="status-indicator status-disconnected" id="graph-status">
                    <span class="status-dot"></span>
                    <span id="graph-status-text">未连接</span>
                </span>
                <span class="provider-info" id="graph-provider">Provider: -</span>
            </div>

            <div class="search-box">
                <input type="text" id="graph-search" placeholder="搜索实体名称（支持模糊查询）...">
                <button onclick="searchGraphEntities()">🔍 搜索</button>
                <button onclick="loadGraphData()">🔄 刷新</button>
            </div>

            <div id="graph-content">
                <div class="loading">加载中...</div>
            </div>
        </div>

        <!-- 连接设置面板 -->
        <div id="settings-panel" class="panel">
            <h2 style="margin-bottom: 20px;">⚙️ 数据库连接设置</h2>

            <div class="connection-form">
                <h3 style="margin-bottom: 15px;">📦 向量存储</h3>
                <div class="form-group">
                    <label>提供商</label>
                    <select id="vector-provider-select">
                        <option value="chromadb">ChromaDB</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>集合名称</label>
                    <input type="text" id="vector-collection-input" value="mem0_memories">
                </div>
                <div class="form-group">
                    <label>持久化目录</label>
                    <input type="text" id="vector-persist-dir" value="./data/chroma">
                </div>
                <button class="btn-primary" onclick="connectVectorStore()">连接</button>
            </div>

            <div class="connection-form">
                <h3 style="margin-bottom: 15px;">🕸️ 图数据库</h3>
                <div class="form-group">
                    <label>提供商</label>
                    <select id="graph-provider-select">
                        <option value="nebula">NebulaGraph</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>主机地址</label>
                    <input type="text" id="graph-host" value="127.0.0.1">
                </div>
                <div class="form-group">
                    <label>端口</label>
                    <input type="number" id="graph-port" value="9669">
                </div>
                <div class="form-group">
                    <label>用户名</label>
                    <input type="text" id="graph-username" value="root">
                </div>
                <div class="form-group">
                    <label>密码</label>
                    <input type="password" id="graph-password" value="nebula">
                </div>
                <div class="form-group">
                    <label>图空间</label>
                    <input type="text" id="graph-space" value="mem0">
                </div>
                <button class="btn-primary" onclick="connectGraphDB()">连接</button>
            </div>

            <div class="connection-form">
                <h3 style="margin-bottom: 15px;">📋 可用插件</h3>
                <div id="plugins-list">
                    <div class="loading">加载中...</div>
                </div>
            </div>
        </div>
    </div>

    <!-- 详情模态框 -->
    <div id="detail-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>详情</h2>
                <span class="modal-close" onclick="closeModal()">&times;</span>
            </div>
            <div id="modal-body"></div>
        </div>
    </div>

    <script>
        // 切换标签页
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));

            event.target.classList.add('active');
            document.getElementById(tab + '-panel').classList.add('active');

            if (tab === 'vector') {
                loadVectorStats();
                loadVectorMemories();
            } else if (tab === 'graph') {
                loadGraphData();
                loadGraphStats();
            } else if (tab === 'settings') {
                loadPluginsList();
            }
        }

        // 关闭模态框
        function closeModal() {
            document.getElementById('detail-modal').classList.remove('active');
        }

        // 更新连接状态显示
        function updateConnectionStatus(type, connected, provider) {
            const statusEl = document.getElementById(type + '-status');
            const statusTextEl = document.getElementById(type + '-status-text');
            const providerEl = document.getElementById(type + '-provider');

            if (connected) {
                statusEl.classList.remove('status-disconnected');
                statusEl.classList.add('status-connected');
                statusTextEl.textContent = '已连接';
            } else {
                statusEl.classList.remove('status-connected');
                statusEl.classList.add('status-disconnected');
                statusTextEl.textContent = '未连接';
            }

            if (provider) {
                providerEl.textContent = 'Provider: ' + provider;
            }
        }

        // 加载向量库统计
        async function loadVectorStats() {
            try {
                const response = await fetch('/api/vector/stats');
                const data = await response.json();
                document.getElementById('vector-total').textContent = data.total_records;
                document.getElementById('vector-collection').textContent = data.collection_name;
                updateConnectionStatus('vector', true, data.provider);
            } catch (e) {
                document.getElementById('vector-total').textContent = '-';
                document.getElementById('vector-collection').textContent = '-';
                updateConnectionStatus('vector', false, null);
            }
        }

        // 加载向量库记忆
        async function loadVectorMemories() {
            const content = document.getElementById('vector-content');
            content.innerHTML = '<div class="loading">加载中...</div>';

            try {
                const response = await fetch('/api/vector/memories?limit=100');
                const data = await response.json();
                renderVectorMemories(data.memories);
            } catch (e) {
                content.innerHTML = `<div class="empty-state">
                    <p>加载失败: ${e.message}</p>
                    <p>请确保向量库已连接</p>
                </div>`;
            }
        }

        // 渲染向量记忆列表
        function renderVectorMemories(memories) {
            const content = document.getElementById('vector-content');

            if (!memories || memories.length === 0) {
                content.innerHTML = `<div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M20 13V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7m16 0v5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-5m16 0h-2.586a1 1 0 0 0-.707.293l-2.414 2.414a1 1 0 0 1-.707.293h-3.172a1 1 0 0 1-.707-.293l-2.414-2.414A1 1 0 0 0 6.586 13H4"/>
                    </svg>
                    <p>暂无记忆数据</p>
                </div>`;
                return;
            }

            let html = `
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>内容</th>
                            <th>类型</th>
                            <th>用户</th>
                            <th>重要性</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            memories.forEach(memory => {
                const metadata = memory.metadata || {};
                const memoryType = metadata.memory_type || 'unknown';
                const userId = metadata.user_id || 'default';
                const importance = metadata.importance || 0.5;

                html += `
                    <tr>
                        <td>${memory.id.substring(0, 8)}...</td>
                        <td class="content-preview" title="${escapeHtml(memory.content)}">${escapeHtml(memory.content)}</td>
                        <td><span class="badge badge-primary">${memoryType}</span></td>
                        <td>${userId}</td>
                        <td>${(importance * 100).toFixed(0)}%</td>
                        <td>
                            <button class="action-btn view" onclick='viewVectorDetail(${JSON.stringify(memory)})'>查看</button>
                            <button class="action-btn delete" onclick="deleteVectorMemory('${memory.id}')">删除</button>
                        </td>
                    </tr>
                `;
            });

            html += '</tbody></table>';
            content.innerHTML = html;
        }

        // 搜索向量记忆
        async function searchVectorMemories() {
            const search = document.getElementById('vector-search').value;
            const content = document.getElementById('vector-content');
            content.innerHTML = '<div class="loading">搜索中...</div>';

            try {
                const response = await fetch(`/api/vector/memories?search=${encodeURIComponent(search)}&limit=100`);
                const data = await response.json();
                renderVectorMemories(data.memories);
            } catch (e) {
                content.innerHTML = `<div class="empty-state"><p>搜索失败: ${e.message}</p></div>`;
            }
        }

        // 查看向量详情
        function viewVectorDetail(memory) {
            const modal = document.getElementById('detail-modal');
            const body = document.getElementById('modal-body');

            body.innerHTML = `
                <div class="json-viewer">${JSON.stringify(memory, null, 2)}</div>
            `;

            modal.classList.add('active');
        }

        // 删除向量记忆
        async function deleteVectorMemory(id) {
            if (!confirm('确定要删除这条记忆吗？')) return;

            try {
                const response = await fetch(`/api/vector/memory/${id}`, { method: 'DELETE' });
                if (response.ok) {
                    loadVectorMemories();
                    loadVectorStats();
                } else {
                    alert('删除失败');
                }
            } catch (e) {
                alert('删除失败: ' + e.message);
            }
        }

        // 加载图数据库统计
        async function loadGraphStats() {
            try {
                const response = await fetch('/api/graph/stats');
                const data = await response.json();
                document.getElementById('graph-nodes').textContent = data.node_count;
                document.getElementById('graph-edges').textContent = data.edge_count;
                updateConnectionStatus('graph', true, data.provider);
            } catch (e) {
                document.getElementById('graph-nodes').textContent = '-';
                document.getElementById('graph-edges').textContent = '-';
                updateConnectionStatus('graph', false, null);
            }
        }

        // 加载图数据库数据
        async function loadGraphData() {
            const content = document.getElementById('graph-content');
            content.innerHTML = '<div class="loading">加载中...</div>';

            try {
                const response = await fetch('/api/graph/entities?limit=100');
                const data = await response.json();
                renderGraphEntities(data.entities);
            } catch (e) {
                content.innerHTML = `<div class="empty-state">
                    <p>加载失败: ${e.message}</p>
                    <p>请确保图数据库已连接</p>
                </div>`;
            }
        }

        // 渲染图实体列表
        function renderGraphEntities(entities) {
            const content = document.getElementById('graph-content');

            if (!entities || entities.length === 0) {
                content.innerHTML = `<div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="16" x2="12" y2="12"/>
                        <line x1="12" y1="8" x2="12.01" y2="8"/>
                    </svg>
                    <p>暂无实体数据</p>
                </div>`;
                return;
            }

            let html = `
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>标签</th>
                            <th>属性</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            entities.forEach(entity => {
                const label = entity.label || 'N/A';
                // 显示名称，若没有 name 字段则显示 memory_id 或 'N/A'
                const name = entity.name || entity.memory_id || 'N/A';
                // 显示简要描述（截取前50字）
                const preview = JSON.stringify(entity) ? JSON.stringify(entity).substring(0, 50) : '无描述';

                html += `
                    <tr>
                        <td>${escapeHtml(name)}</td>
                        <td><span class="badge badge-success">${label}</span></td>
                        <td class="content-preview" title="${escapeHtml(preview)}">${escapeHtml(preview)}</td>
                        <td>
                            <button class="action-btn view" onclick='viewGraphEntity(${JSON.stringify(entity)})'>查看</button>
                        </td>
                    </tr>
                `;
            });

            html += '</tbody></table>';
            content.innerHTML = html;
        }

        // 搜索图实体
        async function searchGraphEntities() {
            const search = document.getElementById('graph-search').value;
            if (!search) {
                loadGraphData();
                return;
            }

            const content = document.getElementById('graph-content');
            content.innerHTML = '<div class="loading">搜索中...</div>';

            try {
                const response = await fetch(`/api/graph/entities?search=${encodeURIComponent(search)}&fuzzy=true&limit=50`);
                const data = await response.json();
                renderGraphEntities(data.entities);
            } catch (e) {
                content.innerHTML = `<div class="empty-state"><p>搜索失败: ${e.message}</p></div>`;
            }
        }

        // 查看图实体详情
        async function viewGraphEntity(entity) {
            const modal = document.getElementById('detail-modal');
            const body = document.getElementById('modal-body');

            try {
                const response = await fetch(`/api/graph/entity/${encodeURIComponent(entity.name)}`);
                const data = await response.json();

                let relationshipsHtml = '';
                if (data.relationships && data.relationships.length > 0) {
                    relationshipsHtml = `
                        <h3 style="margin: 20px 0 10px;">关系 (${data.relationships.length})</h3>
                        <table class="data-table">
                            <thead>
                                <tr><th>源</th><th>关系</th><th>目标</th></tr>
                            </thead>
                            <tbody>
                                ${data.relationships.map(r => `
                                    <tr>
                                        <td>${escapeHtml(r.source)}</td>
                                        <td><span class="badge badge-warning">${escapeHtml(r.type)}</span></td>
                                        <td>${escapeHtml(r.target)}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    `;
                }

                body.innerHTML = `
                    <h3>实体信息</h3>
                    <div class="json-viewer">${JSON.stringify(data.entity, null, 2)}</div>
                    ${relationshipsHtml}
                `;
            } catch (e) {
                body.innerHTML = `
                    <div class="json-viewer">${JSON.stringify(entity, null, 2)}</div>
                    <p style="color: red; margin-top: 10px;">获取关系失败: ${e.message}</p>
                `;
            }

            modal.classList.add('active');
        }

        // 连接向量存储
        async function connectVectorStore() {
            const provider = document.getElementById('vector-provider-select').value;
            const collection = document.getElementById('vector-collection-input').value;
            const persistDir = document.getElementById('vector-persist-dir').value;

            try {
                const response = await fetch('/api/connect/vector', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        provider: provider,
                        config: {
                            collection_name: collection,
                            persist_directory: persistDir,
                        }
                    })
                });

                const data = await response.json();
                if (response.ok) {
                    alert('连接成功: ' + data.message);
                    loadVectorStats();
                } else {
                    alert('连接失败: ' + data.detail);
                }
            } catch (e) {
                alert('连接失败: ' + e.message);
            }
        }

        // 连接图数据库
        async function connectGraphDB() {
            const provider = document.getElementById('graph-provider-select').value;
            const host = document.getElementById('graph-host').value;
            const port = parseInt(document.getElementById('graph-port').value);
            const username = document.getElementById('graph-username').value;
            const password = document.getElementById('graph-password').value;
            const space = document.getElementById('graph-space').value;

            try {
                const response = await fetch('/api/connect/graph', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        provider: provider,
                        config: {
                            host: host,
                            port: port,
                            username: username,
                            password: password,
                            space_name: space,
                        }
                    })
                });

                const data = await response.json();
                if (response.ok) {
                    alert('连接成功: ' + data.message);
                    loadGraphStats();
                } else {
                    alert('连接失败: ' + data.detail);
                }
            } catch (e) {
                alert('连接失败: ' + e.message);
            }
        }

        // 加载插件列表
        async function loadPluginsList() {
            const container = document.getElementById('plugins-list');

            try {
                const response = await fetch('/api/plugins');
                const data = await response.json();

                let html = '<div class="plugin-list">';

                html += '<div class="plugin-category"><h4>📦 向量存储</h4><ul>';
                data.vector_stores.forEach(p => html += `<li>${p}</li>`);
                html += '</ul></div>';

                html += '<div class="plugin-category"><h4>🕸️ 图数据库</h4><ul>';
                data.graph_databases.forEach(p => html += `<li>${p}</li>`);
                html += '</ul></div>';

                html += '<div class="plugin-category"><h4>🔤 嵌入模型</h4><ul>';
                data.embeddings.forEach(p => html += `<li>${p}</li>`);
                html += '</ul></div>';

                html += '<div class="plugin-category"><h4>🤖 大模型</h4><ul>';
                data.llms.forEach(p => html += `<li>${p}</li>`);
                html += '</ul></div>';

                html += '</div>';

                container.innerHTML = html;
            } catch (e) {
                container.innerHTML = `<p style="color: red;">加载失败: ${e.message}</p>`;
            }
        }

        // HTML 转义
        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // 页面加载时初始化
        document.addEventListener('DOMContentLoaded', () => {
            loadVectorStats();
            loadVectorMemories();
        });
    </script>
</body>
</html>'''


def create_visualization_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """创建可视化应用"""
    server = VisualizationServer(config)
    return server.app


# 导出应用
app = create_visualization_app()
