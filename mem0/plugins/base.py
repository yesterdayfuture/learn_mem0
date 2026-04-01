"""
Mem0 插件基础接口模块

本模块定义所有插件的通用基础接口，所有具体插件实现都需要继承这些接口。

插件系统架构：
- PluginInterface: 所有插件的基类接口
- VectorStoreInterface: 向量存储插件接口
- GraphDatabaseInterface: 图数据库插件接口
- EmbeddingInterface: 嵌入模型插件接口
- ModelInterface: 大模型插件接口

插件开发规范：
1. 所有插件必须实现对应的基础接口
2. 插件类名应该清晰表明其功能
3. 插件初始化参数通过 config 字典传递
4. 异步方法必须使用 async def 定义
5. 文档字符串必须包含参数和返回值说明

使用示例:
    # 开发自定义向量存储插件
    class MyVectorStore(VectorStoreInterface):
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.client = MyClient(config["host"], config["port"])

        async def add(self, records: List[VectorRecord]) -> List[str]:
            # 实现添加逻辑
            pass
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class PluginInterface(ABC):
    """
    插件基础接口 - 所有插件的基类

    定义插件的基本生命周期方法：
    - 初始化
    - 健康检查
    - 资源清理

    所有具体插件都应该继承此接口并实现抽象方法。

    Attributes:
        config: 插件配置字典
    """

    def __init__(self):
        """
        初始化插件基类

        注意：实际配置通过 initialize() 方法传递
        """
        self.config: Dict[str, Any] = {}

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        异步初始化插件

        使用配置字典初始化插件，建立连接、加载资源等。
        这是插件生命周期的第一步，必须在其他操作之前调用。

        Args:
            config: 插件配置字典，包含连接参数、认证信息等

        Example:
            plugin = MyPlugin()
            await plugin.initialize({"host": "localhost", "port": 8080})
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        健康检查

        检查插件是否正常工作，可用于服务启动时的依赖检查。

        Returns:
            True 表示健康，False 表示异常

        Example:
            healthy = await plugin.health_check()
            if not healthy:
                raise RuntimeError("Plugin is not healthy")
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        关闭插件，释放资源

        在应用关闭或插件卸载时调用，用于清理连接、释放资源。

        Example:
            await plugin.close()
        """
        pass
