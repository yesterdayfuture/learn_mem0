"""
动态配置使用示例
展示如何使用 Mem0Client 支持动态加载不同的数据库和模型厂商
"""

import asyncio

from mem0.client import Mem0Client, create_client
from mem0.config import registry, Mem0Config


async def example_1_basic_usage():
    """示例1: 基础使用 - 使用默认配置"""
    print("=" * 60)
    print("示例1: 基础使用 - 使用默认配置")
    print("=" * 60)

    # 使用默认配置（从环境变量读取）
    async with Mem0Client() as client:
        # 添加对话
        conversation_id = await client.add_conversation(
            user_id="user_001",
            role="user",
            content="我喜欢打篮球和游泳",
        )
        print(f"✅ 添加对话成功，ID: {conversation_id}")

        # 搜索记忆
        results = await client.search_memories("运动爱好", user_id="user_001")
        print(f"✅ 搜索到 {len(results)} 条记忆")


async def example_2_custom_config():
    """示例2: 使用自定义配置"""
    print("\n" + "=" * 60)
    print("示例2: 使用自定义配置")
    print("=" * 60)

    # 自定义配置
    config = {
        "vector_store": {
            "provider": "chromadb",
            "config": {
                "collection_name": "custom_memories",
                "persist_directory": "./data/custom_chroma",
            },
        },
        "graph_db": {
            "provider": "nebula",
            "config": {
                "host": "127.0.0.1",
                "port": 9669,
                "username": "root",
                "password": "nebula",
                "space_name": "custom_mem0",
            },
        },
        "embedding": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "dimension": 1536,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
            },
        },
    }

    async with Mem0Client(config) as client:
        # 添加对话
        conversation_id = await client.add_conversation(
            user_id="user_002",
            role="user",
            content="我是一名软件工程师",
        )
        print(f"✅ 添加对话成功，ID: {conversation_id}")

        # 获取相关上下文
        context = await client.get_relevant_context("职业", user_id="user_002")
        print(f"✅ 获取到上下文: {context[:100]}...")


async def example_3_convenience_function():
    """示例3: 使用便捷函数创建客户端"""
    print("\n" + "=" * 60)
    print("示例3: 使用便捷函数创建客户端")
    print("=" * 60)

    # 使用便捷函数快速创建客户端
    client = await create_client(
        vector_store="chromadb",
        graph_db="nebula",
        embedding="openai",
        llm="openai",
        vector_store_config={
            "collection_name": "quick_memories",
            "persist_directory": "./data/quick_chroma",
        },
    )

    try:
        # 添加对话
        conversation_id = await client.add_conversation(
            user_id="user_003",
            role="user",
            content="我喜欢阅读科幻小说",
        )
        print(f"✅ 添加对话成功，ID: {conversation_id}")

        # 带记忆的聊天
        response = await client.chat_with_memory(
            message="推荐几本好看的科幻小说",
            user_id="user_003",
        )
        print(f"✅ AI回复: {response[:100]}...")

    finally:
        await client.close()


async def example_4_list_plugins():
    """示例4: 列出所有可用插件"""
    print("\n" + "=" * 60)
    print("示例4: 列出所有可用插件")
    print("=" * 60)

    print("📦 向量存储插件:")
    for name in registry.list_vector_stores():
        print(f"   - {name}")

    print("\n🕸️ 图数据库插件:")
    for name in registry.list_graph_databases():
        print(f"   - {name}")

    print("\n🔤 嵌入模型插件:")
    for name in registry.list_embeddings():
        print(f"   - {name}")

    print("\n🤖 大模型插件:")
    for name in registry.list_llms():
        print(f"   - {name}")


async def example_5_get_plugin_config():
    """示例5: 获取插件配置模板"""
    print("\n" + "=" * 60)
    print("示例5: 获取插件配置模板")
    print("=" * 60)

    # 获取 ChromaDB 配置模板
    chroma_config = registry.get_config_template("vector_store", "chromadb")
    print("📦 ChromaDB 配置模板:")
    for key, value in chroma_config.items():
        print(f"   {key}: {value}")

    # 获取 NebulaGraph 配置模板
    nebula_config = registry.get_config_template("graph_db", "nebula")
    print("\n🕸️ NebulaGraph 配置模板:")
    for key, value in nebula_config.items():
        print(f"   {key}: {value}")

    # 获取 OpenAI 配置模板
    openai_config = registry.get_config_template("embedding", "openai")
    print("\n🔤 OpenAI Embedding 配置模板:")
    for key, value in openai_config.items():
        print(f"   {key}: {value}")


async def example_6_switch_providers():
    """示例6: 运行时切换提供商（可视化界面功能）"""
    print("\n" + "=" * 60)
    print("示例6: 运行时切换提供商")
    print("=" * 60)

    # 创建初始客户端
    async with Mem0Client() as client:
        print(f"📦 当前向量存储: {client.config.get_vector_store_config().get('provider')}")
        print(f"🕸️ 当前图数据库: {client.config.get_graph_db_config().get('provider')}")
        print(f"🔤 当前嵌入模型: {client.config.get_embedding_config().get('provider')}")
        print(f"🤖 当前大模型: {client.config.get_llm_config().get('provider')}")

        # 获取统计信息
        vector_stats = await client.get_vector_store_stats()
        print(f"\n📊 向量库统计:")
        print(f"   总记录数: {vector_stats.get('total_records', 0)}")
        print(f"   提供商: {vector_stats.get('provider', 'unknown')}")


async def example_7_custom_plugin_registration():
    """示例7: 注册自定义插件（演示如何扩展）"""
    print("\n" + "=" * 60)
    print("示例7: 注册自定义插件（演示）")
    print("=" * 60)

    print("💡 要注册自定义插件，请按照以下步骤:")
    print()
    print("1. 创建自定义插件类，实现对应接口:")
    print("""
   from mem0.plugins.vector_stores.base import VectorStoreInterface

   class MyCustomVectorStore(VectorStoreInterface):
       async def initialize(self, config: Dict[str, Any]) -> None:
           # 初始化逻辑
           pass

       async def add(self, records: List[VectorStoreRecord]) -> None:
           # 添加记录逻辑
           pass

       # ... 实现其他方法
    """)

    print("2. 注册插件到注册中心:")
    print("""
   from mem0.config import registry

   registry.register_vector_store(
       "mycustom",
       MyCustomVectorStore,
       config_template={
           "host": "localhost",
           "port": 8080,
       }
   )
    """)

    print("3. 使用自定义插件:")
    print("""
   config = {
       "vector_store": {
           "provider": "mycustom",
           "config": {
               "host": "localhost",
               "port": 8080,
           }
       }
   }

   client = Mem0Client(config)
   await client.initialize()
    """)


async def example_8_config_from_env():
    """示例8: 从环境变量创建配置"""
    print("\n" + "=" * 60)
    print("示例8: 从环境变量创建配置")
    print("=" * 60)

    # 从环境变量创建配置
    config = Mem0Config.from_env()

    print("📋 从环境变量读取的配置:")
    print(f"   向量存储: {config.get_vector_store_config()}")
    print(f"   图数据库: {config.get_graph_db_config()}")
    print(f"   嵌入模型: {config.get_embedding_config()}")
    print(f"   大模型: {config.get_llm_config()}")


async def main():
    """主函数"""
    print("🚀 Mem0 动态配置使用示例")
    print()

    try:
        # 运行所有示例
        await example_4_list_plugins()
        await example_5_get_plugin_config()
        await example_8_config_from_env()
        await example_1_basic_usage()
        await example_2_custom_config()
        await example_3_convenience_function()
        await example_6_switch_providers()
        await example_7_custom_plugin_registration()

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("所有示例执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
