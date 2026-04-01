"""
Mem0 客户端使用示例
演示多种初始化方式
"""

import asyncio

from mem0.client import Mem0Client, create_client, create_client_with_instances
from mem0.plugins.models.openai_adapter import OpenAIEmbedding, OpenAIModel
from mem0.plugins.vector_stores.chroma import ChromaDB
from mem0.plugins.graph_databases.nebula import NebulaGraph


async def example_1_default_config():
    """方式1: 使用默认配置（从环境变量读取）"""
    print("=" * 50)
    print("示例 1: 使用默认配置")
    print("=" * 50)

    # 确保设置了环境变量：
    # export OPENAI_API_KEY="your-api-key"
    # export NEBULA_HOST="127.0.0.1"
    # export NEBULA_PASSWORD="nebula"

    client = Mem0Client()
    await client.initialize()

    # 添加对话
    conversation_id = await client.add_conversation(
        user_id="user_001",
        role="user",
        content="我叫张三，是一名软件工程师"
    )
    print(f"添加对话成功，ID: {conversation_id}")

    # 搜索记忆
    results = await client.search_memories("用户的名字", user_id="user_001")
    print(f"搜索到 {len(results)} 条记忆")

    await client.close()


async def example_2_provider_names():
    """方式2: 指定 provider 名称"""
    print("\n" + "=" * 50)
    print("示例 2: 指定 provider 名称")
    print("=" * 50)

    client = Mem0Client(
        vector_store="chromadb",
        graph_db="nebula",
        embedding="openai",
        llm="openai",
    )
    await client.initialize()

    # 带记忆的聊天
    response = await client.chat_with_memory(
        message="你好，请介绍一下你自己",
        user_id="user_002",
        session_id="session_001"
    )
    print(f"助手回复: {response}")

    await client.close()


async def example_3_full_config():
    """方式3: 使用完整配置字典"""
    print("\n" + "=" * 50)
    print("示例 3: 使用完整配置字典")
    print("=" * 50)

    config = {
        "vector_store": {
            "provider": "chromadb",
            "config": {
                "collection_name": "my_memories",
                "persist_directory": "./data/my_chroma"
            }
        },
        "graph_db": {
            "provider": "nebula",
            "config": {
                "host": "127.0.0.1",
                "port": 9669,
                "username": "root",
                "password": "nebula",
                "space_name": "my_graph"
            }
        },
        "embedding": {
            "provider": "openai",
            "config": {
                "api_key": "your-api-key",
                "model": "text-embedding-3-small"
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": "your-api-key",
                "model": "gpt-4o-mini"
            }
        }
    }

    client = Mem0Client(config)
    await client.initialize()

    # 添加对话
    await client.add_conversation(
        user_id="user_003",
        role="user",
        content="我喜欢用 Python 编程"
    )

    # 获取相关上下文
    context = await client.get_relevant_context("编程语言", user_id="user_003")
    print(f"相关上下文: {context}")

    await client.close()


async def example_4_external_instances():
    """方式4: 传入已初始化的组件实例"""
    print("\n" + "=" * 50)
    print("示例 4: 传入已初始化的组件实例")
    print("=" * 50)

    # 先独立初始化各个组件
    vector_store = ChromaDB()
    await vector_store.initialize({
        "collection_name": "external_memories",
        "persist_directory": "./data/external_chroma"
    })

    embedding = OpenAIEmbedding()
    await embedding.initialize({
        "api_key": "your-api-key",
        "model": "text-embedding-3-small"
    })

    llm = OpenAIModel()
    await llm.initialize({
        "api_key": "your-api-key",
        "model": "gpt-4o-mini"
    })

    # 可选：初始化图数据库
    graph_db = NebulaGraph()
    try:
        await graph_db.initialize({
            "host": "127.0.0.1",
            "port": 9669,
            "username": "root",
            "password": "nebula",
            "space_name": "external_graph"
        })
    except Exception as e:
        print(f"图数据库初始化失败（可选）: {e}")
        graph_db = None

    # 使用已初始化的组件创建客户端
    client = Mem0Client(
        vector_store_instance=vector_store,
        graph_db_instance=graph_db,
        embedding_instance=embedding,
        llm_instance=llm,
    )
    await client.initialize()

    # 添加对话
    conversation_id = await client.add_conversation(
        user_id="user_004",
        role="user",
        content="我在学习机器学习"
    )
    print(f"添加对话成功，ID: {conversation_id}")

    # 注意：关闭客户端时，外部传入的组件不会被关闭
    # 需要手动关闭外部组件
    await client.close()

    # 手动关闭外部组件
    await vector_store.close()
    if graph_db:
        await graph_db.close()
    await embedding.close()
    await llm.close()


async def example_5_helper_function():
    """方式5: 使用便捷函数 create_client"""
    print("\n" + "=" * 50)
    print("示例 5: 使用便捷函数 create_client")
    print("=" * 50)

    # 一行代码创建并初始化客户端
    client = await create_client(
        vector_store="chromadb",
        graph_db="nebula",
        embedding="openai",
        llm="openai",
        # 额外的配置参数
        vector_store_config={
            "collection_name": "helper_memories"
        }
    )

    # 添加对话
    await client.add_conversation(
        user_id="user_005",
        role="user",
        content="我在北京工作"
    )

    # 搜索记忆
    results = await client.search_memories("工作地点", user_id="user_005")
    for result in results:
        print(f"记忆: {result.memory.content}")

    await client.close()


async def example_6_helper_with_instances():
    """方式6: 使用便捷函数 create_client_with_instances"""
    print("\n" + "=" * 50)
    print("示例 6: 使用便捷函数 create_client_with_instances")
    print("=" * 50)

    # 先独立初始化各个组件
    vector_store = ChromaDB()
    await vector_store.initialize({
        "collection_name": "instances_memories"
    })

    embedding = OpenAIEmbedding()
    await embedding.initialize({
        "api_key": "your-api-key"
    })

    llm = OpenAIModel()
    await llm.initialize({
        "api_key": "your-api-key"
    })

    # 使用便捷函数创建客户端
    client = await create_client_with_instances(
        vector_store=vector_store,
        embedding=embedding,
        llm=llm,
    )

    # 添加对话
    await client.add_conversation(
        user_id="user_006",
        role="user",
        content="我喜欢打篮球"
    )

    # 带记忆的聊天
    response = await client.chat_with_memory(
        message="我的爱好是什么？",
        user_id="user_006"
    )
    print(f"助手回复: {response}")

    await client.close()

    # 手动关闭外部组件
    await vector_store.close()
    await embedding.close()
    await llm.close()


async def example_7_context_manager():
    """方式7: 使用异步上下文管理器"""
    print("\n" + "=" * 50)
    print("示例 7: 使用异步上下文管理器")
    print("=" * 50)

    async with Mem0Client(
        vector_store="chromadb",
        embedding="openai",
        llm="openai",
    ) as client:
        # 添加对话
        await client.add_conversation(
            user_id="user_007",
            role="user",
            content="我在学习 FastAPI"
        )

        # 获取统计信息
        stats = await client.get_vector_store_stats()
        print(f"向量库统计: {stats}")

    # 退出上下文时自动关闭


async def example_8_mixed_instances():
    """方式8: 混合使用外部实例和 provider 名称"""
    print("\n" + "=" * 50)
    print("示例 8: 混合使用外部实例和 provider 名称")
    print("=" * 50)

    # 先独立初始化向量存储
    vector_store = ChromaDB()
    await vector_store.initialize({
        "collection_name": "mixed_memories"
    })

    # 使用外部向量存储，但使用 provider 名称初始化其他组件
    client = Mem0Client(
        vector_store_instance=vector_store,  # 外部实例
        embedding="openai",  # provider 名称
        llm="openai",  # provider 名称
    )
    await client.initialize()

    # 添加对话
    await client.add_conversation(
        user_id="user_008",
        role="user",
        content="混合使用方式测试"
    )

    await client.close()

    # 手动关闭外部组件
    await vector_store.close()


async def main():
    """主函数"""
    print("Mem0 客户端使用示例")
    print("=" * 50)

    # 运行所有示例（请根据实际环境选择性运行）
    try:
        await example_1_default_config()
    except Exception as e:
        print(f"示例 1 失败: {e}")

    try:
        await example_2_provider_names()
    except Exception as e:
        print(f"示例 2 失败: {e}")

    try:
        await example_3_full_config()
    except Exception as e:
        print(f"示例 3 失败: {e}")

    try:
        await example_4_external_instances()
    except Exception as e:
        print(f"示例 4 失败: {e}")

    try:
        await example_5_helper_function()
    except Exception as e:
        print(f"示例 5 失败: {e}")

    try:
        await example_6_helper_with_instances()
    except Exception as e:
        print(f"示例 6 失败: {e}")

    try:
        await example_7_context_manager()
    except Exception as e:
        print(f"示例 7 失败: {e}")

    try:
        await example_8_mixed_instances()
    except Exception as e:
        print(f"示例 8 失败: {e}")

    print("\n" + "=" * 50)
    print("所有示例运行完成")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
