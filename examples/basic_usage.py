#!/usr/bin/env python3
"""
基础使用示例 - 使用 Mem0Client
"""

import asyncio
import os

from mem0.client import Mem0Client
from mem0.plugins.models.openai_adapter import OpenAIEmbedding, OpenAIModel
from mem0.plugins.vector_stores.chroma import ChromaDBStore
from mem0.plugins.graph_databases.nebula import NebulaGraphStore
from dotenv import load_dotenv
load_dotenv()


async def basic_example():
    """基础使用示例"""
    print("=" * 60)
    print("基础使用示例")
    print("=" * 60)

    # 先独立初始化各个组件
    vector_store = ChromaDBStore()
    await vector_store.initialize({
        "collection_name": os.getenv("CHROMA_COLLECTION"),
        "persist_directory": os.getenv("CHROMA_PERSIST_DIR")
    })

    embedding = OpenAIEmbedding()
    await embedding.initialize({
        "base_url": os.getenv("EMBED_BASE_URL"),
        "api_key": os.getenv("EMBED_API_KEY"),
        "model": os.getenv("EMBEDDING_MODEL")
    })

    llm = OpenAIModel()
    await llm.initialize({
        "base_url": os.getenv("LLM_BASE_URL"),
        "api_key": os.getenv("LLM_API_KEY"),
        "model": os.getenv("LLM_MODEL")
    })

    # 可选：初始化图数据库
    graph_db = NebulaGraphStore()
    try:
        await graph_db.initialize({
            "host": os.getenv("NEBULA_HOST"),
            "port": os.getenv("NEBULA_PORT"),
            "username": os.getenv("NEBULA_USER"),
            "password": os.getenv("NEBULA_PASSWORD"),
            "space_name": os.getenv("NEBULA_SPACE")
        })
    except Exception as e:
        print(f"图数据库初始化失败（可选）: {e}")
        graph_db = None

    # 使用上下文管理器自动初始化和关闭
    async with Mem0Client(
        vector_store_instance=vector_store,
        embedding_instance=embedding,
        llm_instance=llm,
        graph_db_instance=graph_db
    ) as client:
        user_id = "user_basic"

        # 1. 添加对话
        print("\n1. 添加对话...")
        conversations = [
            ("user", "你好，我叫李四，是一名产品经理。"),
            ("assistant", "你好李四！很高兴认识你。"),
            ("user", "我在一家互联网公司工作，主要负责AI产品。"),
            ("assistant", "AI产品很有前景呢！"),
            ("user", "是的，我对大语言模型特别感兴趣。"),
        ]

        for role, content in conversations:
            await client.add_conversation(user_id, role, content)
            print(f"  [{role}] {content[:40]}...")

        # 2. 强制总结
        print("\n2. 强制总结对话...")
        memory = await client.force_summarize(user_id)
        if memory:
            print(f"  ✓ 记忆已创建: {memory.id}")
            print(f"    内容: {memory.content}")
            print(f"    实体: {[e.get('name') for e in memory.entities]}")

        # 3. 搜索记忆
        print("\n3. 搜索记忆...")
        query = "李四的工作"
        results = await client.search_memories(query, user_id=user_id, top_k=3)
        print(f"  查询: {query}")
        print(f"  找到 {len(results)} 条记忆:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. [{result.score:.2f}] {result.memory.content[:60]}...")

        # 4. 获取上下文
        print("\n4. 获取相关上下文...")
        context = await client.get_relevant_context("李四的职业", user_id)
        print(f"  上下文: {context[:200]}...")

        # 5. 带记忆的聊天
        print("\n5. 带记忆的聊天...")
        response = await client.chat_with_memory(
            message="你还记得我是谁吗？",
            user_id=user_id,
        )
        print(f"  用户: 你还记得我是谁吗？")
        print(f"  助手: {response}")

        # 6. 获取统计信息
        print("\n6. 获取统计信息...")
        vector_stats = await client.get_vector_store_stats()
        print(f"  向量库: {vector_stats}")

        graph_stats = await client.get_graph_stats()
        print(f"  图数据库: {graph_stats}")


async def multi_user_example():
    """多用户示例"""
    print("\n" + "=" * 60)
    print("多用户示例")
    print("=" * 60)

    async with Mem0Client() as client:
        users = {
            "user_a": [
                ("user", "我是小明，喜欢打篮球。"),
                ("user", "我最喜欢看NBA，支持湖人队。"),
            ],
            "user_b": [
                ("user", "我是小红，喜欢画画。"),
                ("user", "我擅长水彩画和素描。"),
            ],
        }

        # 添加对话
        for user_id, conversations in users.items():
            print(f"\n  用户 {user_id}:")
            for role, content in conversations:
                await client.add_conversation(user_id, role, content)
                print(f"    [{role}] {content}")

            # 强制总结
            memory = await client.force_summarize(user_id)
            if memory:
                print(f"    ✓ 记忆: {memory.content}")

        # 分别搜索
        print("\n  搜索用户A的记忆:")
        results_a = await client.search_memories("爱好", user_id="user_a")
        for r in results_a:
            print(f"    - {r.memory.content}")

        print("\n  搜索用户B的记忆:")
        results_b = await client.search_memories("爱好", user_id="user_b")
        for r in results_b:
            print(f"    - {r.memory.content}")


async def memory_management_example():
    """记忆管理示例"""
    print("\n" + "=" * 60)
    print("记忆管理示例")
    print("=" * 60)

    async with Mem0Client() as client:
        user_id = "user_mgmt"

        # 添加一些对话
        conversations = [
            "我喜欢吃川菜，特别是麻婆豆腐。",
            "我也喜欢粤菜，清蒸鱼是我的最爱。",
            "其实我对湘菜也很感兴趣。",
        ]

        print("\n1. 添加关于食物的对话...")
        for content in conversations:
            await client.add_conversation(user_id, "user", content)
            print(f"  - {content}")

        # 强制总结
        memory = await client.force_summarize(user_id)
        if memory:
            print(f"\n2. 创建的记忆:")
            print(f"  ID: {memory.id}")
            print(f"  内容: {memory.content}")
            print(f"  重要性: {memory.importance}")

            # 获取记忆详情
            print(f"\n3. 获取记忆详情...")
            detail = await client.get_memory(memory.id)
            if detail:
                print(f"  访问次数: {detail.access_count}")
                print(f"  实体数量: {len(detail.entities)}")

            # 删除记忆
            print(f"\n4. 删除记忆...")
            success = await client.delete_memory(memory.id)
            print(f"  删除成功: {success}")

            # 验证删除
            print(f"\n5. 验证删除...")
            deleted = await client.get_memory(memory.id)
            print(f"  记忆存在: {deleted is not None}")


async def main():
    """运行所有示例"""
    try:
        await basic_example()
        # await multi_user_example()
        # await memory_management_example()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
