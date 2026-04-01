#!/usr/bin/env python3
"""
高级使用示例 - 批量处理、自定义配置、记忆整合
"""

import asyncio
from datetime import datetime

from mem0.client import Mem0Client
from mem0.core.models import MemoryType


async def batch_processing_example():
    """批量处理示例"""
    print("=" * 60)
    print("批量处理示例")
    print("=" * 60)

    # 自定义配置
    config = {
        "openai_api_key": None,  # 从环境变量读取
        "llm_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "chroma_collection": "batch_test",
        "chroma_persist_dir": "./data/batch_test",
    }

    async with Mem0Client(config) as client:
        user_id = "user_batch"

        # 批量添加对话
        print("\n1. 批量添加对话...")
        dialogues = [
            {"role": "user", "content": "你好，我是王五。"},
            {"role": "assistant", "content": "你好王五！很高兴认识你。"},
            {"role": "user", "content": "我是一名数据科学家。"},
            {"role": "assistant", "content": "数据科学是很有挑战的领域！"},
            {"role": "user", "content": "我擅长Python和机器学习。"},
            {"role": "assistant", "content": "Python是数据科学的首选语言。"},
            {"role": "user", "content": "我也在用TensorFlow和PyTorch。"},
            {"role": "assistant", "content": "这两个框架都很强大！"},
        ]

        for dialogue in dialogues:
            await client.add_conversation(
                user_id=user_id,
                role=dialogue["role"],
                content=dialogue["content"],
            )
        print(f"  ✓ 添加了 {len(dialogues)} 条对话")

        # 强制总结
        print("\n2. 强制总结...")
        memory = await client.force_summarize(user_id)
        if memory:
            print(f"  ✓ 记忆已创建")
            print(f"    内容: {memory.content[:100]}...")

        # 批量查询
        print("\n3. 批量查询...")
        queries = [
            "王五的职业",
            "王五的技能",
            "王五使用的工具",
        ]

        for query in queries:
            results = await client.search_memories(query, user_id=user_id, top_k=2)
            print(f"\n  查询: {query}")
            for r in results:
                print(f"    - [{r.score:.2f}] {r.memory.content[:50]}...")


async def custom_config_example():
    """自定义配置示例"""
    print("\n" + "=" * 60)
    print("自定义配置示例")
    print("=" * 60)

    async with Mem0Client() as client:
        user_id = "user_config"

        # 修改配置参数
        print("\n1. 修改配置参数...")
        client.update_config(
            summary_threshold=2,  # 2条对话就触发总结
            similarity_threshold=0.8,  # 更高的相似度阈值
            max_memories_per_query=3,
        )
        print("  ✓ 配置已更新")
        print("    - 总结阈值: 2")
        print("    - 相似度阈值: 0.8")
        print("    - 最大记忆数: 3")

        # 添加对话（现在2条就会触发总结）
        print("\n2. 添加对话（2条触发总结）...")
        await client.add_conversation(user_id, "user", "我喜欢旅游，去过很多地方。")
        await client.add_conversation(user_id, "assistant", "旅游是很好的体验！")
        print("  ✓ 对话已添加，应该已自动总结")

        # 添加更多对话
        await client.add_conversation(user_id, "user", "我最喜欢去日本，食物很好吃。")
        await client.add_conversation(user_id, "assistant", "日本料理确实很精致。")
        print("  ✓ 又添加了2条对话")


async def memory_consolidation_example():
    """记忆整合示例"""
    print("\n" + "=" * 60)
    print("记忆整合示例")
    print("=" * 60)

    async with Mem0Client() as client:
        user_id = "user_consolidate"

        # 添加多组相似的对话
        print("\n1. 添加相似对话（模拟重复记忆）...")

        # 第一组
        conversations_1 = [
            ("user", "我喜欢喝咖啡，每天早上都要喝。"),
            ("assistant", "咖啡是很多人的必需品。"),
        ]
        for role, content in conversations_1:
            await client.add_conversation(user_id, role, content)
        memory1 = await client.force_summarize(user_id)
        print(f"  ✓ 第一组记忆: {memory1.id if memory1 else 'None'}")

        # 第二组（相似内容）
        conversations_2 = [
            ("user", "我每天早上都会喝一杯咖啡。"),
            ("assistant", "咖啡可以帮助提神。"),
        ]
        for role, content in conversations_2:
            await client.add_conversation(user_id, role, content)
        memory2 = await client.force_summarize(user_id)
        print(f"  ✓ 第二组记忆: {memory2.id if memory2 else 'None'}")

        # 第三组（相似内容）
        conversations_3 = [
            ("user", "咖啡是我每天必不可少的饮品。"),
            ("assistant", "适量喝咖啡对健康有益。"),
        ]
        for role, content in conversations_3:
            await client.add_conversation(user_id, role, content)
        memory3 = await client.force_summarize(user_id)
        print(f"  ✓ 第三组记忆: {memory3.id if memory3 else 'None'}")

        # 搜索当前记忆
        print("\n2. 搜索当前记忆...")
        results = await client.search_memories("咖啡", user_id=user_id, top_k=10)
        print(f"  找到 {len(results)} 条记忆")
        for r in results:
            print(f"    - {r.memory.content[:60]}...")

        # 执行记忆整合
        print("\n3. 执行记忆整合...")
        await client.consolidate_memories(user_id)
        print("  ✓ 记忆整合完成")

        # 再次搜索
        print("\n4. 整合后搜索...")
        results = await client.search_memories("咖啡", user_id=user_id, top_k=10)
        print(f"  找到 {len(results)} 条记忆（应该已合并）")
        for r in results:
            print(f"    - {r.memory.content[:60]}...")


async def chat_session_example():
    """聊天会话示例"""
    print("\n" + "=" * 60)
    print("聊天会话示例")
    print("=" * 60)

    async with Mem0Client() as client:
        user_id = "user_chat"
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n会话ID: {session_id}")
        print("-" * 40)

        # 模拟多轮对话
        messages = [
            "你好，我想了解一下人工智能。",
            "什么是机器学习？",
            "深度学习和大语言模型有什么关系？",
            "你能推荐一些学习资源吗？",
        ]

        for i, message in enumerate(messages, 1):
            print(f"\n  轮次 {i}:")
            print(f"    用户: {message}")

            response = await client.chat_with_memory(
                message=message,
                user_id=user_id,
                session_id=session_id,
            )
            print(f"    助手: {response[:100]}...")

        # 查看积累的记忆
        print("\n" + "-" * 40)
        print("会话结束，查看记忆:")
        results = await client.search_memories("AI学习", user_id=user_id)
        for r in results:
            print(f"  - {r.memory.content[:80]}...")


async def entity_exploration_example():
    """实体探索示例"""
    print("\n" + "=" * 60)
    print("实体探索示例")
    print("=" * 60)

    async with Mem0Client() as client:
        user_id = "user_entities"

        # 添加包含多个实体的对话
        print("\n1. 添加包含实体的对话...")
        conversations = [
            "我在Google工作，和Jeff Dean一起做过项目。",
            "我们使用TensorFlow和Kubernetes构建系统。",
            "公司总部在Mountain View，但我通常在纽约办公室工作。",
        ]

        for content in conversations:
            await client.add_conversation(user_id, "user", content)
            print(f"  - {content}")

        # 强制总结
        memory = await client.force_summarize(user_id)
        if memory:
            print(f"\n2. 提取的实体:")
            for entity in memory.entities:
                print(f"    - {entity.get('name')} ({entity.get('type')}): {entity.get('description', '')}")

            print(f"\n3. 提取的关系:")
            for relation in memory.relations:
                print(f"    - {relation.get('source')} --[{relation.get('type')}]--> {relation.get('target')}")

        # 搜索实体
        print(f"\n4. 搜索实体...")
        try:
            entities = await client.search_entities("Google", fuzzy=True)
            print(f"  找到 {len(entities)} 个相关实体")
            for e in entities:
                print(f"    - {e}")
        except Exception as e:
            print(f"  图数据库查询失败: {e}")


async def main():
    """运行所有高级示例"""
    examples = [
        batch_processing_example,
        custom_config_example,
        memory_consolidation_example,
        chat_session_example,
        entity_exploration_example,
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"\n  错误: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 60)
    print("所有高级示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
