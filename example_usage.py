#!/usr/bin/env python3
"""
Mem0 使用示例
"""

import asyncio
import os

from dotenv import load_dotenv

from mem0.core.memory import MemoryManager
from mem0.plugins.graph_databases.nebula import NebulaGraphStore
from mem0.plugins.models.openai_adapter import OpenAIEmbedding, OpenAIModel
from mem0.plugins.vector_stores.chroma import ChromaDBStore


async def main():
    """示例：使用 Mem0 记忆管理器"""
    
    # 加载环境变量
    load_dotenv()
    
    print("=" * 50)
    print("Mem0 记忆系统示例")
    print("=" * 50)
    
    # 1. 初始化组件
    print("\n1. 初始化组件...")
    
    # 嵌入模型
    embedding_model = OpenAIEmbedding()
    await embedding_model.initialize({
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "text-embedding-3-small",
    })
    print("  ✓ 嵌入模型初始化完成")
    
    # 大模型
    llm_model = OpenAIModel()
    await llm_model.initialize({
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4o-mini",
    })
    print("  ✓ 大模型初始化完成")
    
    # 向量存储
    vector_store = ChromaDBStore()
    await vector_store.initialize({
        "collection_name": "example_memories",
        "persist_directory": "./data/example_chroma",
    })
    print("  ✓ 向量存储初始化完成")
    
    # 图数据库（可选，如果没有 NebulaGraph 可以注释掉）
    graph_db = NebulaGraphStore()
    try:
        await graph_db.initialize({
            "host": "127.0.0.1",
            "port": 9669,
            "username": "root",
            "password": "nebula",
            "space_name": "example_mem0",
        })
        print("  ✓ 图数据库初始化完成")
    except Exception as e:
        print(f"  ⚠ 图数据库初始化失败: {e}")
        print("  继续不使用图数据库...")
        graph_db = None
    
    # 2. 创建记忆管理器
    print("\n2. 创建记忆管理器...")
    memory_manager = MemoryManager(
        vector_store=vector_store,
        graph_db=graph_db or None,  # 如果图数据库不可用则传入 None
        embedding_model=embedding_model,
        llm_model=llm_model,
    )
    print("  ✓ 记忆管理器创建完成")
    
    # 3. 模拟对话
    print("\n3. 模拟对话...")
    user_id = "user_001"
    
    conversations = [
        ("user", "你好，我叫张三，是一名软件工程师。"),
        ("assistant", "你好张三！很高兴认识你。作为一名软件工程师，你主要使用什么编程语言呢？"),
        ("user", "我主要用 Python 和 JavaScript，最近在学习 Rust。"),
        ("assistant", "Rust 是一门很棒的系统编程语言，学习曲线虽然陡峭但性能很好。"),
        ("user", "是的，我喜欢它的内存安全特性。对了，我不喜欢 Java，语法太繁琐了。"),
    ]
    
    for role, content in conversations:
        await memory_manager.add_conversation(
            user_id=user_id,
            role=role,
            content=content,
        )
        print(f"  [{role}] {content[:50]}...")
    
    # 4. 强制总结（因为对话数可能未达到阈值）
    print("\n4. 触发记忆总结...")
    memory = await memory_manager.force_summarize(user_id)
    
    if memory:
        print(f"  ✓ 记忆已创建")
        print(f"    ID: {memory.id}")
        print(f"    内容: {memory.content}")
        print(f"    实体: {[e.get('name') for e in memory.entities]}")
        print(f"    关系: {len(memory.relations)} 个")
    else:
        print("  ⚠ 没有创建记忆")
    
    # 5. 搜索记忆
    print("\n5. 搜索记忆...")
    query = "张三喜欢什么编程语言？"
    print(f"  查询: {query}")
    
    results = await memory_manager.search_memories(
        query=query,
        user_id=user_id,
        top_k=3,
    )
    
    print(f"  找到 {len(results)} 条相关记忆:")
    for i, result in enumerate(results, 1):
        print(f"    {i}. [{result.score:.2f}] {result.memory.content[:80]}...")
    
    # 6. 获取上下文
    print("\n6. 获取相关上下文...")
    context = await memory_manager.get_relevant_context(
        query="张三的技术栈",
        user_id=user_id,
    )
    print(f"  上下文:\n{context}")
    
    # 7. 添加更多对话（测试记忆合并）
    print("\n7. 添加更多对话（测试记忆合并）...")
    more_conversations = [
        ("user", "我最近在用 Python 写一些 AI 相关的项目。"),
        ("assistant", "听起来很有趣！你在做什么类型的 AI 项目？"),
        ("user", "主要是自然语言处理相关的，比如文本分类和情感分析。"),
    ]
    
    for role, content in more_conversations:
        await memory_manager.add_conversation(
            user_id=user_id,
            role=role,
            content=content,
        )
    
    # 强制总结
    new_memory = await memory_manager.force_summarize(user_id)
    if new_memory:
        print(f"  ✓ 新记忆已创建/合并")
        print(f"    内容: {new_memory.content[:100]}...")
    
    # 8. 清理
    print("\n8. 清理资源...")
    await vector_store.close()
    if graph_db:
        await graph_db.close()
    await embedding_model.close()
    await llm_model.close()
    print("  ✓ 资源已清理")
    
    print("\n" + "=" * 50)
    print("示例完成！")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
