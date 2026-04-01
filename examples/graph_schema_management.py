"""
图数据库 Schema 管理示例
演示如何使用节点标签（Tag）和边类型（Edge Type）管理功能
"""

import asyncio

from mem0.client import Mem0Client


async def example_1_list_tags():
    """示例1: 列出所有节点标签"""
    print("=" * 50)
    print("示例1: 列出所有节点标签")
    print("=" * 50)

    async with Mem0Client(graph_db="nebula") as client:
        # 列出所有节点标签
        tags = await client.list_node_tags()
        print(f"共有 {len(tags)} 个节点标签:")
        for tag in tags:
            print(f"  - {tag['name']}")


async def example_2_create_tag():
    """示例2: 创建新的节点标签"""
    print("\n" + "=" * 50)
    print("示例2: 创建新的节点标签")
    print("=" * 50)

    async with Mem0Client(graph_db="nebula") as client:
        # 创建 Person 标签
        success = await client.create_node_tag(
            tag_name="Person",
            properties={
                "name": "string",
                "age": "int",
                "email": "string",
                "created_at": "timestamp"
            }
        )
        print(f"创建 Person 标签: {'成功' if success else '失败'}")

        # 创建 Company 标签
        success = await client.create_node_tag(
            tag_name="Company",
            properties={
                "name": "string",
                "industry": "string",
                "founded_year": "int"
            }
        )
        print(f"创建 Company 标签: {'成功' if success else '失败'}")

        # 列出所有标签
        tags = await client.list_node_tags()
        print(f"\n现在共有 {len(tags)} 个节点标签:")
        for tag in tags:
            print(f"  - {tag['name']}")


async def example_3_get_tag_info():
    """示例3: 获取标签详细信息"""
    print("\n" + "=" * 50)
    print("示例3: 获取标签详细信息")
    print("=" * 50)

    async with Mem0Client(graph_db="nebula") as client:
        # 获取 Entity 标签的详细信息
        info = await client.get_node_tag_info("Entity")
        if info:
            print(f"标签名称: {info['name']}")
            print(f"实体数量: {info['entity_count']}")
            print("属性定义:")
            for prop in info['properties']:
                print(f"  - {prop['name']}: {prop['type']}")
        else:
            print("标签不存在")


async def example_4_list_edge_types():
    """示例4: 列出所有边类型"""
    print("\n" + "=" * 50)
    print("示例4: 列出所有边类型")
    print("=" * 50)

    async with Mem0Client(graph_db="nebula") as client:
        # 列出所有边类型
        edge_types = await client.list_edge_types()
        print(f"共有 {len(edge_types)} 个边类型:")
        for edge_type in edge_types:
            print(f"  - {edge_type['name']}")


async def example_5_create_edge_type():
    """示例5: 创建新的边类型"""
    print("\n" + "=" * 50)
    print("示例5: 创建新的边类型")
    print("=" * 50)

    async with Mem0Client(graph_db="nebula") as client:
        # 创建 WORKS_AT 边类型
        success = await client.create_edge_type(
            edge_type="WORKS_AT",
            properties={
                "since": "timestamp",
                "position": "string",
                "department": "string"
            }
        )
        print(f"创建 WORKS_AT 边类型: {'成功' if success else '失败'}")

        # 创建 KNOWS 边类型
        success = await client.create_edge_type(
            edge_type="KNOWS",
            properties={
                "since": "timestamp",
                "relationship": "string"
            }
        )
        print(f"创建 KNOWS 边类型: {'成功' if success else '失败'}")

        # 列出所有边类型
        edge_types = await client.list_edge_types()
        print(f"\n现在共有 {len(edge_types)} 个边类型:")
        for edge_type in edge_types:
            print(f"  - {edge_type['name']}")


async def example_6_get_edge_type_info():
    """示例6: 获取边类型详细信息"""
    print("\n" + "=" * 50)
    print("示例6: 获取边类型详细信息")
    print("=" * 50)

    async with Mem0Client(graph_db="nebula") as client:
        # 获取 RELATES 边类型的详细信息
        info = await client.get_edge_type_info("RELATES")
        if info:
            print(f"边类型名称: {info['name']}")
            print(f"关系数量: {info['edge_count']}")
            print("属性定义:")
            for prop in info['properties']:
                print(f"  - {prop['name']}: {prop['type']}")
        else:
            print("边类型不存在")


async def example_7_delete_tag():
    """示例7: 删除标签（谨慎使用）"""
    print("\n" + "=" * 50)
    print("示例7: 删除标签")
    print("=" * 50)

    async with Mem0Client(graph_db="nebula") as client:
        # 先创建一个临时标签
        await client.create_node_tag(
            tag_name="TempTag",
            properties={"name": "string"}
        )
        print("创建临时标签 TempTag")

        # 列出所有标签
        tags = await client.list_node_tags()
        print(f"删除前共有 {len(tags)} 个标签")

        # 删除临时标签
        success = await client.delete_node_tag("TempTag")
        print(f"删除 TempTag 标签: {'成功' if success else '失败'}")

        # 再次列出所有标签
        tags = await client.list_node_tags()
        print(f"删除后共有 {len(tags)} 个标签")


async def example_8_complete_workflow():
    """示例8: 完整的 Schema 管理工作流"""
    print("\n" + "=" * 50)
    print("示例8: 完整的 Schema 管理工作流")
    print("=" * 50)

    async with Mem0Client(graph_db="nebula") as client:
        # 1. 创建完整的 Schema
        print("\n1. 创建节点标签...")

        await client.create_node_tag(
            tag_name="Person",
            properties={
                "name": "string",
                "age": "int",
                "occupation": "string",
                "location": "string"
            }
        )
        print("  - 创建 Person 标签")

        await client.create_node_tag(
            tag_name="Company",
            properties={
                "name": "string",
                "industry": "string",
                "size": "int"
            }
        )
        print("  - 创建 Company 标签")

        await client.create_node_tag(
            tag_name="Skill",
            properties={
                "name": "string",
                "category": "string",
                "level": "string"
            }
        )
        print("  - 创建 Skill 标签")

        # 2. 创建边类型
        print("\n2. 创建边类型...")

        await client.create_edge_type(
            edge_type="WORKS_AT",
            properties={
                "since": "timestamp",
                "position": "string"
            }
        )
        print("  - 创建 WORKS_AT 边类型")

        await client.create_edge_type(
            edge_type="HAS_SKILL",
            properties={
                "proficiency": "double",
                "years_experience": "int"
            }
        )
        print("  - 创建 HAS_SKILL 边类型")

        await client.create_edge_type(
            edge_type="COLLEAGUE",
            properties={
                "since": "timestamp"
            }
        )
        print("  - 创建 COLLEAGUE 边类型")

        # 3. 查看所有标签和边类型
        print("\n3. Schema 概览:")

        tags = await client.list_node_tags()
        print(f"\n  节点标签 ({len(tags)} 个):")
        for tag in tags:
            info = await client.get_node_tag_info(tag['name'])
            prop_count = len(info['properties']) if info else 0
            entity_count = info['entity_count'] if info else 0
            print(f"    - {tag['name']}: {prop_count} 个属性, {entity_count} 个实体")

        edge_types = await client.list_edge_types()
        print(f"\n  边类型 ({len(edge_types)} 个):")
        for edge_type in edge_types:
            info = await client.get_edge_type_info(edge_type['name'])
            prop_count = len(info['properties']) if info else 0
            edge_count = info['edge_count'] if info else 0
            print(f"    - {edge_type['name']}: {prop_count} 个属性, {edge_count} 个关系")

        # 4. 查看详细信息
        print("\n4. Person 标签详细信息:")
        info = await client.get_node_tag_info("Person")
        if info:
            print(f"  名称: {info['name']}")
            print(f"  实体数量: {info['entity_count']}")
            print("  属性:")
            for prop in info['properties']:
                print(f"    - {prop['name']}: {prop['type']}")

        print("\n5. WORKS_AT 边类型详细信息:")
        info = await client.get_edge_type_info("WORKS_AT")
        if info:
            print(f"  名称: {info['name']}")
            print(f"  关系数量: {info['edge_count']}")
            print("  属性:")
            for prop in info['properties']:
                print(f"    - {prop['name']}: {prop['type']}")

        # 6. 清理（可选）
        print("\n6. 清理测试数据...")
        await client.delete_node_tag("Person")
        await client.delete_node_tag("Company")
        await client.delete_node_tag("Skill")
        await client.delete_edge_type("WORKS_AT")
        await client.delete_edge_type("HAS_SKILL")
        await client.delete_edge_type("COLLEAGUE")
        print("  已删除所有测试标签和边类型")


async def main():
    """主函数"""
    print("图数据库 Schema 管理示例")
    print("=" * 50)
    print("\n这些示例演示如何管理图数据库的 Schema:")
    print("- 创建、列出、查看、删除节点标签")
    print("- 创建、列出、查看、删除边类型")
    print("\n请确保:")
    print("1. NebulaGraph 服务已启动")
    print("2. 环境变量中配置了正确的连接信息")
    print("3. 有足够的权限执行 Schema 操作")
    print()

    # 运行示例（请根据实际环境选择性运行）
    examples = [
        ("列出节点标签", example_1_list_tags),
        ("创建节点标签", example_2_create_tag),
        ("获取标签信息", example_3_get_tag_info),
        ("列出边类型", example_4_list_edge_types),
        ("创建边类型", example_5_create_edge_type),
        ("获取边类型信息", example_6_get_edge_type_info),
        ("删除标签", example_7_delete_tag),
        ("完整工作流", example_8_complete_workflow),
    ]

    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\n示例 '{name}' 失败: {e}")
            print("跳过此示例，继续下一个...")
            continue

    print("\n" + "=" * 50)
    print("所有示例运行完成")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
