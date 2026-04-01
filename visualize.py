#!/usr/bin/env python3
"""
Mem0 数据可视化工具 - 启动脚本

用于可视化查看 ChromaDB 和 NebulaGraph 中的数据
"""

import os
import sys

import uvicorn
from dotenv import load_dotenv


def main():
    """主函数"""
    # 加载环境变量
    load_dotenv()

    host = os.getenv("VIZ_HOST", "0.0.0.0")
    port = int(os.getenv("VIZ_PORT", "8080"))

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           Mem0 数据可视化工具                             ║
╠══════════════════════════════════════════════════════════╣
║  访问地址: http://{host}:{port}                          ║
╚══════════════════════════════════════════════════════════╝

功能:
  • 查看 ChromaDB 向量库中的记忆数据
  • 搜索和筛选记忆内容
  • 查看 NebulaGraph 图数据库中的实体和关系
  • 支持实体名称模糊查询
  • 删除记忆记录

按 Ctrl+C 停止服务
    """)

    uvicorn.run(
        "web.visualization:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n服务已停止")
        sys.exit(0)
