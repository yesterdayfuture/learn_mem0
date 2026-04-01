#!/usr/bin/env python3
"""
Mem0 Memory Service - 启动脚本
"""

import os
import sys

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def main():
    """主函数"""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(f"""
╔══════════════════════════════════════════════════════════╗
║              Mem0 Memory Service                          ║
╠══════════════════════════════════════════════════════════╣
║  Version: 1.0.0                                          ║
║  Docs: http://{host}:{port}/docs                          ║
║  Health: http://{host}:{port}/health                      ║
╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "mem0.api.server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
