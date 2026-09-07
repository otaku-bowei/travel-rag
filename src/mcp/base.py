import threading
import time

from src.mcp.travel_param_server import mcp


def start_mcp_in_thread(port: int = 9999):
    """在后台线程启动 MCP server

    类似 Java 内嵌 Tomcat：业务代码和 server 同进程。
    """

    def run_server():
        mcp.run(transport="streamable-http")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    time.sleep(2)  # 等 server 起来

    print(f"✅ MCP server 已启动 (后台线程, port: {port})")
    return thread


def stop_mcp_thread(thread):
    """后台线程 daemon=True，主进程退出时自动关，不需要手动 stop"""
    if thread and thread.is_alive():
        # daemon线程会随主进程退出，这里只是示例
        pass