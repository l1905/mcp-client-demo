import asyncio
import json
import re
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params.function_definition import FunctionDefinition

from dotenv import load_dotenv
import os 

load_dotenv()  # load environment variables from .env

class MCPMultiServerClient:
    def __init__(self):
        # 初始化客户端对象
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI()
        # 存储多个服务器连接
        self.servers = {}  # 格式: {server_name: {session, stdio, write, tools_mapping}}
        self.tool_server_map = {}  # 工具名称到服务器的映射: {tool_name: server_name}
    
    async def connect_to_servers(self, config_path: str):
        """连接到配置文件中指定的所有服务器
        
        Args:
            config_path: 服务器配置文件路径
        """
        # 加载并解析配置文件
        with open(config_path, 'r') as f:
            config_text = f.read()
            # 移除JSON中的注释
            # config_text = re.sub(r'//.*$', '', config_text, flags=re.MULTILINE)
            config = json.loads(config_text)
        
        # 连接到每个服务器
        for server_name, server_config in config["servers"].items():
            await self.connect_to_server(server_name, server_config)
        
        print(f"\n已连接到 {len(self.servers)} 个服务器")
        print(f"可用工具总数: {len(self.tool_server_map)}")
        print(f"工具列表: {list(self.tool_server_map.keys())}")

    async def connect_to_server(self, server_name: str, server_config: Dict[str, Any]):
        """连接到单个服务器
        
        Args:
            server_name: 服务器名称
            server_config: 服务器配置信息
        """
        command = server_config["command"]
        args = server_config["args"]
        
        print(f"\n正在连接到服务器 '{server_name}'...")
        
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        
        await session.initialize()
        
        # 获取服务器提供的工具
        response = await session.list_tools()
        tools = response.tools
        
        # 存储服务器信息
        self.servers[server_name] = {
            "session": session,
            "stdio": stdio,
            "write": write,
            "tools": {tool.name: tool for tool in tools}
        }
        
        # 更新工具到服务器的映射
        for tool in tools:
            self.tool_server_map[tool.name] = server_name
        
        print(f"已连接到服务器 '{server_name}' 并获取工具: {[tool.name for tool in tools]}")

    async def process_tool_call(self, tool_call) -> Dict[str, Any]:
        """处理单个工具调用并返回工具消息"""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        # 找到对应的服务器
        server_name = self.tool_server_map.get(tool_name)
        if not server_name:
            return {
                "role": "tool",
                "content": json.dumps({"error": f"工具 '{tool_name}' 在已连接的服务器中不存在"}),
                "tool_call_id": tool_call.id,
            }
        
        # 获取服务器会话
        server = self.servers[server_name]
        session = server["session"]
        
        # 执行工具调用
        result = await session.call_tool(tool_name, tool_args)
        
        # 处理结果内容
        content_data = {}
        if result.isError:
            content_data = {"error": str(result.content)}
        else:
            results = []
            for item in result.content:
                if item.type == "text":
                    results.append(item.text)
                else:
                    results.append(f"Unsupported content type: {item.type}")
            content_data = {**tool_args, tool_name: results, "_server": server_name}
        
        return {
            "role": "tool",
            "content": json.dumps(content_data),
            "tool_call_id": tool_call.id,
        }

    async def process_messages(
        self,
        messages: List[Dict[str, Any]],
        model: str = "deepseek-chat",
    ) -> List[Dict[str, Any]]:
        """递归处理消息和工具调用"""
        if not self.servers:
            raise RuntimeError("未连接到任何服务器")
        
        # 收集所有服务器的工具
        all_tools = []
        for server_name, server in self.servers.items():
            for tool_name, tool in server["tools"].items():
                all_tools.append(
                    ChatCompletionToolParam(
                        type="function",
                        function=FunctionDefinition(
                            name=tool.name,
                            description=tool.description if tool.description else "",
                            parameters=tool.inputSchema,
                        ),
                    )
                )
        
        last_message_role = messages[-1]["role"]
        
        if last_message_role == "user":
            # 如果最后一条是用户消息，调用LLM生成响应
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=all_tools,
                tool_choice="auto",
            )
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == "stop":
                # 普通文本响应
                messages.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                })
                return messages
            elif finish_reason == "tool_calls":
                # 工具调用响应
                tool_calls = response.choices[0].message.tool_calls
                assert tool_calls is not None
                
                # 添加助手的工具调用消息
                tool_calls_list = []
                for tool_call in tool_calls:
                    tool_calls_list.append({
                        "id": tool_call.id,
                        "function": {
                            "arguments": tool_call.function.arguments,
                            "name": tool_call.function.name,
                        },
                        "type": tool_call.type,
                    })
                
                messages.append({
                    "role": "assistant",
                    "tool_calls": tool_calls_list,
                    "content": None,
                })
                
                # 处理每个工具调用
                tasks = [asyncio.create_task(self.process_tool_call(tool_call)) for tool_call in tool_calls]
                tool_messages = await asyncio.gather(*tasks)
                messages.extend(tool_messages)
                
                # 递归处理后续消息
                return await self.process_messages(messages, model)
            else:
                raise ValueError(f"未知的完成原因: {finish_reason}")
        
        elif last_message_role == "tool":
            # 工具消息之后获取LLM的响应
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=all_tools,
                tool_choice="auto",
            )
            
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == "stop":
                # 添加助手的响应
                messages.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                })
                return messages
            elif finish_reason == "tool_calls":
                # 继续处理工具调用
                tool_calls = response.choices[0].message.tool_calls
                assert tool_calls is not None
                
                tool_calls_list = []
                for tool_call in tool_calls:
                    tool_calls_list.append({
                        "id": tool_call.id,
                        "function": {
                            "arguments": tool_call.function.arguments,
                            "name": tool_call.function.name,
                        },
                        "type": tool_call.type,
                    })
                
                messages.append({
                    "role": "assistant",
                    "tool_calls": tool_calls_list,
                    "content": None,
                })
                
                tasks = [asyncio.create_task(self.process_tool_call(tool_call)) for tool_call in tool_calls]
                tool_messages = await asyncio.gather(*tasks)
                messages.extend(tool_messages)
                
                return await self.process_messages(messages, model)
            else:
                raise ValueError(f"未知的完成原因: {finish_reason}")
        
        else:
            # 其他情况
            return messages

    async def process_query(self, query: str) -> str:
        """处理查询并返回结果文本"""
        # 创建初始消息
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        
        # 处理消息并获取完整的对话历史
        processed_messages = await self.process_messages(messages)
        
        # 从处理后的消息中提取结果
        final_text = []
        for message in processed_messages:
            if message["role"] == "assistant":
                if "content" in message and message["content"]:
                    final_text.append(message["content"])
            elif message["role"] == "tool":
                tool_data = json.loads(message["content"])
                server_name = tool_data.get("_server", "未知服务器")
                final_text.append(f"[服务器 '{server_name}' 工具结果: {json.dumps(tool_data, indent=2, ensure_ascii=False)}]")
        
        return "\n".join(final_text)

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP 多服务器客户端已启动!")
        print("输入您的问题或输入 'quit' 退出。")

        while True:
            try:
                query = input("\n问题: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\n错误: {str(e)}")
                import traceback
                print(traceback.format_exc())

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("使用方法: python client_multi_server.py <config_file_path>")
        sys.exit(1)

    client = MCPMultiServerClient()
    try:
        await client.connect_to_servers(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())