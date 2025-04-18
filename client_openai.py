import asyncio
import json
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

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI()  # 使用OpenAI客户端
    
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "uv" if is_python else "node"
        args = [
                "--directory",
                "/Users/test/codesrc/python/weather",
                "run",
                "weather.py"
            ]
        server_params = StdioServerParameters(
            command=command,
            # args=[server_script_path],
            args=args,
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_tool_call(self, tool_call) -> Dict[str, Any]:
        """处理单个工具调用并返回工具消息"""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        # 执行工具调用
        result = await self.session.call_tool(tool_name, tool_args)
        
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
            content_data = {**tool_args, tool_name: results}
        
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
        if not self.session:
            raise RuntimeError("Not connected to any server")
        
        # 获取可用工具
        tools = [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description if tool.description else "",
                    parameters=tool.inputSchema,
                ),
            )
            for tool in (await self.session.list_tools()).tools
        ]
        
        last_message_role = messages[-1]["role"]
        
        if last_message_role == "user":
            # 如果最后一条是用户消息，调用LLM生成响应
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
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
                raise ValueError(f"Unknown finish reason: {finish_reason}")
        
        elif last_message_role == "tool":
            # 工具消息之后获取LLM的响应
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
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
                raise ValueError(f"Unknown finish reason: {finish_reason}")
        
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
                final_text.append(f"[Tool result: {json.dumps(tool_data, indent=2)}]")
        
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())