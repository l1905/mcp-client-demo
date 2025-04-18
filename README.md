

## 一个mcp-client的demo实现

支持vscode的mcp.json, 支持stdio、sse协议

## 项目目的

1. 学习mcp-client的实现


## 项目使用

项目根目录新建`.env`虚拟环境

```
OPENAI_API_KEY="替换为自己的deepseek的api-key"
OPENAI_BASE_URL="https://api.deepseek.com"
```



### client 

mcp的官方demo
```
uv run client.py
```

### client_openai 

一个简单的openai的mcp-client实现
```
uv run client_openai.py file
```

### client_openai_v2

一个简单的openai的mcp-client实现, 支持vscode stdio方式命令行交互
```
uv run client_openai_v2.py servers_config_v2.json
```


### client_openai_v3
一个简单的openai的mcp-client实现, 支持vscode stdio和sse的方式命令行交互

```
uv run client_openai_v3.py servers_config_v3.json
```

### servers_config.json

这里需要替换为自己的目录

```
{
    "servers": {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/Users/test/Downloads",
                "/Users/test/Downloads"
            ]
        },
        "fetch": {
            "command": "/Users/test/.local/bin/uvx",
            "args": [
                "--index-url",
                "https://mirrors.aliyun.com/pypi/simple/",
                "mcp-server-fetch"
            ]
        },
        "my-mcp-server-ee3c675a": {
            "type": "sse",
            "url": "http://localhost:8000/sse/"
        }
    }
}
```



## 参考文章

1. https://modelcontextprotocol.io/introduction
2. https://github.com/S1M0N38/mcp-openai/blob/main/src/mcp_openai/client.py