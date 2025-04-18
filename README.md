

## 一个mcp-client的demo实现

支持vscode的mcp.json, 支持stdio、sse协议

## 项目目的

1. 学习mcp-client的实现


## 项目使用

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


## 参考文章

1. https://modelcontextprotocol.io/introduction
2. https://github.com/S1M0N38/mcp-openai/blob/main/src/mcp_openai/client.py