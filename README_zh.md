# ComfyUI-llama-cpp
在 ComfyUI 中基于 llama.cpp 框架原生运行 LLM & VLM 模型。  
**[[📃English](./README.md)]**   

## 预览
![](./img/preview.jpg) 

## 安装步骤

#### 安装节点:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
#CUDA 13用户请执行:
#python -m pip install -r ComfyUI-llama-cpp/requirements_cu130.txt
```

### 模型路径:
- 请将下载的 `.gguf` 模型放置在 `ComfyUI/models/LLM` 目录中.  

	> 在使用VLM模型进行图像推理之前, 请确保已经下载并选择了主模型对应的`mmproj`权重文件.

## Agent、Skills 与 MCP

需要让本地 llama.cpp 模型读取 skill 或调用 MCP 工具时，请使用 `Llama-cpp Agent Instruct`。可连接：

- `Llama-cpp Skill Library`：读取本地 `SKILL.md` 目录，默认扫描本插件目录下的 `skills/`。
- `Llama-cpp MCP Config`：配置 MCP server，支持 `mcpServers` / `servers` JSON，包含 stdio 与 Streamable HTTP。

MCP 配置示例：

```json
{
  "mcpServers": {
    "demo": {
      "command": "python3",
      "args": ["server.py"],
      "env": {}
    },
    "remote": {
      "transport": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

工具调用默认自动执行；可在 MCP Config 节点中设置最大 agent 步数、工具超时和工具结果长度上限。

## 致谢
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
