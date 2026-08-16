# ComfyUI-llama-cpp  
Run LLM/VLM models natively in ComfyUI based on llama.cpp  
**[[📃中文版](./README_zh.md)]** 

## Preview  
![](./img/preview.jpg)

## Installation  

#### Install the node:  
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-llama-cpp.git
python -m pip install -r ComfyUI-llama-cpp/requirements.txt
#For CUDA 13, please run:
#python -m pip install -r ComfyUI-llama-cpp/requirements_cu130.txt
```

#### Download models:  
- Place your model files in the `ComfyUI/models/LLM` folder.  

	> If you need a VLM model to process image input, don't forget to download the `mmproj` weights.

## Agent, Skills, and MCP

Use `Llama-cpp Agent Instruct` when you want the local llama.cpp model to read local skills or call MCP tools. Connect:

- `Llama-cpp Skill Library` for local `SKILL.md` folders. By default it scans this custom node's `skills/` directory.
- `Llama-cpp MCP Config` for MCP servers. It accepts `mcpServers` / `servers` JSON with stdio or Streamable HTTP entries.

Example MCP config:

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

Tool calls are automatically executed by default. Use the MCP Config node's step, timeout, and result-size limits to keep runs bounded.

## Credits  
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
