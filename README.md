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

Use the Agent nodes when you want the local llama.cpp model to read local skills or call MCP tools:

- `Llama-cpp Agent Instruct` keeps the existing one-by-one / images / video batch workflow for one workflow run.
- `Llama-cpp Agent Chat` is for real multi-turn chat. Feed its `session` output back into the next run's `session` input, or keep the same `state_uid`, to continue across ComfyUI runs. When the model needs clarification it returns `status=awaiting_user`, shows the question in `ai_question`, and continues after you fill `user_answer` and queue again.
- `Llama-cpp Live Chat` keeps one workflow run alive for a blocking, button-driven chat session. Queue once, keep sending messages from the node, and click `End` when you want the workflow to continue. It can also take the same Skill Library and MCP Config inputs as the Agent nodes.

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

Tool calls are automatically executed by default. Use the MCP Config node's step, timeout, and result-size limits to keep runs bounded. `selected_skills` only contains skills actually read in the current run, and `tool_trace` records skill/MCP activity.

## Credits  
- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng  
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai  
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
