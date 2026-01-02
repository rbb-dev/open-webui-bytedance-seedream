# ByteDance Seedream Pipe (Open WebUI)

This is a single-file Open WebUI pipe that lets you generate images (text → image) and edit images (image + text → image) using ByteDance Seedream via an API gateway.

- Pipe file: `bytedance_seedream.py`
- Pipe ID: `seedream-4-5`
- What you get back: Markdown image links to files stored in Open WebUI.

## For Users (How to use it)

### 1) Select the tool
In Open WebUI, choose the tool named `ByteDance: Seedream` (ID `seedream-4-5`) for your chat/message.

### Quick help
Send `help` to the tool to get a short usage guide with examples and supported sizes.

### Prompt rewriting (and how to stop it)
This tool may slightly rewrite/clean up your prompt before generating, and it will show you the exact “Using prompt” text in the chat.

If you don’t want any rewriting, add this to your message:
`Use my prompt verbatim. Don’t embellish or expand it.`

### 2) Generate an image (text-only)
Type what you want and send it.

Good prompt examples:
- “A cinematic photo of a black cat in the rain, 85mm lens, soft bokeh, night city lights.”
- “A minimal flat icon of a rocket, white background, simple shapes.”
- “A watercolor landscape with mountains and a lake, pastel colors.”

If you want a specific size, say it explicitly:
- “Generate a poster at 4096x4096 …”
- “Make it 2048x2048 …”
- “Use 2K / 4K …”

### 3) Edit an image (upload + instructions)
Attach an image to your message and describe the edit.

Edit prompt examples:
- “Remove the watermark and make the background white.”
- “Change the car color to red and keep the same angle.”
- “Make the sky sunset orange, keep the building unchanged.”
- “Upscale to 2048x2048 and sharpen slightly.”

Notes:
- If you don’t request a resize, the pipe tries to keep the original image dimensions for edits.
- If you attach multiple images, the pipe will decide which ones are references and which one is the “base” image to edit.

### 4) What the pipe returns
You’ll get one or more generated images as Markdown image links that you can click/open in Open WebUI.

### 5) If something goes wrong
Common user-visible issues:
- “API_KEY not set in valves.” → ask your admin to configure the pipe.
- “No Task Model configured …” → ask your admin to configure Open WebUI’s Task Model (see Admin section).
- Gateway/API errors → try again later or ask your admin to check the provider status/limits.

## For Admins (What it does + how to run it)

### What this pipe does internally
- Collects the user conversation and any attached images.
- Uses Open WebUI’s global Task Model to classify intent (generate vs edit) and to extract parameters (size, watermark preference, image usage plan).
- Calls the image API endpoint for either generation or editing.
- Uploads the returned images into the Open WebUI file store.
- Streams an OpenAI-compatible response to the UI with status updates.

### Installation
1. Copy `bytedance_seedream.py` into your Open WebUI backend `pipes/` directory.
2. Restart Open WebUI.

### Required configuration
Set valves in the Open WebUI plugin UI:
- `API_KEY` (string, required): API bearer token for your image gateway/provider.
- `API_BASE_URL` (string): defaults to `https://api.cometapi.com/v1`.
- `MODEL` (string): defaults to `doubao-seedream-4-5-251128` (update this when your provider changes model identifiers).

### Task model requirement (Structured Outputs)
This pipe uses Open WebUI’s global Task Model configuration:
- `TASK_MODEL_EXTERNAL` (recommended) or `TASK_MODEL`

Your task model must support Structured Outputs (JSON Schema). If it doesn’t, the task-model step will fail and the pipe will error.

Pipe valves related to task model selection:
- `task_model_mode`: choose `external` (uses `TASK_MODEL_EXTERNAL`) or `internal` (uses `TASK_MODEL`)
- `task_model_fallback`: fallback strategy if the task model call fails (`none`, `other_task_model`, `chat_model`)

### Optional valves
- `ENABLE_LOGGING` (bool): when `True`, logs at INFO; otherwise only errors.
- `GUIDANCE_SCALE` (int): defaults to `3` (higher values follow the prompt more closely).
- `WATERMARK` (bool): default watermark preference when the task model omits the field.
- `DEFAULT_SIZE` (string): fallback size when not specified/invalid (defaults to `2048x2048`).
- `REQUEST_TIMEOUT` (int seconds): defaults to `600`.

### Supported sizes
The pipe validates and maps sizes to a known supported set:
- Exact sizes: 1024x1024, 2048x2048, 2304x1728, 1728x2304, 2560x1440, 1440x2560, 2496x1664, 1664x2496, 3024x1296, 4096x4096, 6240x2656
- Shorthands: 2K, 4K

### Troubleshooting
- Open WebUI reports no task model configured:
  - Set `TASK_MODEL_EXTERNAL` (or `TASK_MODEL`) to a model ID present in Open WebUI’s model catalog and capable of Structured Outputs.
- Gateway rejects multiple images:
  - The pipe retries with a single image automatically.
- Oversized input images:
  - Images estimated over ~10 MB after decode are skipped.
