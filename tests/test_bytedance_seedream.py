import json
import sys
import types
from types import SimpleNamespace

import pytest


def _install_stub_modules():
    if "open_webui" in sys.modules:
        return

    open_webui = types.ModuleType("open_webui")
    sys.modules["open_webui"] = open_webui

    routers = types.ModuleType("open_webui.routers")
    sys.modules["open_webui.routers"] = routers

    files_mod = types.ModuleType("open_webui.routers.files")

    def _stub_upload_file(*args, **kwargs):  # pragma: no cover - should never run in tests
        raise RuntimeError("upload_file stub invoked")

    setattr(files_mod, "upload_file", _stub_upload_file)
    setattr(routers, "files", files_mod)
    sys.modules["open_webui.routers.files"] = files_mod
    setattr(open_webui, "routers", routers)

    models_pkg = types.ModuleType("open_webui.models")
    sys.modules["open_webui.models"] = models_pkg

    users_mod = types.ModuleType("open_webui.models.users")

    class UserModel:  # pragma: no cover - placeholder for type hints
        pass

    class Users:  # pragma: no cover - placeholder
        @staticmethod
        def get_user_by_id(_):
            return None

    setattr(users_mod, "UserModel", UserModel)
    setattr(users_mod, "Users", Users)
    setattr(models_pkg, "users", users_mod)
    sys.modules["open_webui.models.users"] = users_mod

    main_mod = types.ModuleType("open_webui.main")

    async def _stub_generate_legacy(*_, **__):  # pragma: no cover - patched in tests
        raise RuntimeError("generate_chat_completions stub invoked")

    setattr(main_mod, "generate_chat_completions", _stub_generate_legacy)
    sys.modules["open_webui.main"] = main_mod
    setattr(open_webui, "main", main_mod)

    utils_pkg = types.ModuleType("open_webui.utils")
    sys.modules["open_webui.utils"] = utils_pkg

    chat_mod = types.ModuleType("open_webui.utils.chat")

    async def _stub_generate(*_, **__):  # pragma: no cover - patched in tests
        raise RuntimeError("generate_chat_completion stub invoked")

    setattr(chat_mod, "generate_chat_completion", _stub_generate)
    sys.modules["open_webui.utils.chat"] = chat_mod
    setattr(utils_pkg, "chat", chat_mod)


_install_stub_modules()

import bytedance_seedream as seedream_pipe
from bytedance_seedream import Pipe


@pytest.fixture
def pipe_instance():
    return Pipe()


@pytest.mark.asyncio
async def test_analyse_prompt_with_task_model_returns_valid_payload(monkeypatch, pipe_instance):
    async def fake_get_user(user_id):
        return SimpleNamespace(id=user_id, settings={})

    monkeypatch.setattr(pipe_instance, "_get_user_by_id", fake_get_user)

    task_model_payload = {
        "prompt": "Resize this image to 2048x2048 and remove watermark [image:0]",
        "use_user_prompt": False,
        "intent": "edit",
        "watermark": False,
        "size": "2048x2048",
        "resize_target": {"width": 2048, "height": 2048},
        "image_plan": [
            {"index": 0, "action": "use", "role": "base"}
        ],
    }

    captured_form_data = {}

    async def fake_generate_chat_completion(*_, **__):
        form_data = __.get("form_data") or {}
        captured_form_data.update(form_data)
        return {"choices": [{"message": {"content": json.dumps(task_model_payload)}}]}

    monkeypatch.setattr(seedream_pipe, "generate_chat_completion", fake_generate_chat_completion)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings={})))
    conversation = [
        {"message_index": 0, "role": "user", "text": "resize this image", "image_refs": [0]}
    ]

    request.app.state.config = SimpleNamespace(TASK_MODEL_EXTERNAL="task-model", TASK_MODEL="")
    request.app.state.MODELS = {"task-model": {"id": "task-model"}, "chat-model": {"id": "chat-model"}}

    result = await pipe_instance._analyse_prompt_with_task_model(
        conversation=conversation,
        image_context={
            "count": 1,
            "details": [
                {
                    "index": 0,
                    "mime_type": "image/png",
                    "width": 1024,
                    "height": 1024,
                    "size_label": "1024x1024",
                }
            ],
        },
        raw_user_prompt="resize this image",
        __user__={"id": "user-1"},
        body={"model": "chat-model"},
        user_obj=None,
        __request__=request,
        emitter=None,
    )

    assert result["prompt"] == "Resize this image to 2048x2048 and remove watermark"
    assert result["use_user_prompt"] is False
    assert result["intent"] == "edit"
    assert result["use_reference_images"] is True
    assert result["size"] == "2048x2048"
    assert result["resize_target"] == {"width": 2048, "height": 2048}
    assert result["image_plan"] == [{"index": 0, "action": "use", "role": "base", "target_size": None}]
    assert captured_form_data.get("temperature") == 0
    assert "max_tokens" not in captured_form_data
    response_format = captured_form_data.get("response_format") or {}
    assert response_format.get("type") == "json_schema"
    assert response_format.get("json_schema", {}).get("strict") is True
    schema_required = response_format.get("json_schema", {}).get("schema", {}).get("required") or []
    assert "use_user_prompt" in schema_required


def test_validate_task_model_params_rejects_unsupported_size(pipe_instance):
    with pytest.raises(ValueError):
        pipe_instance._validate_task_model_params(
            {
                "prompt": "Make something",
                "use_user_prompt": False,
                "intent": "generate",
                "size": "999x999",
                "watermark": True,
                "resize_target": None,
                "image_plan": [],
            },
            image_count=0,
            fallback_prompt="fallback",
        )


def test_validate_task_model_params_strips_placeholders(pipe_instance):
    params = {
        "prompt": "Paint the dog [image:1] orange",
        "use_user_prompt": False,
        "intent": "edit",
        "size": "2048x2048",
        "watermark": False,
        "resize_target": None,
        "image_plan": [{"index": 0, "action": "use", "role": "base"}],
    }

    result = pipe_instance._validate_task_model_params(
        params,
        image_count=1,
        fallback_prompt="ignored",
    )

    assert result["prompt"] == "Paint the dog orange"
    assert result["use_user_prompt"] is False


@pytest.mark.asyncio
async def test_collect_conversation_and_images_handles_inline_data_uri(pipe_instance):
    base64_data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "Xw8AAkMBgXkGLdIAAAAASUVORK5CYII="
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Use this ![alt](data:image/png;base64,{base64_data}) for the edit",
                }
            ],
        }
    ]

    conversation, images, last_user_text = await pipe_instance._collect_conversation_and_images(messages)

    assert conversation[0]["text"].startswith("Use this")
    assert len(images) == 1
    assert images[0]["mimeType"] == "image/png"
    assert images[0]["data"] == base64_data
    assert last_user_text.startswith("Use this")
    assert images[0]["origin"] == {"type": "data_uri"}


@pytest.mark.asyncio
async def test_pipe_stream_emits_using_prompt_before_images(monkeypatch, pipe_instance):
    async def fake_get_user(user_id):
        return SimpleNamespace(id=user_id, settings={})

    monkeypatch.setattr(pipe_instance, "_get_user_by_id", fake_get_user)

    async def fake_analyse(*_, **__):
        return {
            "prompt": "A test prompt",
            "use_user_prompt": False,
            "intent": "generate",
            "watermark": False,
            "size": "2048x2048",
            "use_reference_images": False,
            "resize_target": None,
            "notes": "",
            "image_plan": [],
            "status_message": "",
        }

    monkeypatch.setattr(pipe_instance, "_analyse_prompt_with_task_model", fake_analyse)

    async def fake_upload(*_, **__):
        return "/files/fake"

    monkeypatch.setattr(pipe_instance, "_upload_image", fake_upload)

    class _StubResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"url": "https://example.test/image.png"}], "usage": {}}

        @property
        def text(self):
            return ""

    class _StubClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *_, **__):
            return _StubResponse()

    monkeypatch.setattr(seedream_pipe.httpx, "AsyncClient", lambda *_, **__: _StubClient())

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings={})))
    body = {"model": "chat-model", "stream": True, "messages": [{"role": "user", "content": "hello"}]}

    pipe_instance.valves.API_KEY = "test"

    response = await pipe_instance.pipe(
        body=body,
        __user__={"id": "user-1"},
        __request__=request,
        __event_emitter__=None,
    )

    first_chunk = None
    async for chunk in response.body_iterator:
        first_chunk = chunk
        break

    assert first_chunk is not None
    chunk_text = first_chunk.decode("utf-8") if isinstance(first_chunk, (bytes, bytearray)) else str(first_chunk)
    assert "Using prompt:" in chunk_text


@pytest.mark.asyncio
async def test_pipe_event_emitter_replace_keeps_prompt_visible(monkeypatch, pipe_instance):
    async def fake_get_user(user_id):
        return SimpleNamespace(id=user_id, settings={})

    monkeypatch.setattr(pipe_instance, "_get_user_by_id", fake_get_user)

    async def fake_analyse(*_, **__):
        return {
            "prompt": "A test prompt",
            "use_user_prompt": False,
            "intent": "generate",
            "watermark": False,
            "size": "2048x2048",
            "use_reference_images": False,
            "resize_target": None,
            "notes": "",
            "image_plan": [],
            "status_message": "",
        }

    monkeypatch.setattr(pipe_instance, "_analyse_prompt_with_task_model", fake_analyse)

    class _StubResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"url": "https://example.test/image.png"}], "usage": {}}

        @property
        def text(self):
            return ""

    class _StubClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *_, **__):
            return _StubResponse()

    monkeypatch.setattr(seedream_pipe.httpx, "AsyncClient", lambda *_, **__: _StubClient())

    events = []

    async def emitter(event_data):
        events.append(event_data)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings={})))
    body = {"model": "chat-model", "stream": True, "messages": [{"role": "user", "content": "hello"}]}

    pipe_instance.valves.API_KEY = "test"

    response = await pipe_instance.pipe(
        body=body,
        __user__={"id": "user-1"},
        __request__=request,
        __event_emitter__=emitter,
    )

    # Drain stream to completion
    async for _ in response.body_iterator:
        pass

    replace_events = [e for e in events if e.get("type") == "replace"]
    assert replace_events, "expected at least one replace event"
    assert "Using prompt:" in replace_events[0].get("data", {}).get("content", "")
    assert "image" in replace_events[-1].get("data", {}).get("content", "")


@pytest.mark.asyncio
async def test_pipe_help_short_circuits(monkeypatch, pipe_instance):
    async def fake_get_user(user_id):
        return SimpleNamespace(id=user_id, settings={})

    monkeypatch.setattr(pipe_instance, "_get_user_by_id", fake_get_user)

    called = {"task": 0, "http": 0}

    async def fake_analyse(*_, **__):
        called["task"] += 1
        raise AssertionError("task model should not be called for help")

    monkeypatch.setattr(pipe_instance, "_analyse_prompt_with_task_model", fake_analyse)

    class _StubClient:
        async def __aenter__(self):
            called["http"] += 1
            raise AssertionError("http client should not be used for help")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(seedream_pipe.httpx, "AsyncClient", lambda *_, **__: _StubClient())

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings={})))
    body = {"model": "chat-model", "stream": True, "messages": [{"role": "user", "content": "help"}]}

    response = await pipe_instance.pipe(
        body=body,
        __user__={"id": "user-1"},
        __request__=request,
        __event_emitter__=None,
    )

    first_chunk = None
    async for chunk in response.body_iterator:
        first_chunk = chunk
        break

    assert first_chunk is not None
    chunk_text = first_chunk.decode("utf-8") if isinstance(first_chunk, (bytes, bytearray)) else str(first_chunk)
    assert "Seedream help" in chunk_text
    assert called["task"] == 0
    assert called["http"] == 0
