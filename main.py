import asyncio
import base64
import contextlib
import json
import mimetypes
import os
import re
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import aiohttp
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import File, Image
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_NATURAL_LANGUAGE_POLISH_PROMPT_TEMPLATE = """你是生图提示词优化助手。

你的任务是把用户的自然语言绘图需求，整理成适合图像生成模型直接使用的提示词。

要求：
1. 必须严格保留用户本意，只做轻度润色、补全歧义和整理语序，不能擅自改变主体、动作、关系、场景、风格、时代、情绪、用途。
2. 如果用户没有明确指定画风，不要强行添加二次元、写实、电影感、赛博朋克、摄影参数、镜头语言、构图术语、画质标签。
3. 优先输出自然、清晰、适合中文图像模型理解的描述，不要堆砌关键词。
4. 如果用户需求本身已经很完整，只做轻微整理，不要过度扩写。
5. 只输出最终提示词本身，不要解释，不要分点，不要加引号，不要输出 JSON、Markdown 或代码块。
6. 默认沿用用户原始语言；如果用户明确指定了提示词语言，就按用户指定的语言输出。
"""
COMMAND_NAMES = {"画图", "draw", "image", "绘图"}
IMAGE_EDIT_COMMAND_NAMES = {"图生图", "img2img", "image2image"}
RETOUCH_COMMAND_NAMES = {"修图", "改图", "edit", "inpaint"}
ALL_COMMAND_NAMES = COMMAND_NAMES | IMAGE_EDIT_COMMAND_NAMES | RETOUCH_COMMAND_NAMES
CONFIRM_KEYWORDS = {"确认", "确定", "开始", "开始生成", "确认生成", "可以", "行", "好", "ok", "yes"}
CANCEL_KEYWORDS = {"取消", "算了", "不要了", "停", "停止", "结束"}
ORDER_PLACEHOLDER_PATTERN = re.compile(r"(?:\[\s*图片\s*\]|\[\s*image\s*\]|\[\s*img\s*\]|\s)+", re.IGNORECASE)
DEFAULT_PENDING_IMAGE_EXPIRE_SECONDS = 600
DEFAULT_PENDING_CONFIRMATION_EXPIRE_SECONDS = 600
MAX_STAGED_IMAGE_COUNT = 8
DEFAULT_ORDER_ADJUST_PROMPT = """你是图片顺序调整解析器。

你的任务是根据用户对图片顺序的自然语言要求，输出最终图片顺序。

要求：
1. 只输出 JSON，不要解释，不要代码块。
2. JSON 格式固定为 {"order":[1,2,3]}。
3. order 使用 1 开始的图片序号。
4. 如果用户只调整部分顺序，其余未提及图片按原顺序补在后面。
5. 如果用户要求不明确、无法确定、或者和现有图片数量冲突，输出 {"order":[]}。
"""


@dataclass
class StagedImageBatch:
    image_paths: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.monotonic)


@dataclass
class PendingEditConfirmation:
    mode: str
    prompt: str
    image_paths: list[str]
    created_at: float = field(default_factory=time.monotonic)


@pydantic_dataclass
class GenerateImageTool(FunctionTool[AstrAgentContext]):
    """根据用户描述生成图片，并在完成后推送到当前会话。"""

    name: str = "generate_image"
    description: str = "当用户要求你画图、生成图片、制作插画或配图时调用。图片会在生成完成后自动发送到当前会话，你可以继续按当前人格口吻回复用户。"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "将用户的画图需求尽量原样传给生图模型。除非用户明确要求，否则不要主动添加新的风格、构图、光照、背景、画质标签或其他会改变语义侧重点的描述。用户只是自然地说“帮我画一只猫”时，也应该直接提炼成生图提示词并调用本工具。",
                }
            },
            "required": ["prompt"],
        }
    )
    plugin: Any = None

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs: Any,
    ) -> ToolExecResult:
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return "请先给出要生成的图片描述。"

        plugin = self.plugin
        if not plugin:
            return "图片插件未正确初始化。"

        event = None
        if hasattr(context, "context") and isinstance(context.context, AstrAgentContext):
            event = context.context.event
        elif isinstance(context, dict):
            event = context.get("event")

        if not event:
            logger.warning("OpenAIImage tool call missing event context")
            return "当前无法获取会话上下文，暂时不能直接发送图片。"

        cleaned_prompt = plugin._cleanup_prompt(prompt)
        if not cleaned_prompt:
            return "请先给出要生成的图片描述。"

        cleaned_prompt, polished_prompt = await plugin._maybe_polish_tool_prompt(
            event=event,
            prompt=cleaned_prompt,
            source="llm_tool",
        )

        try:
            await plugin._queue_generation_task(
                event=event,
                prompt=cleaned_prompt,
                source="llm_tool",
                polished_prompt=polished_prompt,
            )
        except ValueError as exc:
            return str(exc)
        return "图像生成任务已启动，结果会在完成后自动发送到当前会话。"


@pydantic_dataclass
class EditImageTool(FunctionTool[AstrAgentContext]):
    """根据用户提供的图片进行图生图或修图，并在完成后推送到当前会话。"""

    name: str = "edit_image"
    description: str = "当用户要求修改当前消息附带的图片、给现有图片换风格、局部修图、重绘或图生图时调用。需要用户当前消息里已经带有至少一张图片。"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "将用户对现有图片的修改要求尽量原样传给图像编辑模型，不要擅自改变原意。",
                }
            },
            "required": ["prompt"],
        }
    )
    plugin: Any = None

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs: Any,
    ) -> ToolExecResult:
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return "请先说明你想怎么修改这张图片。"

        plugin = self.plugin
        if not plugin:
            return "图片插件未正确初始化。"

        event = None
        if hasattr(context, "context") and isinstance(context.context, AstrAgentContext):
            event = context.context.event
        elif isinstance(context, dict):
            event = context.get("event")

        if not event:
            logger.warning("OpenAIImage edit tool call missing event context")
            return "当前无法获取会话上下文，暂时不能直接发送图片。"

        cleaned_prompt = plugin._cleanup_prompt(prompt)
        if not cleaned_prompt:
            return "请先说明你想怎么修改这张图片。"

        cleaned_prompt, polished_prompt = await plugin._maybe_polish_tool_prompt(
            event=event,
            prompt=cleaned_prompt,
            source="llm_tool_edit",
        )

        try:
            await plugin._queue_edit_task(
                event=event,
                prompt=cleaned_prompt,
                source="llm_tool_edit",
                polished_prompt=polished_prompt,
            )
        except ValueError as exc:
            return str(exc)
        return "图像编辑任务已启动，结果会在完成后自动发送到当前会话。"


@register("OpenAIImage", "SanHans", "使用 OpenAI 兼容接口生成图片", "1.1.5")
class OpenAIImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._tool_registered = False
        self._staged_images: dict[str, StagedImageBatch] = {}
        self._pending_edit_confirmations: dict[str, PendingEditConfirmation] = {}
        max_concurrent_tasks = int(self.config.get("max_concurrent_tasks", 3) or 3)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_tasks))
        self._image_temp_dir = os.path.abspath("data/temp/openai_image")
        os.makedirs(self._image_temp_dir, exist_ok=True)

    async def initialize(self):
        if not self._tool_registered:
            self.context.add_llm_tools(GenerateImageTool(plugin=self), EditImageTool(plugin=self))
            self._tool_registered = True

    async def terminate(self):
        if self._background_tasks:
            tasks = list(self._background_tasks)
            self._background_tasks.clear()
            for task in tasks:
                task.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*tasks, return_exceptions=True)
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def _create_background_task(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _cleanup(done_task: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(done_task)
            try:
                exc = done_task.exception()
            except asyncio.CancelledError:
                self._log_detail("info", "OpenAIImage background task cancelled")
                return
            if exc is not None:
                logger.error(f"OpenAIImage background task failed: {exc}")

        task.add_done_callback(_cleanup)
        return task

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        return self._session

    def _get_generation_timeout(self) -> int:
        value = self.config.get("generation_timeout", 180)
        return max(5, int(value or 180))

    def _get_download_timeout(self) -> int:
        fallback = min(self._get_generation_timeout(), 60)
        value = self.config.get("download_timeout", fallback)
        return max(5, int(value or fallback))

    def _get_retry_count(self) -> int:
        return max(0, min(int(self.config.get("retry_count", 1) or 1), 5))

    def _get_retry_backoff_seconds(self) -> float:
        return max(0.0, float(int(self.config.get("retry_backoff_seconds", 2) or 2)))

    def _get_proxy_url(self) -> str:
        return str(self.config.get("proxy_url") or "").strip()

    def _is_natural_language_polish_enabled(self) -> bool:
        return bool(self.config.get("natural_language_polish_enabled", False))

    @staticmethod
    def _should_retry_status(status_code: int) -> bool:
        return status_code in {408, 429, 500, 502, 503, 504}

    def _build_request_kwargs(self, timeout_seconds: int) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "timeout": aiohttp.ClientTimeout(total=max(5, int(timeout_seconds))),
        }
        proxy_url = self._get_proxy_url()
        if proxy_url:
            request_kwargs["proxy"] = proxy_url
        return request_kwargs

    async def _sleep_before_retry(self, attempt: int) -> None:
        delay = self._get_retry_backoff_seconds() * max(1, attempt)
        if delay > 0:
            await asyncio.sleep(delay)

    def _get_api_key(self) -> str:
        return str(self.config.get("api_key") or os.getenv("OPENAI_API_KEY") or "").strip()

    def _is_detailed_logging_enabled(self) -> bool:
        return bool(self.config.get("detailed_logging", True))

    def _log_detail(self, level: str, message: str) -> None:
        if not self._is_detailed_logging_enabled():
            return
        log_func = getattr(logger, level, logger.info)
        log_func(message)

    def _get_pending_image_expire_seconds(self) -> int:
        return DEFAULT_PENDING_IMAGE_EXPIRE_SECONDS

    def _get_pending_confirmation_expire_seconds(self) -> int:
        return DEFAULT_PENDING_CONFIRMATION_EXPIRE_SECONDS

    @staticmethod
    def _normalize_message_text(text: str) -> str:
        return ORDER_PLACEHOLDER_PATTERN.sub(" ", str(text or "")).strip()

    @staticmethod
    def _sanitize_image_paths(image_paths: list[str]) -> list[str]:
        return [os.path.abspath(path) for path in image_paths if path and os.path.exists(path)]

    def _extract_sender_id(self, event: AstrMessageEvent) -> str:
        message_obj = getattr(event, "message_obj", None)
        sender = getattr(message_obj, "sender", None)
        for attr_name in ("user_id", "id", "uin", "qq", "uid"):
            value = getattr(sender, attr_name, None)
            if value not in (None, ""):
                return str(value)

        getter = getattr(event, "get_sender_id", None)
        if callable(getter):
            with contextlib.suppress(Exception):
                value = getter()
                if value not in (None, ""):
                    return str(value)

        for attr_name in ("sender_id", "user_id", "from_user_id"):
            value = getattr(message_obj, attr_name, None)
            if value not in (None, ""):
                return str(value)
        return ""

    def _get_user_scope_key(self, event: AstrMessageEvent) -> str:
        unified_msg_origin = str(getattr(event, "unified_msg_origin", "") or "")
        sender_id = self._extract_sender_id(event) or "unknown"
        return f"{unified_msg_origin}::{sender_id}"

    def _is_known_command_text(self, text: str) -> bool:
        normalized_text = self._normalize_message_text(text)
        if not normalized_text:
            return False
        command = normalized_text.split()[0].lstrip("/").lower()
        return command in {name.lower() for name in ALL_COMMAND_NAMES}

    def _prune_pending_state(self) -> None:
        now = time.monotonic()
        staged_expire = self._get_pending_image_expire_seconds()
        confirm_expire = self._get_pending_confirmation_expire_seconds()

        for key, batch in list(self._staged_images.items()):
            batch.image_paths = self._sanitize_image_paths(batch.image_paths)
            if not batch.image_paths or now - batch.updated_at > staged_expire:
                self._staged_images.pop(key, None)

        for key, pending in list(self._pending_edit_confirmations.items()):
            pending.image_paths = self._sanitize_image_paths(pending.image_paths)
            if not pending.image_paths or now - pending.created_at > confirm_expire:
                self._pending_edit_confirmations.pop(key, None)

    def _stage_images(self, event: AstrMessageEvent, image_paths: list[str]) -> None:
        self._prune_pending_state()
        sanitized_paths = self._sanitize_image_paths(image_paths)
        if not sanitized_paths:
            return

        key = self._get_user_scope_key(event)
        existing_batch = self._staged_images.get(key, StagedImageBatch())
        combined_paths = [*existing_batch.image_paths]
        remaining_slots = max(0, MAX_STAGED_IMAGE_COUNT - len(combined_paths))
        if remaining_slots:
            combined_paths.extend(sanitized_paths[:remaining_slots])
        existing_batch.image_paths = combined_paths
        existing_batch.updated_at = time.monotonic()
        self._staged_images[key] = existing_batch
        self._log_detail(
            "info",
            f"OpenAIImage staged {len(sanitized_paths[:remaining_slots])} image(s) for {key}: {existing_batch.image_paths}",
        )

    def _consume_staged_images(self, event: AstrMessageEvent) -> list[str]:
        self._prune_pending_state()
        key = self._get_user_scope_key(event)
        batch = self._staged_images.pop(key, None)
        if not batch:
            return []
        return self._sanitize_image_paths(batch.image_paths)

    def _clear_staged_images(self, event: AstrMessageEvent) -> None:
        self._staged_images.pop(self._get_user_scope_key(event), None)

    def _get_pending_confirmation(self, event: AstrMessageEvent) -> PendingEditConfirmation | None:
        self._prune_pending_state()
        return self._pending_edit_confirmations.get(self._get_user_scope_key(event))

    def _set_pending_confirmation(self, event: AstrMessageEvent, pending: PendingEditConfirmation) -> None:
        pending.image_paths = self._sanitize_image_paths(pending.image_paths)
        pending.created_at = time.monotonic()
        self._pending_edit_confirmations[self._get_user_scope_key(event)] = pending

    def _clear_pending_confirmation(self, event: AstrMessageEvent) -> None:
        self._pending_edit_confirmations.pop(self._get_user_scope_key(event), None)

    def _get_api_root_url(self) -> str:
        base_url = str(self.config.get("base_url") or DEFAULT_BASE_URL).strip().rstrip("/")
        if not base_url:
            base_url = DEFAULT_BASE_URL
        for suffix in ("/images/generations", "/images/edits"):
            if base_url.endswith(suffix):
                return base_url[: -len(suffix)]
        return base_url

    def _get_generation_url(self) -> str:
        return f"{self._get_api_root_url()}/images/generations"

    def _get_edits_url(self) -> str:
        return f"{self._get_api_root_url()}/images/edits"

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": str(self.config.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            "prompt": prompt,
            "n": max(1, min(int(self.config.get("image_count", 1) or 1), 4)),
        }

        option_keys = (
            "size",
            "quality",
            "background",
            "output_format",
            "moderation",
        )
        for key in option_keys:
            value = str(self.config.get(key) or "").strip()
            if value:
                payload[key] = value

        output_format = str(self.config.get("output_format") or "png").strip().lower()
        output_compression = int(self.config.get("output_compression", 90) or 90)
        if output_format in {"jpeg", "webp"}:
            payload["output_compression"] = max(0, min(output_compression, 100))

        return payload

    def _compose_prompt(self, prompt: str) -> str:
        prefix = str(self.config.get("default_prompt_prefix") or "").strip()
        prompt = prompt.strip()
        if prefix:
            return f"{prefix}\n{prompt}".strip()
        return prompt

    def _get_platform_context(self, event: AstrMessageEvent) -> dict[str, Any]:
        platform_name = "unknown"
        platform_id = ""
        try:
            platform_name = str(event.get_platform_name() or "unknown")
        except Exception:
            platform_name = str(getattr(getattr(event, "platform_meta", None), "name", "unknown"))
        try:
            platform_id = str(event.get_platform_id() or "")
        except Exception:
            platform_id = str(getattr(getattr(event, "platform_meta", None), "id", ""))

        platform_inst = self.context.get_platform_inst(platform_id) if platform_id else None
        has_wecom_webhook = bool(
            getattr(platform_inst, "msg_push_webhook_url", "")
            or getattr(platform_inst, "webhook_client", None)
        )
        is_wecom_ai_bot = platform_name == "wecom_ai_bot"
        supports_image_output = True
        image_limit_reason = ""
        allow_initial_reply = True

        if is_wecom_ai_bot:
            allow_initial_reply = False
            if not has_wecom_webhook:
                image_limit_reason = (
                    "当前企业微信智能机器人未配置消息推送 Webhook URL。"
                    "插件会避免先发送中间消息，尽量把文本和图片合并为一次最终回复；"
                    "如果仍遇到图片不回传，建议在该渠道配置 `msg_push_webhook_url`，"
                    "或改用 `wecom` 企业微信应用渠道。"
                )

        platform_context = {
            "platform_name": platform_name,
            "platform_id": platform_id,
            "platform_inst": platform_inst,
            "is_wecom_ai_bot": is_wecom_ai_bot,
            "has_wecom_webhook": has_wecom_webhook,
            "supports_image_output": supports_image_output,
            "image_limit_reason": image_limit_reason,
            "allow_initial_reply": allow_initial_reply,
        }
        self._log_detail("info", f"OpenAIImage platform context: {platform_context}")
        if image_limit_reason:
            logger.warning(f"OpenAIImage platform advisory: {image_limit_reason}")
        return platform_context

    @staticmethod
    def _cleanup_prompt(prompt: str) -> str:
        return prompt.strip().strip("：:，,。.!！？? ")

    def _extract_command_prompt(self, event: AstrMessageEvent, raw_prompt: str) -> str:
        full = str(event.message_str or "").strip()
        if not full:
            return self._cleanup_prompt(raw_prompt)

        tokens = full.split()
        if tokens:
            command = tokens[0].lstrip("/").lower()
            if command in {name.lower() for name in ALL_COMMAND_NAMES}:
                tokens = tokens[1:]

        reconstructed = " ".join(tokens).strip()
        return self._cleanup_prompt(reconstructed or raw_prompt)

    @staticmethod
    def _is_supported_image_file(file_path: str) -> bool:
        if not file_path:
            return False
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith("image/"):
            return True
        extension = os.path.splitext(file_path)[1].lower()
        return extension in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    @staticmethod
    def _guess_mime_type_from_path(file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

    @staticmethod
    def _iter_message_components(event: AstrMessageEvent) -> list[Any]:
        message_obj = getattr(event, "message_obj", None)
        if not message_obj:
            return []

        for attr_name in ("message", "chain", "messages"):
            components = getattr(message_obj, attr_name, None)
            if isinstance(components, list):
                return components
        return []

    async def _extract_input_images(self, event: AstrMessageEvent) -> list[str]:
        image_paths: list[str] = []

        async def _collect_from_components(components: list[Any]) -> None:
            for component in components:
                if isinstance(component, Image):
                    file_path = await component.convert_to_file_path()
                    if file_path and os.path.exists(file_path):
                        image_paths.append(os.path.abspath(file_path))
                    continue

                if isinstance(component, File):
                    file_path = await component.get_file()
                    if file_path and os.path.exists(file_path) and self._is_supported_image_file(file_path):
                        image_paths.append(os.path.abspath(file_path))
                    continue

                reply_chain = getattr(component, "chain", None)
                if isinstance(reply_chain, list) and reply_chain:
                    await _collect_from_components(reply_chain)

        await _collect_from_components(self._iter_message_components(event))
        self._log_detail(
            "info",
            f"OpenAIImage extracted {len(image_paths)} input image(s) from current event: {image_paths}",
        )
        return image_paths

    async def _resolve_edit_input_images(self, event: AstrMessageEvent) -> list[str]:
        current_images = await self._extract_input_images(event)
        if current_images:
            self._clear_staged_images(event)
            return self._sanitize_image_paths(current_images)
        return self._consume_staged_images(event)

    @staticmethod
    def _get_edit_mode_display_name(mode: str) -> str:
        return "图生图" if mode == "img2img" else "修图"

    def _build_image_role_label(self, mode: str, index: int) -> str:
        if mode == "img2img":
            return "主图" if index == 1 else f"参考图 {index - 1}"
        if index == 1:
            return "原图"
        if index == 2:
            return "遮罩图"
        return f"待选图 {index - 2}"

    def _build_confirmation_chain(
        self,
        mode: str,
        prompt: str,
        image_paths: list[str],
    ) -> MessageChain:
        chain = MessageChain()
        lines = [
            f"{self._get_edit_mode_display_name(mode)}要求：{prompt}",
            f"已收到 {len(image_paths)} 张图",
        ]
        if mode == "img2img":
            lines.append("第 1 张会当主图  后续图片会当参考图")
        else:
            lines.append("修图只会上传前 2 张  第 1 张是原图  第 2 张是遮罩图")
        lines.extend(
            [
                "回复 确认 开始处理",
                "回复 取消 放弃本次任务",
                "顺序不对可以直接说  把第二张放第一张前面",
            ]
        )
        chain.message("\n".join(lines))

        for index, image_path in enumerate(image_paths, start=1):
            chain.message(f"{index}  {self._build_image_role_label(mode, index)}")
            chain.file_image(image_path)
        return chain

    async def _send_confirmation_preview(
        self,
        event: AstrMessageEvent,
        pending: PendingEditConfirmation,
        source: str,
    ) -> None:
        chain = self._build_confirmation_chain(
            mode=pending.mode,
            prompt=pending.prompt,
            image_paths=pending.image_paths,
        )
        await self._send_message_chain(
            unified_msg_origin=event.unified_msg_origin,
            chain=chain,
            source=source,
            stage="confirm_preview",
            text_count=1 + len(pending.image_paths),
            image_count=len(pending.image_paths),
        )

    async def _queue_edit_confirmation(
        self,
        event: AstrMessageEvent,
        prompt: str,
        mode: str,
        source: str,
    ) -> None:
        final_prompt = self._compose_prompt(prompt)
        if not final_prompt:
            if mode == "img2img":
                raise ValueError("请输入修改要求，例如：/图生图 把第一张里的贝壳换成后面参考图里的贝壳")
            raise ValueError("请输入修图要求，例如：/修图 去掉背景里的路人")

        image_paths = await self._resolve_edit_input_images(event)
        if not image_paths:
            if mode == "img2img":
                raise ValueError("请先连续发送图片，再发 /图生图 修改要求。第一张默认是主图，后续图片默认是参考图。")
            raise ValueError("请先发送原图，再发 /修图 修改要求。第二张会被当作遮罩图。")

        pending = PendingEditConfirmation(
            mode=mode,
            prompt=final_prompt,
            image_paths=image_paths,
        )
        self._set_pending_confirmation(event, pending)
        await self._send_confirmation_preview(event, pending, source=source)

    @staticmethod
    def _compact_message_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).strip().lower()

    def _is_confirm_reply(self, text: str) -> bool:
        return self._compact_message_text(text) in CONFIRM_KEYWORDS

    def _is_cancel_reply(self, text: str) -> bool:
        return self._compact_message_text(text) in CANCEL_KEYWORDS

    @staticmethod
    def _parse_position_token(token: str) -> int | None:
        cleaned = str(token or "").strip()
        if not cleaned:
            return None
        if cleaned.isdigit():
            value = int(cleaned)
            return value if value > 0 else None

        mapping = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        if cleaned == "十":
            return 10
        if cleaned in mapping:
            return mapping[cleaned]
        if cleaned.startswith("十") and len(cleaned) == 2 and cleaned[1] in mapping:
            return 10 + mapping[cleaned[1]]
        if cleaned.endswith("十") and len(cleaned) == 2 and cleaned[0] in mapping:
            return mapping[cleaned[0]] * 10
        if len(cleaned) == 3 and cleaned[1] == "十" and cleaned[0] in mapping and cleaned[2] in mapping:
            return mapping[cleaned[0]] * 10 + mapping[cleaned[2]]
        return None

    def _extract_position_numbers(self, text: str) -> list[int]:
        matches = re.findall(r"第\s*([0-9一二两三四五六七八九十]+)\s*(?:张|个|幅|号)?", text or "")
        numbers: list[int] = []
        for match in matches:
            value = self._parse_position_token(match)
            if value is not None:
                numbers.append(value)
        return numbers

    @staticmethod
    def _move_order_item(order: list[int], item: int, target_index: int) -> list[int]:
        new_order = [value for value in order if value != item]
        target_index = max(0, min(target_index, len(new_order)))
        new_order.insert(target_index, item)
        return new_order

    def _parse_order_adjustment_simple(
        self,
        instruction: str,
        image_count: int,
        mode: str,
    ) -> list[int] | None:
        if image_count <= 1:
            return list(range(1, image_count + 1))

        text = self._cleanup_prompt(instruction)
        if not text:
            return None

        order = list(range(1, image_count + 1))
        positions = [value for value in self._extract_position_numbers(text) if 1 <= value <= image_count]
        unique_positions = list(dict.fromkeys(positions))

        if ("交换" in text or "互换" in text or "对调" in text) and len(unique_positions) >= 2:
            first, second = unique_positions[:2]
            first_index = order.index(first)
            second_index = order.index(second)
            order[first_index], order[second_index] = order[second_index], order[first_index]
            return order

        before_match = re.search(
            r"第\s*([0-9一二两三四五六七八九十]+)\s*(?:张|个|幅|号)?.*?"
            r"第\s*([0-9一二两三四五六七八九十]+)\s*(?:张|个|幅|号)?.*?前面",
            text,
        )
        if before_match:
            source = self._parse_position_token(before_match.group(1))
            target = self._parse_position_token(before_match.group(2))
            if source and target and source != target and source in order and target in order:
                return self._move_order_item(order, source, order.index(target))

        after_match = re.search(
            r"第\s*([0-9一二两三四五六七八九十]+)\s*(?:张|个|幅|号)?.*?"
            r"第\s*([0-9一二两三四五六七八九十]+)\s*(?:张|个|幅|号)?.*?后面",
            text,
        )
        if after_match:
            source = self._parse_position_token(after_match.group(1))
            target = self._parse_position_token(after_match.group(2))
            if source and target and source != target and source in order and target in order:
                return self._move_order_item(order, source, order.index(target) + 1)

        if unique_positions:
            first_position = unique_positions[0]
            if any(keyword in text for keyword in ("主图", "原图", "第一张", "最前面", "排第一", "放前面", "当前面")):
                return self._move_order_item(order, first_position, 0)
            if "遮罩" in text and mode == "edit":
                return self._move_order_item(order, first_position, 1)
            if "最后" in text:
                return self._move_order_item(order, first_position, len(order) - 1)
            if any(keyword in text for keyword in ("第二张", "排第二", "放第二")):
                return self._move_order_item(order, first_position, 1)

        explicit_match = re.search(r"(?:顺序|改成|调整为|排成|排列为)\s*[:：]?\s*(.*)", text)
        if explicit_match:
            explicit_positions = [
                value for value in self._extract_position_numbers(explicit_match.group(1))
                if 1 <= value <= image_count
            ]
            explicit_unique = list(dict.fromkeys(explicit_positions))
            if explicit_unique:
                for index in range(1, image_count + 1):
                    if index not in explicit_unique:
                        explicit_unique.append(index)
                return explicit_unique

        return None

    async def _parse_order_adjustment_with_llm(
        self,
        event: AstrMessageEvent,
        instruction: str,
        image_count: int,
        mode: str,
    ) -> list[int] | None:
        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if not provider:
            return None

        user_prompt = (
            f"当前任务：{self._get_edit_mode_display_name(mode)}\n"
            f"当前图片数量：{image_count}\n"
            f"当前顺序：{list(range(1, image_count + 1))}\n"
            f"用户要求：{instruction}"
        )
        try:
            llm_resp = await provider.text_chat(
                prompt=user_prompt,
                system_prompt=DEFAULT_ORDER_ADJUST_PROMPT,
            )
            data = self._parse_json_response_text(llm_resp.completion_text or "")
            raw_order = data.get("order")
            if not isinstance(raw_order, list):
                return None

            parsed_order: list[int] = []
            for item in raw_order:
                try:
                    value = int(item)
                except (TypeError, ValueError):
                    continue
                if 1 <= value <= image_count and value not in parsed_order:
                    parsed_order.append(value)

            if not parsed_order:
                return None

            for value in range(1, image_count + 1):
                if value not in parsed_order:
                    parsed_order.append(value)
            return parsed_order
        except Exception as exc:
            self._log_detail("warning", f"OpenAIImage failed to parse order adjustment with llm: {exc}")
            return None

    async def _resolve_adjusted_image_paths(
        self,
        event: AstrMessageEvent,
        pending: PendingEditConfirmation,
        instruction: str,
    ) -> list[str] | None:
        image_paths = self._sanitize_image_paths(pending.image_paths)
        if not image_paths:
            return None

        image_count = len(image_paths)
        order = self._parse_order_adjustment_simple(
            instruction=instruction,
            image_count=image_count,
            mode=pending.mode,
        )
        if not order:
            order = await self._parse_order_adjustment_with_llm(
                event=event,
                instruction=instruction,
                image_count=image_count,
                mode=pending.mode,
            )
        if not order:
            return None
        return [image_paths[index - 1] for index in order if 1 <= index <= image_count]

    async def _start_confirmed_edit_task(
        self,
        event: AstrMessageEvent,
        pending: PendingEditConfirmation,
    ) -> None:
        image_paths = self._sanitize_image_paths(pending.image_paths)
        if not image_paths:
            raise ValueError("待处理图片已失效，请重新发送图片后再试。")

        if pending.mode == "img2img":
            request_image_paths = image_paths
            mask_path = None
            source = "img2img_confirmed"
        else:
            request_image_paths = [image_paths[0]]
            mask_path = image_paths[1] if len(image_paths) > 1 else None
            source = "edit_confirmed"

        platform_context = self._get_platform_context(event)
        if platform_context["allow_initial_reply"]:
            chain = MessageChain().message("已确认  开始处理")
            await self._send_message_chain(
                unified_msg_origin=event.unified_msg_origin,
                chain=chain,
                source=source,
                stage="confirm_accepted",
                text_count=1,
                image_count=0,
            )

        self._create_background_task(
            self._run_edit_task(
                unified_msg_origin=event.unified_msg_origin,
                prompt=pending.prompt,
                image_paths=request_image_paths,
                mask_path=mask_path,
                source=source,
                platform_context=platform_context,
            )
        )

    @staticmethod
    def _normalize_error_message(message: str) -> str:
        text = str(message or "").strip()
        if not text:
            return "未知错误"

        if "<html" in text.lower():
            if "504" in text:
                return "上游图像接口超时（504 Gateway Time-out）"
            if "502" in text:
                return "上游图像接口网关错误（502 Bad Gateway）"
            if "503" in text:
                return "上游图像接口暂时不可用（503 Service Unavailable）"
            title_match = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
            if title_match:
                return re.sub(r"\s+", " ", title_match.group(1)).strip()
            return "上游图像接口返回了 HTML 错误页"

        status_match = re.search(r"请求失败（(\d+)）[:：]?\s*(.*)", text, flags=re.DOTALL)
        if status_match:
            status_code = status_match.group(1)
            detail = status_match.group(2).strip()
            if status_code == "504":
                return "上游图像接口超时（504）"
            if status_code == "503":
                return f"上游图像接口暂时不可用（503）{f'：{detail}' if detail else ''}"
            if status_code == "502":
                return "上游图像接口网关错误（502）"
            return f"请求失败（{status_code}）{f'：{detail}' if detail else ''}"

        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _parse_json_response_text(text: str) -> dict[str, Any]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"raw_text": text}
        if isinstance(data, dict):
            return data
        return {"raw_data": data}

    @staticmethod
    def _sanitize_polished_prompt(text: str) -> str:
        prompt = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE).strip()
        prompt = re.sub(r"^(prompt|提示词)\s*[:：]\s*", "", prompt, flags=re.IGNORECASE).strip()
        if prompt.startswith(("```", '"""', "'''")):
            prompt = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", prompt).strip()
            prompt = re.sub(r"\n?```$", "", prompt).strip()
            prompt = prompt.strip('"\'')
        return prompt.strip().strip('"\'')

    def _get_natural_language_polish_prompt(self) -> str:
        return str(
            self.config.get("natural_language_polish_prompt_template")
            or DEFAULT_NATURAL_LANGUAGE_POLISH_PROMPT_TEMPLATE
        ).strip()

    async def _maybe_polish_tool_prompt(
        self,
        event: AstrMessageEvent,
        prompt: str,
        source: str,
    ) -> tuple[str, str | None]:
        cleaned_prompt = self._cleanup_prompt(prompt)
        if not cleaned_prompt or not self._is_natural_language_polish_enabled():
            return cleaned_prompt, None

        platform_context = self._get_platform_context(event)
        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if not provider:
            self._log_detail(
                "warning",
                f"OpenAIImage natural language polish skipped source={source}: no provider found; fallback to raw prompt",
            )
            return cleaned_prompt, None

        system_prompt = self._get_natural_language_polish_prompt()
        user_prompt = cleaned_prompt

        try:
            meta = provider.meta()
            provider_id = getattr(meta, "id", "unknown")
            provider_model = getattr(meta, "model", "unknown")
        except Exception:
            provider_id = "unknown"
            provider_model = "unknown"

        self._log_detail(
            "info",
            f"OpenAIImage natural language polish request source={source} provider={provider_id} "
            f"model={provider_model} platform={platform_context['platform_name']} original_prompt={cleaned_prompt}",
        )

        try:
            started_at = time.monotonic()
            llm_resp = await provider.text_chat(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
            duration = time.monotonic() - started_at
            raw_output = llm_resp.completion_text or ""
            self._log_detail(
                "info",
                f"OpenAIImage natural language polish response source={source} took={duration:.2f}s raw={raw_output}",
            )
            polished_prompt = self._sanitize_polished_prompt(raw_output)
            if polished_prompt:
                logger.info(
                    f"OpenAIImage polished natural prompt source={source} original={cleaned_prompt} polished={polished_prompt}"
                )
                return polished_prompt, polished_prompt
        except Exception as exc:
            error_text = str(exc)
            if "auth_unavailable" in error_text:
                logger.warning(
                    "OpenAIImage failed to polish natural prompt because the current AstrBot text provider auth is unavailable; "
                    f"fallback to raw prompt: {error_text}"
                )
            else:
                logger.warning(f"OpenAIImage failed to polish natural prompt, fallback to raw prompt: {error_text}")

        return cleaned_prompt, None

    async def _request_image_generation(self, prompt: str) -> dict[str, Any]:
        api_key = self._get_api_key()
        endpoint = self._get_generation_url()
        if not api_key:
            raise ValueError("未配置 API Key。请在插件配置中填写 api_key，或设置环境变量 OPENAI_API_KEY。")
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("base_url 必须以 http:// 或 https:// 开头，并建议填写到 /v1。")

        payload = self._build_payload(prompt)
        session = await self._ensure_session()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        generation_timeout = self._get_generation_timeout()
        retry_count = self._get_retry_count()
        total_attempts = retry_count + 1
        proxy_url = self._get_proxy_url()
        self._log_detail(
            "info",
            f"OpenAIImage request endpoint={endpoint} model={payload.get('model')} size={payload.get('size')} "
            f"quality={payload.get('quality')} n={payload.get('n')} moderation={payload.get('moderation')} "
            f"timeout={generation_timeout}s retries={retry_count} proxy={proxy_url or 'none'} prompt={prompt}",
        )

        for attempt in range(1, total_attempts + 1):
            started_at = time.monotonic()
            try:
                async with session.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    **self._build_request_kwargs(generation_timeout),
                ) as response:
                    text = await response.text()
                    data: dict[str, Any]
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        data = {"raw_text": text}
                    duration = time.monotonic() - started_at
                    data_count = len(data.get("data", [])) if isinstance(data.get("data"), list) else 0
                    self._log_detail(
                        "info",
                        f"OpenAIImage response status={response.status} took={duration:.2f}s "
                        f"attempt={attempt}/{total_attempts} data_count={data_count} "
                        f"keys={list(data.keys()) if isinstance(data, dict) else type(data)}",
                    )

                    if response.status >= 400:
                        error_message = (
                            data.get("error", {}).get("message")
                            if isinstance(data.get("error"), dict)
                            else data.get("error")
                        ) or data.get("message") or text
                        if attempt < total_attempts and self._should_retry_status(response.status):
                            logger.warning(
                                f"OpenAIImage generation request failed with retryable status={response.status} "
                                f"attempt={attempt}/{total_attempts} retrying: {self._normalize_error_message(error_message)}"
                            )
                            await self._sleep_before_retry(attempt)
                            continue
                        raise RuntimeError(f"请求失败（{response.status}）：{error_message}")

                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                duration = time.monotonic() - started_at
                if attempt < total_attempts:
                    logger.warning(
                        f"OpenAIImage generation request exception attempt={attempt}/{total_attempts} "
                        f"took={duration:.2f}s retrying: {exc}"
                    )
                    await self._sleep_before_retry(attempt)
                    continue
                raise

        raise RuntimeError("请求失败：超过最大重试次数")

    async def _request_image_edit(
        self,
        prompt: str,
        image_paths: list[str],
        mask_path: str | None = None,
    ) -> dict[str, Any]:
        api_key = self._get_api_key()
        endpoint = self._get_edits_url()
        if not api_key:
            raise ValueError("未配置 API Key。请在插件配置中填写 api_key，或设置环境变量 OPENAI_API_KEY。")
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("base_url 必须以 http:// 或 https:// 开头，并建议填写到 /v1。")
        if not image_paths:
            raise ValueError("未找到要编辑的原图文件。")
        missing_images = [image_path for image_path in image_paths if not os.path.exists(image_path)]
        if missing_images:
            raise ValueError(f"未找到要编辑的原图文件：{missing_images[0]}")
        if mask_path and not os.path.exists(mask_path):
            raise ValueError("未找到修图遮罩文件。")

        payload = self._build_payload(prompt)
        session = await self._ensure_session()
        headers = {"Authorization": f"Bearer {api_key}"}
        generation_timeout = self._get_generation_timeout()
        retry_count = self._get_retry_count()
        total_attempts = retry_count + 1
        proxy_url = self._get_proxy_url()
        self._log_detail(
            "info",
            f"OpenAIImage edit request endpoint={endpoint} model={payload.get('model')} size={payload.get('size')} "
            f"quality={payload.get('quality')} n={payload.get('n')} moderation={payload.get('moderation')} "
            f"timeout={generation_timeout}s retries={retry_count} proxy={proxy_url or 'none'} "
            f"image_count={len(image_paths)} images={image_paths} mask={mask_path or 'none'} prompt={prompt}",
        )

        for attempt in range(1, total_attempts + 1):
            started_at = time.monotonic()
            form = aiohttp.FormData()
            for key, value in payload.items():
                form.add_field(key, str(value))

            try:
                image_field_name = "image[]" if len(image_paths) > 1 else "image"
                with contextlib.ExitStack() as stack:
                    for image_path in image_paths:
                        image_file = stack.enter_context(open(image_path, "rb"))
                        form.add_field(
                            image_field_name,
                            image_file,
                            filename=os.path.basename(image_path),
                            content_type=self._guess_mime_type_from_path(image_path),
                        )
                    if mask_path:
                        mask_file = stack.enter_context(open(mask_path, "rb"))
                        form.add_field(
                            "mask",
                            mask_file,
                            filename=os.path.basename(mask_path),
                            content_type=self._guess_mime_type_from_path(mask_path),
                        )
                    async with session.post(
                            endpoint,
                            headers=headers,
                            data=form,
                            **self._build_request_kwargs(generation_timeout),
                        ) as response:
                            text = await response.text()
                            data = self._parse_json_response_text(text)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                duration = time.monotonic() - started_at
                if attempt < total_attempts:
                    logger.warning(
                        f"OpenAIImage edit request exception attempt={attempt}/{total_attempts} "
                        f"took={duration:.2f}s retrying: {exc}"
                    )
                    await self._sleep_before_retry(attempt)
                    continue
                raise

            duration = time.monotonic() - started_at
            data_count = len(data.get("data", [])) if isinstance(data.get("data"), list) else 0
            status = response.status
            self._log_detail(
                "info",
                f"OpenAIImage edit response status={status} took={duration:.2f}s "
                f"attempt={attempt}/{total_attempts} data_count={data_count} "
                f"keys={list(data.keys()) if isinstance(data, dict) else type(data)}",
            )

            if status >= 400:
                error_message = (
                    data.get("error", {}).get("message")
                    if isinstance(data.get("error"), dict)
                    else data.get("error")
                ) or data.get("message") or text
                if attempt < total_attempts and self._should_retry_status(status):
                    logger.warning(
                        f"OpenAIImage edit request failed with retryable status={status} "
                        f"attempt={attempt}/{total_attempts} retrying: {self._normalize_error_message(error_message)}"
                    )
                    await self._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(f"请求失败（{status}）：{error_message}")

            return data

        raise RuntimeError("请求失败：超过最大重试次数")

    def _save_base64_image(self, image_base64: str) -> str:
        output_format = str(self.config.get("output_format") or "png").strip().lower()
        extension = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp"}.get(output_format, "png")
        return self._save_image_bytes(base64.b64decode(image_base64), extension)

    def _save_image_bytes(self, image_bytes: bytes, extension: str) -> str:
        extension = extension.strip(".").lower() or "png"
        filename = f"openai_image_{uuid.uuid4().hex}.{extension}"
        file_path = os.path.join(self._image_temp_dir, filename)
        with open(file_path, "wb") as file:
            file.write(image_bytes)
        self._log_detail("info", f"OpenAIImage saved image to {file_path}")
        return file_path

    @staticmethod
    def _guess_extension_from_response(image_url: str, content_type: str) -> str:
        mime = (content_type or "").split(";", 1)[0].strip().lower()
        extension = mimetypes.guess_extension(mime) or ""
        if not extension:
            extension = os.path.splitext(urlparse(image_url).path)[1]
        extension = extension.lower()
        if extension in {".jpg", ".jpeg"}:
            return "jpg"
        if extension in {".png", ".webp", ".gif"}:
            return extension.lstrip(".")
        return "png"

    async def _download_remote_image(self, image_url: str) -> str:
        session = await self._ensure_session()
        download_timeout = self._get_download_timeout()
        retry_count = self._get_retry_count()
        total_attempts = retry_count + 1
        proxy_url = self._get_proxy_url()
        self._log_detail(
            "info",
            f"OpenAIImage downloading remote image url={image_url} timeout={download_timeout}s "
            f"retries={retry_count} proxy={proxy_url or 'none'}",
        )
        for attempt in range(1, total_attempts + 1):
            started_at = time.monotonic()
            try:
                async with session.get(
                    image_url,
                    **self._build_request_kwargs(download_timeout),
                ) as response:
                    if response.status >= 400:
                        if attempt < total_attempts and self._should_retry_status(response.status):
                            logger.warning(
                                f"OpenAIImage remote image download failed with retryable status={response.status} "
                                f"attempt={attempt}/{total_attempts} retrying url={image_url}"
                            )
                            await self._sleep_before_retry(attempt)
                            continue
                        raise RuntimeError(f"下载图片失败（{response.status}）：{image_url}")
                    image_bytes = await response.read()
                    extension = self._guess_extension_from_response(
                        image_url=image_url,
                        content_type=response.headers.get("Content-Type", ""),
                    )
                    duration = time.monotonic() - started_at
                    self._log_detail(
                        "info",
                        f"OpenAIImage downloaded remote image url={image_url} took={duration:.2f}s "
                        f"attempt={attempt}/{total_attempts} bytes={len(image_bytes)}",
                    )
                    return self._save_image_bytes(image_bytes, extension)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                duration = time.monotonic() - started_at
                if attempt < total_attempts:
                    logger.warning(
                        f"OpenAIImage remote image download exception attempt={attempt}/{total_attempts} "
                        f"took={duration:.2f}s retrying url={image_url}: {exc}"
                    )
                    await self._sleep_before_retry(attempt)
                    continue
                raise

        raise RuntimeError(f"下载图片失败：超过最大重试次数：{image_url}")

    async def _collect_image_files(self, data: dict[str, Any]) -> list[str]:
        image_files: list[str] = []
        images = data.get("data", [])
        if not isinstance(images, list):
            raise RuntimeError("接口返回格式异常：缺少 data 数组。")

        for item in images:
            if not isinstance(item, dict):
                continue

            b64_json = item.get("b64_json")
            image_url = item.get("url")

            if isinstance(b64_json, str) and b64_json.strip():
                image_files.append(self._save_base64_image(b64_json.strip()))
                continue

            if isinstance(image_url, str) and image_url.strip():
                image_files.append(await self._download_remote_image(image_url.strip()))

        if not image_files:
            raise RuntimeError("接口未返回可发送的图片数据。")
        self._log_detail("info", f"OpenAIImage prepared {len(image_files)} image file(s) for sending")
        return image_files

    async def _send_success_response(
        self,
        unified_msg_origin: str,
        response: dict[str, Any],
        source: str,
        final_preamble_texts: list[str] | None = None,
        polished_prompt: str | None = None,
    ) -> None:
        preamble_texts = list(final_preamble_texts or [])
        if (
            self.config.get("send_prompt_back", False)
            and self._is_natural_language_polish_enabled()
            and polished_prompt
        ):
            preamble_texts.append(f"最终提示词：{polished_prompt}")
        image_files = await self._collect_image_files(response)
        chain = self._build_message_chain(image_files, preamble_texts=preamble_texts)
        await self._send_message_chain(
            unified_msg_origin=unified_msg_origin,
            chain=chain,
            source=source,
            stage="final",
            text_count=len([text for text in preamble_texts if text]),
            image_count=len(image_files),
        )

    def _build_message_chain(
        self,
        image_files: list[str],
        preamble_texts: list[str] | None = None,
    ) -> MessageChain:
        chain = MessageChain()
        for text in preamble_texts or []:
            if text:
                chain.message(text)
        for file_path in image_files:
            chain.file_image(file_path)
        return chain

    async def _send_message_chain(
        self,
        unified_msg_origin: str,
        chain: MessageChain,
        source: str,
        stage: str,
        text_count: int,
        image_count: int,
    ) -> None:
        self._log_detail(
            "info",
            f"OpenAIImage sending active message source={source} stage={stage} "
            f"umo={unified_msg_origin} texts={text_count} images={image_count}",
        )
        await self.context.send_message(unified_msg_origin, chain)
        self._log_detail(
            "info",
            f"OpenAIImage active message sent source={source} stage={stage} "
            f"umo={unified_msg_origin} texts={text_count} images={image_count}",
        )

    async def _send_error_message(self, unified_msg_origin: str, text: str, source: str) -> None:
        chain = MessageChain().message(text)
        await self._send_message_chain(
            unified_msg_origin=unified_msg_origin,
            chain=chain,
            source=source,
            stage="error",
            text_count=1,
            image_count=0,
        )

    async def _run_generation_task(
        self,
        unified_msg_origin: str,
        prompt: str,
        source: str,
        platform_context: dict[str, Any],
        final_preamble_texts: list[str] | None = None,
        polished_prompt: str | None = None,
    ) -> None:
        async with self._semaphore:
            try:
                logger.info(
                    f"OpenAIImage request source={source} platform={platform_context['platform_name']} "
                    f"platform_id={platform_context['platform_id']} prompt={prompt}"
                )
                response = await self._request_image_generation(prompt)
                await self._send_success_response(
                    unified_msg_origin=unified_msg_origin,
                    response=response,
                    source=source,
                    final_preamble_texts=final_preamble_texts,
                    polished_prompt=polished_prompt,
                )
            except ValueError as exc:
                logger.warning(f"OpenAIImage invalid config: {exc}")
                await self._send_error_message(
                    unified_msg_origin,
                    f"❌ 生成失败: {self._normalize_error_message(exc)}",
                    source,
                )
            except aiohttp.ClientError as exc:
                logger.error(f"OpenAIImage network error: {exc}")
                await self._send_error_message(
                    unified_msg_origin,
                    f"❌ 生成失败: 网络请求失败：{self._normalize_error_message(exc)}",
                    source,
                )
            except asyncio.TimeoutError:
                logger.error("OpenAIImage request timed out")
                await self._send_error_message(
                    unified_msg_origin,
                    "❌ 生成失败: 请求超时，请稍后重试。",
                    source,
                )
            except Exception as exc:
                logger.exception(f"OpenAIImage generation failed: {exc}")
                await self._send_error_message(
                    unified_msg_origin,
                    f"❌ 生成失败: {self._normalize_error_message(exc)}",
                    source,
                )

    async def _run_edit_task(
        self,
        unified_msg_origin: str,
        prompt: str,
        image_paths: list[str],
        mask_path: str | None,
        source: str,
        platform_context: dict[str, Any],
        final_preamble_texts: list[str] | None = None,
        polished_prompt: str | None = None,
    ) -> None:
        async with self._semaphore:
            try:
                logger.info(
                    f"OpenAIImage edit request source={source} platform={platform_context['platform_name']} "
                    f"platform_id={platform_context['platform_id']} image_count={len(image_paths)} "
                    f"images={image_paths} mask={mask_path or 'none'} "
                    f"prompt={prompt}"
                )
                response = await self._request_image_edit(prompt, image_paths=image_paths, mask_path=mask_path)
                await self._send_success_response(
                    unified_msg_origin=unified_msg_origin,
                    response=response,
                    source=source,
                    final_preamble_texts=final_preamble_texts,
                    polished_prompt=polished_prompt,
                )
            except ValueError as exc:
                logger.warning(f"OpenAIImage invalid edit input/config: {exc}")
                await self._send_error_message(
                    unified_msg_origin,
                    f"❌ 修图失败: {self._normalize_error_message(exc)}",
                    source,
                )
            except aiohttp.ClientError as exc:
                logger.error(f"OpenAIImage edit network error: {exc}")
                await self._send_error_message(
                    unified_msg_origin,
                    f"❌ 修图失败: 网络请求失败：{self._normalize_error_message(exc)}",
                    source,
                )
            except asyncio.TimeoutError:
                logger.error("OpenAIImage edit request timed out")
                await self._send_error_message(
                    unified_msg_origin,
                    "❌ 修图失败: 请求超时，请稍后重试。",
                    source,
                )
            except Exception as exc:
                logger.exception(f"OpenAIImage edit failed: {exc}")
                await self._send_error_message(
                    unified_msg_origin,
                    f"❌ 修图失败: {self._normalize_error_message(exc)}",
                    source,
                )

    async def _queue_generation_task(
        self,
        event: AstrMessageEvent,
        prompt: str,
        source: str,
        initial_reply_text: str | None = None,
        final_preamble_texts: list[str] | None = None,
        polished_prompt: str | None = None,
    ) -> str | None:
        final_prompt = self._compose_prompt(prompt)
        if not final_prompt:
            raise ValueError("请输入要绘制的内容，例如：/画图 一只戴墨镜的赛博朋克橘猫")

        platform_context = self._get_platform_context(event)
        send_initial_reply = bool(initial_reply_text and platform_context["allow_initial_reply"])
        effective_final_preamble_texts = list(final_preamble_texts or [])
        if initial_reply_text and not send_initial_reply:
            effective_final_preamble_texts.insert(0, initial_reply_text)

        self._log_detail(
            "info",
            f"OpenAIImage queue generation source={source} platform={platform_context['platform_name']} "
            f"send_initial_reply={send_initial_reply} final_preamble_count={len(effective_final_preamble_texts)} "
            f"prompt={final_prompt}",
        )
        self._create_background_task(
            self._run_generation_task(
                unified_msg_origin=event.unified_msg_origin,
                prompt=final_prompt,
                source=source,
                platform_context=platform_context,
                final_preamble_texts=effective_final_preamble_texts,
                polished_prompt=polished_prompt,
            )
        )
        return initial_reply_text if send_initial_reply else None

    async def _queue_edit_task(
        self,
        event: AstrMessageEvent,
        prompt: str,
        source: str,
        initial_reply_text: str | None = None,
        final_preamble_texts: list[str] | None = None,
        use_mask: bool = False,
        use_all_images: bool = False,
        polished_prompt: str | None = None,
    ) -> str | None:
        final_prompt = self._compose_prompt(prompt)
        if not final_prompt:
            raise ValueError("请输入你希望如何修改图片，例如：/修图 改成赛博朋克夜景风格")

        input_images = await self._extract_input_images(event)
        if not input_images:
            raise ValueError("请在同一条消息里附带至少一张图片，再使用图生图或修图命令。")

        source_image_paths = input_images if use_all_images else [input_images[0]]
        mask_image_path = input_images[1] if use_mask and len(input_images) > 1 else None
        platform_context = self._get_platform_context(event)
        send_initial_reply = bool(initial_reply_text and platform_context["allow_initial_reply"])
        effective_final_preamble_texts = list(final_preamble_texts or [])
        if initial_reply_text and not send_initial_reply:
            effective_final_preamble_texts.insert(0, initial_reply_text)

        self._log_detail(
            "info",
            f"OpenAIImage queue edit source={source} platform={platform_context['platform_name']} "
            f"send_initial_reply={send_initial_reply} final_preamble_count={len(effective_final_preamble_texts)} "
            f"image_count={len(source_image_paths)} images={source_image_paths} "
            f"mask={mask_image_path or 'none'} prompt={final_prompt}",
        )
        self._create_background_task(
            self._run_edit_task(
                unified_msg_origin=event.unified_msg_origin,
                prompt=final_prompt,
                image_paths=source_image_paths,
                mask_path=mask_image_path,
                source=source,
                platform_context=platform_context,
                final_preamble_texts=effective_final_preamble_texts,
                polished_prompt=polished_prompt,
            )
        )
        return initial_reply_text if send_initial_reply else None

    @filter.command("画图", alias={"draw", "image", "绘图"})
    async def draw_image(self, event: AstrMessageEvent, prompt: str = ""):
        """使用 OpenAI 图像模型生成图片。"""
        command_prompt = self._extract_command_prompt(event, prompt)
        if not command_prompt:
            yield event.plain_result("请输入要绘制的内容，例如：/画图 一只戴墨镜的赛博朋克橘猫")
            return

        try:
            initial_reply = await self._queue_generation_task(
                event=event,
                prompt=command_prompt,
                source="command",
                initial_reply_text="正在生成图片，请稍等……",
            )
        except ValueError as exc:
            yield event.plain_result(str(exc))
            return
        if initial_reply:
            yield event.plain_result(initial_reply)

    @filter.command("图生图", alias={"img2img", "image2image"})
    async def image_to_image(self, event: AstrMessageEvent, prompt: str = ""):
        """基于用户先发送的一张或多张图片生成新图片。第一张为主图，后续图片为参考图。"""
        command_prompt = self._extract_command_prompt(event, prompt)
        if not command_prompt:
            yield event.plain_result("请先发图片  再发  /图生图 修改要求")
            return

        try:
            await self._queue_edit_confirmation(
                event=event,
                prompt=command_prompt,
                mode="img2img",
                source="img2img_command",
            )
        except ValueError as exc:
            yield event.plain_result(str(exc))
            return
        event.stop_event()

    @filter.command("修图", alias={"改图", "edit", "inpaint"})
    async def edit_image(self, event: AstrMessageEvent, prompt: str = ""):
        """基于用户先发送的原图进行修图。第二张图片会被当作遮罩使用。"""
        command_prompt = self._extract_command_prompt(event, prompt)
        if not command_prompt:
            yield event.plain_result("请先发图  再发  /修图 修改要求")
            return

        try:
            await self._queue_edit_confirmation(
                event=event,
                prompt=command_prompt,
                mode="edit",
                source="edit_command",
            )
        except ValueError as exc:
            yield event.plain_result(str(exc))
            return
        event.stop_event()

    @filter.event_message_type(filter.EventMessageType.ALL, priority=1)
    async def handle_pending_edit_confirmation(self, event: AstrMessageEvent):
        normalized_text = self._normalize_message_text(str(event.message_str or ""))
        input_images = await self._extract_input_images(event)
        pending = self._get_pending_confirmation(event)

        if pending:
            if self._is_known_command_text(normalized_text):
                return

            if input_images and not normalized_text:
                yield event.plain_result("当前有待确认的任务  回复 确认  取消  或直接说怎么调顺序")
                event.stop_event()
                return

            if not normalized_text:
                return

            if self._is_confirm_reply(normalized_text):
                self._clear_pending_confirmation(event)
                try:
                    await self._start_confirmed_edit_task(event, pending)
                except ValueError as exc:
                    yield event.plain_result(str(exc))
                event.stop_event()
                return

            if self._is_cancel_reply(normalized_text):
                self._clear_pending_confirmation(event)
                self._clear_staged_images(event)
                yield event.plain_result("已取消  这次不处理了")
                event.stop_event()
                return

            updated_image_paths = await self._resolve_adjusted_image_paths(
                event=event,
                pending=pending,
                instruction=normalized_text,
            )
            if not updated_image_paths:
                yield event.plain_result("没看懂  可以直接说  把第二张放第一张前面  或回复 确认 取消")
                event.stop_event()
                return

            pending.image_paths = updated_image_paths
            self._set_pending_confirmation(event, pending)
            await self._send_confirmation_preview(event, pending, source=f"{pending.mode}_confirm_adjust")
            event.stop_event()
            return

        if input_images and not normalized_text and not self._is_known_command_text(normalized_text):
            self._stage_images(event, input_images)
            return

        if normalized_text and not self._is_known_command_text(normalized_text):
            self._clear_staged_images(event)
