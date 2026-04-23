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
from typing import Any
from urllib.parse import urlparse

import aiohttp
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-image-2"
COMMAND_NAMES = {"画图", "draw", "image", "绘图"}


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

        try:
            await plugin._queue_generation_task(
                event=event,
                prompt=cleaned_prompt,
                source="llm_tool",
            )
        except ValueError as exc:
            return str(exc)
        return "图像生成任务已启动，结果会在完成后自动发送到当前会话。"


@register("OpenAIImage", "SanHans", "使用 OpenAI 兼容接口生成图片", "1.0.8")
class OpenAIImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._tool_registered = False
        max_concurrent_tasks = int(self.config.get("max_concurrent_tasks", 3) or 3)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_tasks))
        self._image_temp_dir = os.path.abspath("data/temp/openai_image")
        os.makedirs(self._image_temp_dir, exist_ok=True)

    async def initialize(self):
        if not self._tool_registered:
            self.context.add_llm_tools(GenerateImageTool(plugin=self))
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

    def _get_legacy_timeout(self) -> int:
        return max(5, int(self.config.get("timeout", 180) or 180))

    def _get_generation_timeout(self) -> int:
        value = self.config.get("generation_timeout", self._get_legacy_timeout())
        return max(5, int(value or self._get_legacy_timeout()))

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

    def _get_base_url(self) -> str:
        base_url = str(self.config.get("base_url") or DEFAULT_BASE_URL).strip().rstrip("/")
        if not base_url:
            base_url = DEFAULT_BASE_URL
        if base_url.endswith("/images/generations"):
            return base_url
        return f"{base_url}/images/generations"

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
            if command in {name.lower() for name in COMMAND_NAMES}:
                tokens = tokens[1:]

        reconstructed = " ".join(tokens).strip()
        return self._cleanup_prompt(reconstructed or raw_prompt)

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

    async def _request_image_generation(self, prompt: str) -> dict[str, Any]:
        api_key = self._get_api_key()
        endpoint = self._get_base_url()
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

    def _extract_revised_prompt(self, data: dict[str, Any]) -> str | None:
        data_list = data.get("data")
        if isinstance(data_list, list) and data_list and isinstance(data_list[0], dict):
            revised_prompt = str(data_list[0].get("revised_prompt") or "").strip()
            return revised_prompt or None
        return None

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
    ) -> None:
        async with self._semaphore:
            try:
                logger.info(
                    f"OpenAIImage request source={source} platform={platform_context['platform_name']} "
                    f"platform_id={platform_context['platform_id']} prompt={prompt}"
                )
                response = await self._request_image_generation(prompt)
                revised_prompt = self._extract_revised_prompt(response)
                preamble_texts = list(final_preamble_texts or [])
                if self.config.get("send_prompt_back", False) and revised_prompt:
                    preamble_texts.append(f"最终提示词：{revised_prompt}")
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

    async def _queue_generation_task(
        self,
        event: AstrMessageEvent,
        prompt: str,
        source: str,
        initial_reply_text: str | None = None,
        final_preamble_texts: list[str] | None = None,
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
