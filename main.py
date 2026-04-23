import asyncio
import json
import os
import re
from typing import Any

import aiohttp

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-image-2"
COMMAND_NAMES = {"画图", "draw", "image", "绘图"}
NATURAL_LANGUAGE_PATTERNS = (
    re.compile(r"^\s*画图[:：\s]*(?P<prompt>.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:请|请你)?(?:帮我|给我|替我|麻烦你)?画(?:一张|个|幅|张)图?\s*(?P<prompt>.+?)\s*$", re.IGNORECASE),
    re.compile(
        r"^\s*(?:请|请你)?(?:帮我|给我|替我|麻烦你)?(?:生成|做)(?:一张|个|幅)?(?:图片|图像|图)\s*(?P<prompt>.+?)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:请|请你)?(?:帮我|给我|替我|麻烦你)?绘制(?:一张|个|幅)?(?:图片|图像|图)?\s*(?P<prompt>.+?)\s*$",
        re.IGNORECASE,
    ),
)


@register("OpenAIImage", "Codex", "使用 OpenAI 兼容接口生成图片", "1.0.0")
class OpenAIImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        max_concurrent_tasks = int(self.config.get("max_concurrent_tasks", 3) or 3)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_tasks))

    async def terminate(self):
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        timeout = int(self.config.get("timeout", 180) or 180)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout))
        return self._session

    def _get_api_key(self) -> str:
        return str(self.config.get("api_key") or os.getenv("OPENAI_API_KEY") or "").strip()

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

    def _extract_natural_prompt(self, message: str) -> str | None:
        text = str(message or "").strip()
        if not text:
            return None

        if text.startswith(("/", "!", ".", "#")):
            return None

        for pattern in NATURAL_LANGUAGE_PATTERNS:
            match = pattern.match(text)
            if not match:
                continue

            prompt = self._cleanup_prompt(match.group("prompt"))
            if prompt:
                return prompt

        return None

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

        async with session.post(endpoint, headers=headers, json=payload) as response:
            text = await response.text()
            data: dict[str, Any]
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"raw_text": text}

            if response.status >= 400:
                error_message = (
                    data.get("error", {}).get("message")
                    if isinstance(data.get("error"), dict)
                    else data.get("error")
                ) or data.get("message") or text
                raise RuntimeError(f"请求失败（{response.status}）：{error_message}")

            return data

    def _build_image_chain(self, data: dict[str, Any]) -> list[Any]:
        chain: list[Any] = []
        images = data.get("data", [])
        if not isinstance(images, list):
            raise RuntimeError("接口返回格式异常：缺少 data 数组。")

        for item in images:
            if not isinstance(item, dict):
                continue

            b64_json = item.get("b64_json")
            image_url = item.get("url")

            if isinstance(b64_json, str) and b64_json.strip():
                chain.append(Image.fromBase64(b64_json.strip()))
                continue

            if isinstance(image_url, str) and image_url.strip():
                chain.append(Image.fromURL(image_url.strip()))

        if not chain:
            raise RuntimeError("接口未返回可发送的图片数据。")

        return chain

    async def _generate_and_reply(self, event: AstrMessageEvent, prompt: str, source: str):
        final_prompt = self._compose_prompt(prompt)
        if not final_prompt:
            yield event.plain_result("请输入要绘制的内容，例如：/画图 一只戴墨镜的赛博朋克橘猫")
            return

        async with self._semaphore:
            try:
                logger.info(f"OpenAIImage request source={source} prompt={final_prompt}")
                yield event.plain_result("正在生成图片，请稍等……")

                response = await self._request_image_generation(final_prompt)
                if self.config.get("send_prompt_back", False):
                    revised_prompt = None
                    data_list = response.get("data")
                    if isinstance(data_list, list) and data_list and isinstance(data_list[0], dict):
                        revised_prompt = data_list[0].get("revised_prompt")
                    if revised_prompt:
                        yield event.plain_result(f"最终提示词：{revised_prompt}")

                yield event.chain_result(self._build_image_chain(response))
            except ValueError as exc:
                logger.warning(f"OpenAIImage invalid config: {exc}")
                yield event.plain_result(f"配置错误：{exc}")
            except aiohttp.ClientError as exc:
                logger.error(f"OpenAIImage network error: {exc}")
                yield event.plain_result(f"网络请求失败：{exc}")
            except asyncio.TimeoutError:
                logger.error("OpenAIImage request timed out")
                yield event.plain_result("请求超时，请稍后重试，或适当调低图片质量/尺寸。")
            except Exception as exc:
                logger.exception(f"OpenAIImage generation failed: {exc}")
                yield event.plain_result(f"生成失败：{exc}")

    @filter.command("画图", alias={"draw", "image", "绘图"})
    async def draw_image(self, event: AstrMessageEvent, prompt: str = ""):
        """使用 OpenAI 图像模型生成图片。"""
        command_prompt = self._extract_command_prompt(event, prompt)
        async for result in self._generate_and_reply(event, command_prompt, "command"):
            yield result

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_natural_language_draw(self, event: AstrMessageEvent):
        """识别高置信度的自然语言画图请求。"""
        if not self.config.get("natural_language_enabled", True):
            return

        prompt = self._extract_natural_prompt(event.message_str or "")
        if not prompt:
            return

        if self.config.get("stop_event_after_natural_language", True):
            event.stop_event()

        async for result in self._generate_and_reply(event, prompt, "natural_language"):
            yield result

    @filter.llm_tool(name="generate_image")
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str):
        """根据用户描述生成图片。

        Args:
            prompt(string): 用于生成图片的完整描述，应该直接包含主体、风格、构图、背景、光照等要求。
        """
        async for result in self._generate_and_reply(event, self._cleanup_prompt(prompt), "llm_tool"):
            yield result
