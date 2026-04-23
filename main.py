import asyncio
import base64
import json
import os
import re
import time
import uuid
from typing import Any

import aiohttp

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star, register


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_NATURAL_LANGUAGE_POLISH_PROMPT_TEMPLATE = """你是 AstrBot 画图插件的自然语言解析器。
请根据“用户原话”和“当前人格设定”，输出一个 JSON 对象，格式严格如下：
{"prompt":"适合图像模型的最终提示词","reply":"一条符合当前人格口吻的简短回复"}

规则：
1. prompt 必须保留用户本意，只做轻度润色和必要补全，不得擅自改变主体、动作、关系、场景、风格、时代、情绪和媒介。
2. 如果用户没有明确指定风格，不要擅自添加二次元、写实、电影感、赛博朋克等强风格词。
3. 不要添加大量画质、镜头、构图、光照标签，不要把简单需求扩写成堆砌关键词。
4. reply 要符合当前人格口吻，自然、简短，表达“我来帮你画/这就画”，不要输出“正在生成图片，请稍等……”这句原话。
5. 只输出 JSON，不要解释，不要 Markdown，不要代码块。

当前人格设定：
{{persona_prompt}}

平台类型：
{{platform_name}}

用户原话：
{{user_prompt}}
"""
COMMAND_NAMES = {"画图", "draw", "image", "绘图"}
NATURAL_LANGUAGE_PATTERNS = (
    re.compile(r"^\s*画图[:：\s]*(?P<prompt>.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:请|请你)?(?:帮我|给我|替我|麻烦你)\s*画(?P<prompt>.+?)\s*$", re.IGNORECASE),
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
NATURAL_LANGUAGE_EXCLUDED_PREFIXES = ("重点", "个重点", "重点画", "线", "圈", "标记")


@register("OpenAIImage", "SanHans", "使用 OpenAI 兼容接口生成图片", "1.0.2")
class OpenAIImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        max_concurrent_tasks = int(self.config.get("max_concurrent_tasks", 3) or 3)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_tasks))
        self._image_temp_dir = os.path.abspath("data/temp/openai_image")
        os.makedirs(self._image_temp_dir, exist_ok=True)

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
        allow_progress_message = True

        if is_wecom_ai_bot:
            allow_progress_message = False
            if not has_wecom_webhook:
                image_limit_reason = (
                    "当前企业微信智能机器人未配置消息推送 Webhook URL。"
                    "插件会避免先发送进度消息，尽量把文本和图片合并为一次最终回复；"
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
            "allow_progress_message": allow_progress_message,
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
            if prompt and not prompt.startswith(NATURAL_LANGUAGE_EXCLUDED_PREFIXES):
                return prompt

        return None

    @staticmethod
    def _sanitize_polished_prompt(text: str) -> str:
        prompt = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE).strip()
        prompt = re.sub(r"^(prompt|提示词)\s*[:：]\s*", "", prompt, flags=re.IGNORECASE).strip()
        if prompt.startswith(("```", '"""', "'''")):
            prompt = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", prompt).strip()
            prompt = re.sub(r"\n?```$", "", prompt).strip()
            prompt = prompt.strip('"\'')
        return prompt.strip().strip('"\'')

    @staticmethod
    def _sanitize_json_response(text: str) -> str:
        content = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE).strip()
        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
        if code_match:
            content = code_match.group(1).strip()
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        return content.strip()

    def _parse_polish_payload(self, text: str) -> tuple[str | None, str | None]:
        cleaned = self._sanitize_json_response(text)
        if not cleaned:
            return None, None

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            self._log_detail("warning", f"OpenAIImage failed to parse natural language JSON: {exc}; raw={text}")
            return None, None

        if not isinstance(data, dict):
            return None, None

        prompt = self._cleanup_prompt(str(data.get("prompt") or data.get("final_prompt") or ""))
        reply = self._cleanup_prompt(str(data.get("reply") or data.get("message") or ""))
        return (prompt or None), (reply or None)

    def _render_natural_language_polish_prompt(
        self,
        prompt: str,
        persona_prompt: str,
        platform_name: str,
    ) -> str:
        template = str(
            self.config.get("natural_language_polish_prompt_template")
            or DEFAULT_NATURAL_LANGUAGE_POLISH_PROMPT_TEMPLATE
        ).strip()
        return (
            template.replace("{{user_prompt}}", prompt)
            .replace("{{persona_prompt}}", persona_prompt or "未配置人格设定")
            .replace("{{platform_name}}", platform_name or "unknown")
        )

    async def _polish_natural_language_prompt(
        self,
        event: AstrMessageEvent,
        prompt: str,
    ) -> tuple[str, str | None]:
        platform_context = self._get_platform_context(event)
        provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        if not provider:
            self._log_detail("warning", "OpenAIImage no provider found for natural language polishing; fallback to raw prompt")
            return prompt, self.config.get("natural_language_fallback_reply", "好，我来画这个。")

        try:
            persona = await self.context.persona_manager.get_default_persona_v3(
                event.unified_msg_origin
            )
            persona_prompt = str(persona.get("prompt") or "").strip()
        except Exception as exc:
            logger.warning(f"OpenAIImage failed to load persona, fallback to raw prompt: {exc}")
            persona_prompt = ""
        system_prompt = self._render_natural_language_polish_prompt(
            prompt=prompt,
            persona_prompt=persona_prompt,
            platform_name=str(platform_context["platform_name"]),
        )
        user_prompt = "请严格按照系统要求输出 JSON 结果。"
        self._log_detail(
            "info",
            f"OpenAIImage natural language polish request provider={provider.meta().id} model={provider.meta().model} "
            f"platform={platform_context['platform_name']} original_prompt={prompt} persona_prompt={persona_prompt}",
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
                f"OpenAIImage natural language polish response took={duration:.2f}s raw={raw_output}",
            )
            polished_prompt, reply_text = self._parse_polish_payload(raw_output)
            if polished_prompt:
                logger.info(
                    f"OpenAIImage polished natural prompt original={prompt} polished={polished_prompt} reply={reply_text}"
                )
                return polished_prompt, reply_text or self.config.get("natural_language_fallback_reply", "好，我来画这个。")
        except Exception as exc:
            logger.warning(f"OpenAIImage failed to polish natural prompt, fallback to raw prompt: {exc}")

        return prompt, self.config.get("natural_language_fallback_reply", "好，我来画这个。")

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
        self._log_detail(
            "info",
            f"OpenAIImage request endpoint={endpoint} model={payload.get('model')} size={payload.get('size')} "
            f"quality={payload.get('quality')} n={payload.get('n')} moderation={payload.get('moderation')} "
            f"prompt={prompt}",
        )

        started_at = time.monotonic()
        async with session.post(endpoint, headers=headers, json=payload) as response:
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
                f"OpenAIImage response status={response.status} took={duration:.2f}s data_count={data_count} "
                f"keys={list(data.keys()) if isinstance(data, dict) else type(data)}",
            )

            if response.status >= 400:
                error_message = (
                    data.get("error", {}).get("message")
                    if isinstance(data.get("error"), dict)
                    else data.get("error")
                ) or data.get("message") or text
                raise RuntimeError(f"请求失败（{response.status}）：{error_message}")

            return data

    def _save_base64_image(self, image_base64: str) -> str:
        output_format = str(self.config.get("output_format") or "png").strip().lower()
        extension = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp"}.get(output_format, "png")
        filename = f"openai_image_{uuid.uuid4().hex}.{extension}"
        file_path = os.path.join(self._image_temp_dir, filename)
        with open(file_path, "wb") as file:
            file.write(base64.b64decode(image_base64))
        self._log_detail("info", f"OpenAIImage saved image to {file_path}")
        return file_path

    def _extract_revised_prompt(self, data: dict[str, Any]) -> str | None:
        data_list = data.get("data")
        if isinstance(data_list, list) and data_list and isinstance(data_list[0], dict):
            revised_prompt = str(data_list[0].get("revised_prompt") or "").strip()
            return revised_prompt or None
        return None

    def _build_image_chain(
        self,
        data: dict[str, Any],
        preamble_texts: list[str] | None = None,
    ) -> list[Any]:
        chain: list[Any] = []
        for text in preamble_texts or []:
            if text:
                chain.append(Plain(text))

        images = data.get("data", [])
        if not isinstance(images, list):
            raise RuntimeError("接口返回格式异常：缺少 data 数组。")

        for item in images:
            if not isinstance(item, dict):
                continue

            b64_json = item.get("b64_json")
            image_url = item.get("url")

            if isinstance(b64_json, str) and b64_json.strip():
                file_path = self._save_base64_image(b64_json.strip())
                chain.append(Image.fromFileSystem(file_path))
                continue

            if isinstance(image_url, str) and image_url.strip():
                self._log_detail("info", f"OpenAIImage using remote image URL {image_url.strip()}")
                chain.append(Image.fromURL(image_url.strip()))

        if not any(isinstance(component, Image) for component in chain):
            raise RuntimeError("接口未返回可发送的图片数据。")

        return chain

    async def _generate_and_reply(
        self,
        event: AstrMessageEvent,
        prompt: str,
        source: str,
        natural_language_reply: str | None = None,
        send_progress_message: bool | None = None,
    ):
        final_prompt = self._compose_prompt(prompt)
        if not final_prompt:
            yield event.plain_result("请输入要绘制的内容，例如：/画图 一只戴墨镜的赛博朋克橘猫")
            return

        platform_context = self._get_platform_context(event)
        if send_progress_message is None:
            send_progress_message = bool(platform_context["allow_progress_message"] and source != "natural_language")

        async with self._semaphore:
            try:
                logger.info(
                    f"OpenAIImage request source={source} platform={platform_context['platform_name']} "
                    f"platform_id={platform_context['platform_id']} prompt={final_prompt}"
                )
                if send_progress_message:
                    yield event.plain_result("正在生成图片，请稍等……")

                response = await self._request_image_generation(final_prompt)
                revised_prompt = self._extract_revised_prompt(response)
                preamble_texts: list[str] = []
                if natural_language_reply:
                    preamble_texts.append(natural_language_reply)
                if self.config.get("send_prompt_back", False) and revised_prompt:
                    preamble_texts.append(f"最终提示词：{revised_prompt}")

                chain = self._build_image_chain(response, preamble_texts=preamble_texts)
                self._log_detail(
                    "info",
                    f"OpenAIImage built final message chain components={[type(component).__name__ for component in chain]}",
                )
                yield event.chain_result(chain)
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

        prompt, reply_text = await self._polish_natural_language_prompt(event, prompt)

        if self.config.get("stop_event_after_natural_language", True):
            event.stop_event()

        async for result in self._generate_and_reply(
            event,
            prompt,
            "natural_language",
            natural_language_reply=reply_text,
            send_progress_message=False,
        ):
            yield result

    @filter.llm_tool(name="generate_image")
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str):
        """根据用户描述生成图片。

        Args:
            prompt(string): 将用户的画图需求尽量原样传给生图模型。除非用户明确要求，否则不要主动添加新的风格、构图、光照、背景、画质标签或其他会改变语义侧重点的描述。
        """
        async for result in self._generate_and_reply(event, self._cleanup_prompt(prompt), "llm_tool"):
            yield result
