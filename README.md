# AstrBot OpenAI 画图插件

这是一个 AstrBot 插件，使用 OpenAI 兼容的图像接口进行生图、图生图和修图，默认模型为 `gpt-image-2`。

支持独立的生图/下载超时、失败重试，以及可选代理转发。

它提供以下入口：

- `/画图 提示词`
- `/图生图 提示词`，同一条消息里附带原图
- `/修图 提示词`，同一条消息里附带原图；如果再附带第二张图，会当作遮罩图上传
- `LLM Tool`，当 AstrBot 的对话模型支持工具调用时，可自动调用 `generate_image` 或 `edit_image`

当前行为：

- `/画图` 指令：原样使用用户提示词，不额外调用 LLM 润色
- `/图生图` 和 `/修图`：直接读取当前消息链里的图片，不依赖平台私有字段；`/修图` 在有第二张图片时会把第二张图作为 `mask`
- 普通自然语言对话：不再由插件直接拦截，而是交给 AstrBot Agent 按当前人格正常回复；当 Agent 判断用户是在要图片或改图时，再调用对应工具
- `LLM Tool`：会注册 `generate_image` 和 `edit_image` 两个工具，工具本身只负责启动任务并把最终图片主动推送回当前会话
- 可选自然语言润色：可以开启 `natural_language_polish_enabled`，让插件在 Agent 调用工具后，再按你自定义的优化提示词做一次轻量整理

## 文件结构

插件目录应包含以下文件：

- `main.py`
- `metadata.yaml`
- `_conf_schema.json`
- `requirements.txt`

## 安装方式

将当前目录放入 AstrBot 的插件目录，例如：

```text
AstrBot/data/plugins/astrbot_plugin_openai_image
```

然后在 AstrBot 面板中启用插件，并填写配置项。

## 关键配置

- `base_url`：OpenAI 兼容接口根地址，建议写到 `/v1`
- `api_key`：接口密钥；留空时会尝试读取环境变量 `OPENAI_API_KEY`
- `model`：默认 `gpt-image-2`
- `generation_timeout`：生图接口总超时，默认 `180` 秒
- `download_timeout`：下载远程图片 URL 的总超时，默认 `60` 秒
- `retry_count`：失败重试次数，默认 `1`
- `retry_backoff_seconds`：重试退避秒数，默认 `2`
- `proxy_url`：可选代理地址，会同时用于生图请求和远程图片下载
- `natural_language_polish_enabled`：是否对 Agent 传给工具的自然语言提示词再做一次轻量润色，默认关闭
- `natural_language_polish_prompt_template`：自然语言优化提示词，可自行修改；直接填写优化提示词正文，不需要变量
- `detailed_logging`：输出更详细的插件日志
- `size`：默认 `auto`，也可以填 `1024x1024`、`1536x1024`、`1024x1536` 等
- `quality`：`auto` / `low` / `medium` / `high`
- `background`：`auto` / `opaque` / `transparent`
- `output_format`：`png` / `jpeg` / `webp`
- `moderation`：`auto` / `low`

## 使用示例

```text
/画图 一只戴墨镜的赛博朋克橘猫，霓虹灯雨夜，电影感，高细节
/图生图 改成宫崎骏动画风
/修图 去掉背景路人并补全建筑立面
```

## 说明

- 文字生图调用 `POST {base_url}/images/generations`
- 图生图和修图调用 `POST {base_url}/images/edits`
- `/画图` 指令会先快速结束当前处理，再通过 `self.context.send_message(...)` 主动推送最终图片，避免长耗时请求卡住平台回复窗口
- `/图生图` 和 `/修图` 需要把图片和命令放在同一条消息里；如果是 `/修图`，第二张图会被当作遮罩图一起上传
- 普通“帮我画一只猫”“把这张图改成电影海报风”这类自然语言消息不会被插件直接拦截，而是由 AstrBot Agent 按人格回复并按需调用工具，所以用户看到的是 Agent 在说话
- 生图请求和远程图片下载分别使用独立超时；对 408/429/500/502/503/504、网络异常和本地超时会按配置自动重试
- `gpt-image-2` 官方当前不支持透明背景；如果你使用的是兼容服务并支持其他模型，可自行切换
- 如果你使用的是 `wecom_ai_bot`（企业微信智能机器人）渠道，建议配置 `msg_push_webhook_url`，并优先使用 Webhook 主动推送结果；插件会额外记录平台判断、排队、下载图片、主动发消息成败等详细日志
