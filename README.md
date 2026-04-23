# AstrBot OpenAI 画图插件

这是一个 AstrBot 插件，使用 OpenAI 兼容的 `images/generations` 接口进行生图，默认模型为 `gpt-image-2`。

它提供三种入口：

- `/画图 提示词`
- 高置信度自然语言触发，例如 `画图 一只戴墨镜的柴犬`、`帮我画一只小猫`、`帮我生成一张图片 赛博朋克上海夜景`
- `LLM Tool`，当 AstrBot 的对话模型支持工具调用时，可在对话中自动调用 `generate_image`

当前行为：

- `/画图` 指令：原样使用用户提示词，不额外调用 LLM 润色
- 插件自然语言触发：先使用当前会话正在使用的 AstrBot provider 结合当前人格做一次轻度润色，再发图；要求是不改变用户本意
- `LLM Tool`：工具描述会要求 Agent 尽量原样传达用户意图，减少自行脑补风格词

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
- `size`：默认 `auto`，也可以填 `1024x1024`、`1536x1024`、`1024x1536` 等
- `quality`：`auto` / `low` / `medium` / `high`
- `background`：`auto` / `opaque` / `transparent`
- `output_format`：`png` / `jpeg` / `webp`
- `moderation`：`auto` / `low`

## 使用示例

```text
/画图 一只戴墨镜的赛博朋克橘猫，霓虹灯雨夜，电影感，高细节
```

```text
画图 一个漂浮在云海上的玻璃图书馆，晨光，超现实主义
```

```text
帮我生成一张图片：雪山脚下的温泉木屋，黄昏，写实摄影风格
```

## 说明

- 插件直接调用 `POST {base_url}/images/generations`
- 默认会返回一条“正在生成图片，请稍等……”的提示，再发送图片结果
- 如果自然语言触发已处理消息，默认会调用 `event.stop_event()`，避免同一条消息继续流向后续 LLM 处理造成重复回复
- `gpt-image-2` 官方当前不支持透明背景；如果你使用的是兼容服务并支持其他模型，可自行切换
