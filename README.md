# AI 物理解题助手

一个基于 **Qwen + Gradio** 的物理解题 Web 应用。

支持：

- 直接输入文字物理题
- 上传物理题图片
- 调用 Qwen 进行深度推理
- 页面仅展示正式解题过程和最终答案
- 流式输出解题内容
- 提交后显示运行进度与状态
- `run.py` 支持 Microsoft Edge TTS 中文语音朗读
- 语音可在答案尚未全部生成完成时提前开始播放

---

## 项目结构

```text
project/
├── app.py
├── run.py
├── .env
├── README.md
└── .gitignore
```

其中：

### `app.py`

基础版本。

主要功能：

- 输入文字物理题
- 上传物理题图片
- 调用 Qwen 模型解题
- 流式显示答案
- 显示提交进度
- 不包含实时语音朗读功能

### `run.py`

增强版本。

在 `app.py` 基础能力之上增加：

- 更简洁的 Gradio 界面
- 隐藏模型内部 `reasoning_content`
- 只显示正式解题过程和最终答案
- Microsoft Edge TTS 中文语音
- 文本生成与语音生成并行
- 答案按句分段后立即进入 TTS
- 尽可能实现边生成答案、边生成语音、边播放
- 对前端刷新做节流，减少 Gradio 页面闪烁

---

## 运行环境

建议：

- macOS
- Python 3.11
- `uv` 虚拟环境

其他支持 Python 3.11 的系统通常也可以运行。

---

## 创建虚拟环境

如果本地已经安装 `uv`：

```bash
uv venv --python 3.11
source .venv/bin/activate
```

---

## 安装依赖

基础依赖：

```bash
uv pip install openai gradio python-dotenv pillow
```

如果需要运行带语音功能的 `run.py`：

```bash
uv pip install edge-tts
```

也可以一次安装：

```bash
uv pip install openai gradio python-dotenv pillow edge-tts
```

---

## 配置 API

在项目根目录创建：

```text
.env
```

写入：

```env
DASHSCOPE_API_KEY=你的_API_KEY
```

程序默认使用：

```text
qwen3.7-flash
```

默认 OpenAI Compatible API 地址：

```text
https://llm-0nr9mznstg4wrz4h.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

如需修改，也可以继续在 `.env` 中配置：

```env
DASHSCOPE_API_KEY=你的_API_KEY
DASHSCOPE_MODEL=qwen3.7-flash
DASHSCOPE_BASE_URL=https://llm-0nr9mznstg4wrz4h.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
SERVER_NAME=127.0.0.1
SERVER_PORT=7862
TTS_VOICE=zh-CN-XiaoxiaoNeural
```

> `.env` 中包含 API Key，不要提交到 GitHub 或其他公开仓库。

---

## 运行基础版

```bash
python app.py
```

默认访问：

```text
http://127.0.0.1:7862
```

---

## 运行增强版

```bash
python run.py
```

默认访问：

```text
http://127.0.0.1:7862
```

增强版会在 Qwen 正式回答后，将完整句子逐步交给 Edge TTS，因此无需等待整篇答案全部生成完成后才开始生成语音。

---

## 使用方式

### 文字题

直接在输入框中输入物理题，然后点击：

```text
提交解题
```

系统会自动调用 Qwen，并流式输出解题结果。

### 图片题

上传包含物理题的图片。

支持识别例如：

- 题目文字
- 数字与单位
- 电路图
- 受力图
- 坐标图
- 几何关系
- 长度与角度
- 图中标注

也可以在上传图片后额外输入要求，例如：

```text
请重点解释第 2 问。
```

---

## 模型回答策略

程序允许 Qwen 在内部开启深度推理：

```python
extra_body={"enable_thinking": True}
```

但页面不会展示模型内部的：

```text
reasoning_content
```

用户只会看到正式的：

```text
解题过程
最终答案
```

因此内部推理和最终展示内容是分离的。

---

## 语音功能

`run.py` 使用：

```text
edge-tts
```

调用 Microsoft Edge 在线语音服务。

默认中文声音：

```text
zh-CN-XiaoxiaoNeural
```

程序还可配置其他中文声音。

语音处理流程大致如下：

```text
Qwen 流式生成正式答案
        ↓
检测完整句子
        ↓
送入 TTS 队列
        ↓
Edge TTS 生成语音
        ↓
Gradio Audio 播放
```

Qwen 和 TTS 使用独立工作线程，因此 Qwen 可以继续生成后面的答案，而前面的内容已经开始生成语音。

---

## 注意事项

### 1. API Key 安全

不要把 `.env` 上传到 Git。

如果 API Key 曾经：

- 发到公开聊天
- 上传到公开仓库
- 出现在截图
- 提交到 Git 历史

建议立即重新生成新的 Key。

### 2. Edge TTS

`edge-tts` 不需要 Azure Speech API Key，但依赖 Microsoft Edge 在线语音服务以及网络连接。

如果网络不可用，文字解题仍然可以正常工作，语音功能可能失败。

### 3. 浏览器自动播放

`run.py` 会尝试自动播放语音。

部分浏览器可能因为自动播放策略，在第一次使用时要求用户手动点击一次音频播放器。

### 4. 图片识别

图片越清楚，模型识别效果越好。

建议：

- 图片保持正方向
- 避免严重模糊
- 避免过暗或强反光
- 尽量让题目完整出现在画面中
- 电路图、受力图等尽量保证标注清晰

---

## 当前主要文件

本项目核心代码仅保留：

```text
app.py
run.py
```

其中：

- `app.py`：基础稳定版
- `run.py`：流式文本 + 实时语音增强版

后续功能建议优先在 `run.py` 上继续迭代。
