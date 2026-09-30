from __future__ import annotations

import base64
import mimetypes
import os
import queue
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import Iterator

import edge_tts
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


# =========================================================
# 基础配置
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# 读取同目录 .env
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv(
    "DASHSCOPE_API_KEY",
    "",
).strip()

BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://llm-0nr9mznstg4wrz4h.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
).strip()

MODEL = os.getenv(
    "DASHSCOPE_MODEL",
    "qwen3.7-flash",
).strip()

SERVER_NAME = os.getenv(
    "SERVER_NAME",
    "127.0.0.1",
)

SERVER_PORT = int(
    os.getenv(
        "SERVER_PORT",
        "7862",
    )
)

DEFAULT_VOICE = os.getenv(
    "TTS_VOICE",
    "zh-CN-XiaoxiaoNeural",
)


# =========================================================
# Microsoft Edge TTS 声音
# =========================================================

VOICE_OPTIONS = [
    (
        "晓晓 · 女声 · 自然",
        "zh-CN-XiaoxiaoNeural",
    ),
    (
        "云希 · 男声 · 青年",
        "zh-CN-YunxiNeural",
    ),
    (
        "云健 · 男声 · 沉稳",
        "zh-CN-YunjianNeural",
    ),
]


# =========================================================
# 系统提示词
# =========================================================

SYSTEM_PROMPT = r"""
你是一名专业、严谨、准确的物理解题助手。

你可以在内部进行充分的深度推理，但是绝对不要把：
- 内部思考
- reasoning
- reasoning_content
- 思维链
- 分析草稿
- 试错过程
- 自我反思
- 隐藏推理过程

展示给用户。

用户最终只能看到正式、清楚、准确、容易理解的解题内容。

请遵守以下规则：

1. 不要输出：
   “思考过程”
   “分析过程”
   “我的推理过程”
   “首先我要思考”
   “让我分析一下”
   等内容。

2. 不要大段重复原题。

3. 直接识别：
   - 已知条件
   - 求解目标
   - 所需物理规律

4. 给出必要、清晰的：
   - 公式
   - 推导
   - 数据代入
   - 单位
   - 计算结果

5. 解题过程必须完整，但不要写无关的探索过程。

6. 如果题目来自图片，请仔细识别：
   - 题目文字
   - 数字
   - 单位
   - 坐标
   - 图形
   - 方向
   - 电路
   - 受力关系
   - 长度
   - 角度
   - 图中标注

7. 如果图片中某些内容确实看不清楚，
   必须明确指出，不能自行编造。

8. 如果题目有多个小问，
   按顺序分别解答。

9. 特别检查：
   - 单位换算
   - 正负号
   - 方向
   - 有效数字
   - 公式适用条件

10. 最终答案采用这种结构：

### 解题过程

给出正式、清晰的解题过程。

### 最终答案

明确写出最终结果。

不要输出内部 reasoning_content。
""".strip()


# =========================================================
# OpenAI Compatible Client
# =========================================================

if API_KEY:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=300.0,
    )
else:
    client = None


# =========================================================
# 图片处理
# =========================================================

def image_to_data_uri(
    image_path: str,
) -> str:
    """
    将上传的图片转换成 Base64 Data URI。

    小图片直接发送。
    大图片自动缩放压缩。
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"找不到上传图片：{image_path}"
        )

    mime_type, _ = mimetypes.guess_type(
        path.name
    )

    mime_type = (
        mime_type
        or "image/jpeg"
    )

    raw = path.read_bytes()

    # 2MB 以下直接发送
    if (
        len(raw) <= 2 * 1024 * 1024
        and mime_type
        in {
            "image/jpeg",
            "image/png",
            "image/webp",
        }
    ):
        encoded = base64.b64encode(
            raw
        ).decode(
            "utf-8"
        )

        return (
            f"data:{mime_type};base64,"
            f"{encoded}"
        )

    # 大图片自动压缩
    with Image.open(path) as img:

        img = img.convert(
            "RGB"
        )

        max_side = 2000

        width, height = img.size

        scale = min(
            1.0,
            max_side
            / max(
                width,
                height,
            ),
        )

        if scale < 1.0:

            new_width = max(
                1,
                int(width * scale),
            )

            new_height = max(
                1,
                int(height * scale),
            )

            img = img.resize(
                (
                    new_width,
                    new_height,
                ),
                Image.Resampling.LANCZOS,
            )

        buffer = BytesIO()

        img.save(
            buffer,
            format="JPEG",
            quality=92,
            optimize=True,
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode(
            "utf-8"
        )

        return (
            "data:image/jpeg;base64,"
            + encoded
        )


# =========================================================
# 构造 Messages
# =========================================================

def build_messages(
    question: str,
    image_path: str | None,
) -> list[dict]:

    question = (
        question
        or ""
    ).strip()

    # 图片题
    if image_path:

        if question:
            user_text = question
        else:
            user_text = (
                "请识别图片中的物理题，"
                "直接给出清晰、完整、准确的"
                "正式解题过程和最终答案。"
            )

        content = [
            {
                "type": "text",
                "text": user_text,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_uri(
                        image_path
                    )
                },
            },
        ]

    # 纯文字题
    else:
        content = question

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": content,
        },
    ]


# =========================================================
# 流式文本切句
# =========================================================

_SENTENCE_END_RE = re.compile(
    r"[。！？!?；;]\s*|\n+"
)


def pop_ready_sentences(
    buffer: str,
    force: bool = False,
) -> tuple[list[str], str]:
    """
    将 Qwen 正式回答按照句子切开，
    完整句子立即交给 TTS。
    """

    ready = []

    cursor = 0

    # 优先按照句号、问号、感叹号、分号、换行切分
    for match in _SENTENCE_END_RE.finditer(
        buffer
    ):

        end = match.end()

        piece = buffer[
            cursor:end
        ].strip()

        cursor = end

        if piece:
            ready.append(
                piece
            )

    remainder = buffer[
        cursor:
    ]

    # 如果长时间没有句号，
    # 超过 120 字主动按照逗号等切开
    while len(remainder) > 120:

        cut = -1

        search_to = min(
            len(remainder),
            140,
        )

        for token in [
            "，",
            ",",
            "：",
            ":",
            "、",
            " ",
        ]:

            pos = remainder.rfind(
                token,
                60,
                search_to,
            )

            if pos > cut:
                cut = (
                    pos
                    + len(token)
                )

        if cut < 60:
            cut = 100

        piece = remainder[
            :cut
        ].strip()

        remainder = remainder[
            cut:
        ]

        if piece:
            ready.append(
                piece
            )

    # 回答结束后把剩余文字也送去朗读
    if (
        force
        and remainder.strip()
    ):
        ready.append(
            remainder.strip()
        )

        remainder = ""

    return (
        ready,
        remainder,
    )


# =========================================================
# TTS 文本清洗
# =========================================================

def clean_for_tts(
    text: str,
) -> str:
    """
    将 Markdown / LaTeX / 数学符号
    转换为更适合中文语音朗读的形式。
    """

    text = text.strip()

    if not text:
        return ""

    # Markdown 代码块
    text = re.sub(
        r"```.*?```",
        "",
        text,
        flags=re.S,
    )

    # 行内代码
    text = re.sub(
        r"`([^`]*)`",
        r"\1",
        text,
    )

    # Markdown 标题
    text = re.sub(
        r"^\s{0,3}#{1,6}\s*",
        "",
        text,
        flags=re.M,
    )

    # 粗体
    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text,
    )

    # 斜体
    text = re.sub(
        r"\*(.*?)\*",
        r"\1",
        text,
    )

    text = re.sub(
        r"__(.*?)__",
        r"\1",
        text,
    )

    # 列表
    text = re.sub(
        r"^\s*[-+*]\s+",
        "",
        text,
        flags=re.M,
    )

    # 数字列表
    text = re.sub(
        r"^\s*\d+[.)、]\s*",
        "",
        text,
        flags=re.M,
    )

    # LaTeX
    replacements = {
        "\\times": "乘以",
        "\\cdot": "乘以",
        "\\div": "除以",
        "\\approx": "约等于",
        "\\le": "小于等于",
        "\\ge": "大于等于",
        "\\Delta": "变化量",
        "\\theta": "西塔",
        "\\omega": "欧米伽",
        "\\alpha": "阿尔法",
        "\\beta": "贝塔",
        "\\pi": "派",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    # \frac{a}{b} -> a 除以 b
    frac_re = re.compile(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
    )

    for _ in range(4):

        new_text = frac_re.sub(
            r"\1 除以 \2",
            text,
        )

        if new_text == text:
            break

        text = new_text

    # 去除 LaTeX 外壳
    text = text.replace(
        "$$",
        "",
    )

    text = text.replace(
        "$",
        "",
    )

    text = text.replace(
        "\\(",
        "",
    )

    text = text.replace(
        "\\)",
        "",
    )

    text = text.replace(
        "\\[",
        "",
    )

    text = text.replace(
        "\\]",
        "",
    )

    # 删除剩余 LaTeX 命令
    text = re.sub(
        r"\\[a-zA-Z]+",
        "",
        text,
    )

    text = text.replace(
        "{",
        "",
    )

    text = text.replace(
        "}",
        "",
    )

    # 常用数学物理符号
    symbol_replacements = {
        "≈": "约等于",
        "≠": "不等于",
        "≤": "小于等于",
        "≥": "大于等于",
        "×": "乘以",
        "÷": "除以",
        "→": "得到",
        "⇒": "因此得到",
        "√": "根号",
        "Δ": "变化量",
        "∞": "无穷大",
        "°": "度",
        "²": "的平方",
        "³": "的立方",
    }

    for old, new in symbol_replacements.items():
        text = text.replace(
            old,
            new,
        )

    text = text.replace(
        "=",
        "等于",
    )

    # Markdown 图片
    text = re.sub(
        r"!\[[^\]]*\]\([^)]*\)",
        "",
        text,
    )

    # Markdown 链接
    text = re.sub(
        r"\[([^\]]+)\]\([^)]*\)",
        r"\1",
        text,
    )

    # URL 不朗读
    text = re.sub(
        r"https?://\S+",
        "",
        text,
    )

    # 压缩空格
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# =========================================================
# Qwen 工作线程
# =========================================================

def llm_worker(
    messages: list[dict],
    event_queue: queue.Queue,
    tts_queue: queue.Queue,
) -> None:
    """
    Qwen 流式生成。

    reasoning_content：
    完全隐藏。

    content：
    才发送给用户。
    """

    try:

        assert client is not None

        completion = (
            client
            .chat
            .completions
            .create(
                model=MODEL,
                messages=messages,

                # 保留 Qwen 深度思考
                extra_body={
                    "enable_thinking": True,
                },

                stream=True,
            )
        )

        tts_buffer = ""

        reasoning_seen = False
        answer_seen = False

        for chunk in completion:

            if not chunk.choices:
                continue

            delta = (
                chunk
                .choices[0]
                .delta
            )

            # =============================================
            # 内部 reasoning
            #
            # 只检测模型是否正在推理
            # 不展示任何内容
            # =============================================

            reasoning = getattr(
                delta,
                "reasoning_content",
                None,
            )

            if (
                reasoning
                and not reasoning_seen
            ):
                reasoning_seen = True

                event_queue.put(
                    (
                        "status",
                        "🧠 正在分析题目…",
                    )
                )

            # =============================================
            # 正式答案
            # =============================================

            content = getattr(
                delta,
                "content",
                None,
            )

            if not content:
                continue

            if not answer_seen:

                answer_seen = True

                event_queue.put(
                    (
                        "status",
                        "✍️ 正在生成解答，并同步准备语音…",
                    )
                )

            # 正式文本发送给界面
            event_queue.put(
                (
                    "text",
                    content,
                )
            )

            # 同时进入 TTS 缓冲
            tts_buffer += content

            ready, tts_buffer = (
                pop_ready_sentences(
                    tts_buffer,
                    force=False,
                )
            )

            # 已经完整的句子立即送入 TTS
            for sentence in ready:

                spoken = clean_for_tts(
                    sentence
                )

                if len(spoken) >= 2:

                    tts_queue.put(
                        spoken
                    )

        # =============================================
        # 模型输出结束
        # =============================================

        ready, _ = (
            pop_ready_sentences(
                tts_buffer,
                force=True,
            )
        )

        for sentence in ready:

            spoken = clean_for_tts(
                sentence
            )

            if len(spoken) >= 2:

                tts_queue.put(
                    spoken
                )

        # 通知 TTS：
        # 已经不会有新的句子
        tts_queue.put(
            None
        )

        event_queue.put(
            (
                "llm_done",
                None,
            )
        )

    except Exception as exc:

        try:
            tts_queue.put(
                None
            )
        except Exception:
            pass

        event_queue.put(
            (
                "error",
                (
                    "模型调用失败："
                    f"{exc}"
                ),
            )
        )

        event_queue.put(
            (
                "llm_done",
                None,
            )
        )


# =========================================================
# Edge TTS 工作线程
# =========================================================

def tts_worker(
    voice: str,
    tts_queue: queue.Queue,
    event_queue: queue.Queue,
) -> None:
    """
    独立 TTS 线程。

    Qwen 可以继续生成后续内容，
    TTS 同时朗读前面的完整句子。
    """

    try:

        first_audio = True

        while True:

            sentence = (
                tts_queue.get()
            )

            # Qwen 已结束
            if sentence is None:
                break

            try:

                communicate = (
                    edge_tts.Communicate(
                        text=sentence,
                        voice=voice,
                        rate="+0%",
                        volume="+0%",
                        pitch="+0Hz",
                    )
                )

                audio_buffer = (
                    bytearray()
                )

                # Edge TTS 流式输出
                for chunk in (
                    communicate
                    .stream_sync()
                ):

                    if (
                        chunk.get("type")
                        != "audio"
                    ):
                        continue

                    data = chunk.get(
                        "data"
                    )

                    if not data:
                        continue

                    audio_buffer.extend(
                        data
                    )

                    # 累积约 12KB 后推送
                    if (
                        len(audio_buffer)
                        >= 12_000
                    ):

                        if first_audio:

                            first_audio = False

                            event_queue.put(
                                (
                                    "status",
                                    "🔊 已开始实时朗读，后续答案仍在生成…",
                                )
                            )

                        event_queue.put(
                            (
                                "audio",
                                bytes(
                                    audio_buffer
                                ),
                            )
                        )

                        audio_buffer.clear()

                # 当前句子的剩余音频
                if audio_buffer:

                    if first_audio:

                        first_audio = False

                        event_queue.put(
                            (
                                "status",
                                "🔊 已开始实时朗读，后续答案仍在生成…",
                            )
                        )

                    event_queue.put(
                        (
                            "audio",
                            bytes(
                                audio_buffer
                            ),
                        )
                    )

            except Exception as exc:

                # 单句 TTS 出错不影响文字回答
                event_queue.put(
                    (
                        "tts_warning",
                        (
                            "语音生成出现异常："
                            f"{exc}"
                        ),
                    )
                )

    finally:

        event_queue.put(
            (
                "tts_done",
                None,
            )
        )


# =========================================================
# Gradio 主函数
# =========================================================

def solve_stream(
    question: str,
    image_path: str | None,
    voice: str,
    progress=gr.Progress(
        track_tqdm=False
    ),
) -> Iterator[tuple]:

    question = (
        question
        or ""
    ).strip()

    # =============================================
    # 没有题目
    # =============================================

    if (
        not question
        and not image_path
    ):

        yield (
            "请先输入物理题，或者上传物理题图片。",
            gr.skip(),
            "⚠️ 尚未提交题目",
        )

        return

    # =============================================
    # API KEY
    # =============================================

    if (
        not API_KEY
        or client is None
    ):

        yield (
            (
                "未检测到 `DASHSCOPE_API_KEY`。\n\n"
                "请检查 `run2.py` 同目录下的 `.env` 文件。"
            ),
            gr.skip(),
            "❌ API Key 未配置",
        )

        return

    # =============================================
    # 进度
    # =============================================

    progress(
        0.03,
        desc="正在接收题目",
    )

    yield (
        "",
        gr.skip(),
        "📥 已收到题目，正在准备…",
    )

    # =============================================
    # 构造请求
    # =============================================

    try:

        progress(
            0.10,
            desc="正在处理题目",
        )

        messages = build_messages(
            question,
            image_path,
        )

    except Exception as exc:

        yield (
            (
                "题目或图片处理失败："
                f"{exc}"
            ),
            gr.skip(),
            "❌ 图片处理失败",
        )

        return

    progress(
        0.20,
        desc="正在连接 Qwen",
    )

    # =============================================
    # Queue
    # =============================================

    event_queue: queue.Queue = (
        queue.Queue()
    )

    tts_queue: queue.Queue = (
        queue.Queue()
    )

    # =============================================
    # Qwen Thread
    # =============================================

    llm_thread = threading.Thread(
        target=llm_worker,
        args=(
            messages,
            event_queue,
            tts_queue,
        ),
        daemon=True,
    )

    # =============================================
    # TTS Thread
    # =============================================

    tts_thread = threading.Thread(
        target=tts_worker,
        args=(
            voice
            or DEFAULT_VOICE,
            tts_queue,
            event_queue,
        ),
        daemon=True,
    )

    # 同时启动
    llm_thread.start()
    tts_thread.start()

    answer_text = ""

    llm_done = False
    tts_done = False

    last_status = (
        "🧠 正在分析题目…"
    )

    tts_warning = ""

    progress(
        0.32,
        desc="模型正在分析",
    )

    yield (
        gr.skip(),
        gr.skip(),
        last_status,
    )

    # =============================================
    # 主事件循环
    # =============================================

    while not (
        llm_done
        and tts_done
    ):

        try:

            event_type, payload = (
                event_queue.get(
                    timeout=0.5
                )
            )

        except queue.Empty:

            if not llm_thread.is_alive():
                llm_done = True

            if not tts_thread.is_alive():
                tts_done = True

            continue

        # =========================================
        # 状态
        # =========================================

        if event_type == "status":

            last_status = str(
                payload
            )

            yield (
                gr.skip(),
                gr.skip(),
                last_status,
            )

        # =========================================
        # 正式答案文本
        # =========================================

        elif event_type == "text":

            answer_text += str(
                payload
            )

            visible_progress = min(
                0.88,
                0.45
                + (
                    len(answer_text)
                    / 6000.0
                )
                * 0.43,
            )

            progress(
                visible_progress,
                desc="正在生成解答",
            )

            yield (
                answer_text,
                gr.skip(),
                last_status,
            )

        # =========================================
        # 流式音频
        # =========================================

        elif event_type == "audio":

            progress(
                0.90,
                desc="正在生成语音",
            )

            yield (
                gr.skip(),
                payload,
                last_status,
            )

        # =========================================
        # TTS 警告
        # =========================================

        elif (
            event_type
            == "tts_warning"
        ):

            tts_warning = str(
                payload
            )

            yield (
                gr.skip(),
                gr.skip(),
                (
                    "⚠️ "
                    + tts_warning
                ),
            )

        # =========================================
        # Qwen 错误
        # =========================================

        elif event_type == "error":

            err = str(
                payload
            )

            if answer_text:

                answer_text += (
                    "\n\n"
                    "> ⚠️ "
                    + err
                )

            else:
                answer_text = err

            last_status = (
                "❌ 模型调用出现错误"
            )

            yield (
                answer_text,
                gr.skip(),
                last_status,
            )

        # =========================================
        # Qwen 完成
        # =========================================

        elif (
            event_type
            == "llm_done"
        ):

            llm_done = True

            progress(
                0.94,
                desc="文字答案已完成",
            )

        # =========================================
        # TTS 完成
        # =========================================

        elif (
            event_type
            == "tts_done"
        ):

            tts_done = True

            progress(
                0.98,
                desc="正在完成语音",
            )

    # =============================================
    # 全部完成
    # =============================================

    progress(
        1.0,
        desc="完成",
    )

    if answer_text:

        if tts_warning:

            final_status = (
                "✅ 解题完成；"
                "文字答案正常，"
                "部分语音片段可能生成失败"
            )

        else:

            final_status = (
                "✅ 解题完成"
            )

        yield (
            answer_text,
            gr.skip(),
            final_status,
        )

    else:

        yield (
            (
                "没有获得有效答案，"
                "请重新提交。"
            ),
            gr.skip(),
            "⚠️ 未获得有效答案",
        )


# =========================================================
# CSS
#
# Gradio 6.x：
# 不再放入 gr.Blocks(css=...)
# 而是在 launch(css=CSS)
# =========================================================

CSS = """
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    background: #f7f8fa !important;
}

#hero {
    padding: 28px 8px 14px 8px;
}

#hero h1 {
    font-size: 31px;
    line-height: 1.2;
    margin: 0 0 9px 0;
    letter-spacing: -0.5px;
    font-weight: 700;
}

#hero p {
    margin: 0;
    color: #6b7280;
    font-size: 15px;
}

.panel-card {
    background: white;
    border: 1px solid rgba(0, 0, 0, 0.06);
    border-radius: 18px;
    padding: 12px;
    box-shadow:
        0 8px 28px
        rgba(0, 0, 0, 0.045);
}

#submit-btn {
    min-height: 46px;
    border-radius: 12px;
    font-weight: 650;
}

#clear-btn {
    min-height: 46px;
    border-radius: 12px;
}

#status-box {
    min-height: 38px;
    padding: 4px 4px;
    color: #4b5563;
}

#answer-box {
    min-height: 500px;
}

#answer-box .prose {
    max-width: none !important;
}

footer {
    display: none !important;
}
"""


# =========================================================
# Gradio UI
# =========================================================
#
# Gradio 6.x：
# Blocks 中不再传 css=CSS
# =========================================================

with gr.Blocks(
    title="AI 物理解题",
) as demo:

    # =====================================================
    # 顶部
    # =====================================================

    gr.HTML(
        """
        <div id="hero">
            <h1>AI 物理解题</h1>

            <p>
                输入物理题或上传题目图片，
                AI 自动分析、解答并同步语音朗读。
            </p>
        </div>
        """
    )

    # =====================================================
    # 主体
    # =====================================================

    with gr.Row(
        equal_height=False
    ):

        # =================================================
        # 左侧
        # =================================================

        with gr.Column(
            scale=5,
            min_width=360,
            elem_classes=[
                "panel-card"
            ],
        ):

            gr.Markdown(
                "### 题目"
            )

            # Gradio 6.x：
            # 删除 show_copy_button=False
            question_box = gr.Textbox(
                label="文字题目",
                placeholder=(
                    "在这里输入物理题。\n\n"
                    "也可以直接上传题目图片。"
                ),
                lines=8,
                max_lines=16,
            )

            image_input = gr.Image(
                label="题目图片",
                type="filepath",
                sources=[
                    "upload",
                    "clipboard",
                ],
                height=280,
            )

            voice_selector = gr.Dropdown(
                choices=VOICE_OPTIONS,
                value=DEFAULT_VOICE,
                label="朗读声音",
                interactive=True,
            )

            with gr.Row():

                submit_btn = gr.Button(
                    "提交解题",
                    variant="primary",
                    elem_id="submit-btn",
                    scale=3,
                )

                clear_btn = gr.Button(
                    "清空",
                    variant="secondary",
                    elem_id="clear-btn",
                    scale=1,
                )

            status_md = gr.Markdown(
                "准备就绪。",
                elem_id="status-box",
            )

        # =================================================
        # 右侧
        # =================================================

        with gr.Column(
            scale=7,
            min_width=460,
            elem_classes=[
                "panel-card"
            ],
        ):

            gr.Markdown(
                "### 解答"
            )

            answer_md = gr.Markdown(
                value="",
                elem_id="answer-box",
                latex_delimiters=[
                    {
                        "left": "$$",
                        "right": "$$",
                        "display": True,
                    },
                    {
                        "left": "$",
                        "right": "$",
                        "display": False,
                    },
                    {
                        "left": "\\[",
                        "right": "\\]",
                        "display": True,
                    },
                    {
                        "left": "\\(",
                        "right": "\\)",
                        "display": False,
                    },
                ],
            )

            gr.Markdown(
                "### 实时朗读"
            )

            audio_output = gr.Audio(
                label="语音",
                streaming=True,
                autoplay=True,
                interactive=False,
            )

    # =====================================================
    # 提交
    # =====================================================

    submit_btn.click(
        fn=solve_stream,
        inputs=[
            question_box,
            image_input,
            voice_selector,
        ],
        outputs=[
            answer_md,
            audio_output,
            status_md,
        ],
        show_progress="full",
    )

    # =====================================================
    # 输入框 Enter 提交
    # =====================================================

    question_box.submit(
        fn=solve_stream,
        inputs=[
            question_box,
            image_input,
            voice_selector,
        ],
        outputs=[
            answer_md,
            audio_output,
            status_md,
        ],
        show_progress="full",
    )

    # =====================================================
    # 清空
    # =====================================================

    clear_btn.click(
        fn=lambda: (
            "",
            None,
            "",
            None,
            "准备就绪。",
        ),
        inputs=None,
        outputs=[
            question_box,
            image_input,
            answer_md,
            audio_output,
            status_md,
        ],
        queue=False,
    )


# =========================================================
# 启动
# =========================================================

if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=4
    ).launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=False,
        inbrowser=True,
        show_error=True,

        # Gradio 6.x：
        # CSS 放到 launch()
        css=CSS,
    )
