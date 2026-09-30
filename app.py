import base64
import mimetypes
import os
from io import BytesIO
from pathlib import Path

import gradio as gr
from openai import OpenAI
from PIL import Image, ImageOps
from dotenv import load_dotenv


# =========================
# 环境变量
# =========================
# 自动读取当前项目目录下的 .env 文件。
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


# =========================
# 基础配置
# =========================
MODEL_NAME = os.getenv("QWEN_MODEL", "qwen3.7-flash")
BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://llm-0nr9mznstg4wrz4h.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
PORT = int(os.getenv("PORT", "7862"))

SYSTEM_PROMPT = r"""
你是一名严谨、耐心、擅长中学和大学基础物理的教师。
你的任务是准确解答用户输入或图片中的物理题。

请遵循以下规则：
1. 先准确理解题意；如果是图片，先识别题干、图示、已知量、单位、选项和要求。
2. 不要凭空补充图片里看不清或题目未提供的关键条件；若关键条件无法识别，要明确指出。
3. 解题时优先按“题意理解 → 已知量 → 求解目标 → 物理规律/公式 → 推导与代入 → 结果 → 验算/说明”的顺序组织。
4. 所有物理量尽量标明单位；涉及矢量时说明方向；涉及正负号时解释符号含义。
5. 公式推导要完整，但避免无关冗长内容。
6. 数值计算要检查数量级、单位和有效数字，避免低级算术错误。
7. 对选择题，最后明确给出选项，并解释其他关键选项为何不成立（如有必要）。
8. 对证明题、推导题，要给出逻辑完整的推导链条。
9. 若存在多种方法，先给出最直接、最标准的方法；必要时再补充另一种方法。
10. 使用清晰的 Markdown 排版；数学公式尽量使用 LaTeX。
11. 最后一节使用“最终答案”明确给出结论。
12. 如果题目本身条件矛盾、缺失或存在歧义，要指出问题，不要强行给出唯一答案。

回答语言默认使用简体中文。
""".strip()


def get_client() -> OpenAI:
    """延迟创建客户端，避免未配置 Key 时应用直接启动失败。"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            f"未检测到 DASHSCOPE_API_KEY。请检查 {ENV_PATH} 文件中的配置。"
        )

    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=180.0,
        max_retries=2,
    )


def _encode_bytes(data: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def image_to_data_url(image_path: str) -> str:
    """
    将 Gradio 上传的图片转换为 Base64 Data URI。

    普通图片尽量保持原图；当图片过大时进行压缩，降低接口请求体过大的概率。
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在：{image_path}")

    raw = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"

    # Base64 会比原文件大约增加 1/3。为了给 Data URI 留余量，
    # 原图超过约 13 MB 时做一次压缩/缩放。
    if len(raw) <= 13 * 1024 * 1024:
        return _encode_bytes(raw, mime_type)

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")

        # 保留公式、图线和小字细节，同时避免超大照片导致请求体过大。
        max_side = 3200
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92, optimize=True)
        compressed = buf.getvalue()

    return _encode_bytes(compressed, "image/jpeg")


def build_user_content(question: str, image_path: str | None):
    question = (question or "").strip()

    if not image_path:
        return question

    if not question:
        question = (
            "请完整识别这张图片中的物理题，并给出清晰、完整、准确的解答。"
            "如果图片中有示意图、坐标图、受力图、电路图或几何关系，请把它们纳入分析。"
        )
    else:
        question += (
            "\n\n请同时结合我上传的图片作答。若文字描述与图片信息有关，请以完整题意为准。"
        )

    return [
        {"type": "text", "text": question},
        {
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(image_path),
            },
        },
    ]


def solve_physics(question: str, image_path: str | None):
    """Gradio 流式解题函数，只展示最终答案，不展示模型内部思考内容。"""
    question = (question or "").strip()

    if not question and not image_path:
        yield "⚠️ 请先输入物理题，或者上传一张物理题图片。"
        return

    try:
        client = get_client()
        user_content = build_user_content(question, image_path)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_body={"enable_thinking": True},
            stream=True,
            max_tokens=8192,
        )

        answer = ""
        received_content = False

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)

            # reasoning_content 属于模型内部推理过程，这里不显示，只呈现最终回答。
            if content:
                received_content = True
                answer += content
                yield answer

        if not received_content:
            yield "⚠️ 模型没有返回可显示的最终答案，请稍后重试。"

    except Exception as exc:
        error_text = str(exc)
        if "401" in error_text or "Unauthorized" in error_text:
            message = (
                "❌ API 鉴权失败。请检查 DASHSCOPE_API_KEY 是否正确、是否已失效，"
                "以及该 Key 是否属于当前百炼业务空间。"
            )
        elif "model" in error_text.lower() and (
            "not found" in error_text.lower() or "does not exist" in error_text.lower()
        ):
            message = (
                f"❌ 模型 `{MODEL_NAME}` 当前不可用。请确认该百炼业务空间已经开通/部署此模型。\n\n"
                f"原始错误：`{error_text}`"
            )
        else:
            message = f"❌ 调用失败：\n\n```text\n{error_text}\n```"

        yield message


def clear_all():
    return "", None, ""


with gr.Blocks(title="Qwen 物理解题助手") as demo:
    gr.Markdown(
        """
# Qwen 物理解题助手

支持两种输入方式：
- **直接输入物理题文字**
- **上传物理题图片**（可同时补充文字要求）

模型会识别题意、公式、图示和单位，并给出完整解题过程与最终答案。
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                label="物理题 / 补充要求",
                placeholder=(
                    "例如：一个质量为 2 kg 的物体，在 10 N 水平恒力作用下……\n\n"
                    "如果已经上传图片，也可以在这里输入：请重点解释第 3 问。"
                ),
                lines=12,
            )

            image_input = gr.Image(
                label="上传物理题图片（可选）",
                type="filepath",
                sources=["upload"],
            )

            with gr.Row():
                submit_btn = gr.Button("提交解题", variant="primary")
                clear_btn = gr.Button("清空")

        with gr.Column(scale=1):
            answer_output = gr.Markdown(label="解答")

    submit_btn.click(
        fn=solve_physics,
        inputs=[question_input, image_input],
        outputs=answer_output,
    )

    question_input.submit(
        fn=solve_physics,
        inputs=[question_input, image_input],
        outputs=answer_output,
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[question_input, image_input, answer_output],
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Qwen 物理解题助手")
    print(f"Model    : {MODEL_NAME}")
    print(f"Base URL : {BASE_URL}")
    print(f"Web UI   : http://127.0.0.1:{PORT}")
    print("=" * 70)

    demo.queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1",
        server_port=PORT,
        share=False,
        inbrowser=True,
        show_error=True,
    )
