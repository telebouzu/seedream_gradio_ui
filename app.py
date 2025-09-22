"""Seedream 4.0 image generation and editing UI using fal.ai and Gradio."""
from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Sequence, Tuple

import fal_client
import gradio as gr
import requests
from PIL import Image

# Preset aspect ratios derived from the reference image shared in the request.
# Users can still opt in for a fully custom size by choosing the "カスタムサイズ" entry.
ASPECT_RATIO_PRESETS: OrderedDict[str, Tuple[int, int]] = OrderedDict(
    [
        ("スクエア 1:1 (1024x1024)", (1024, 1024)),
        ("ポートレート 3:4 (768x1024)", (768, 1024)),
        ("ポートレート 9:16 (720x1280)", (720, 1280)),
        ("ランドスケープ 4:3 (1024x768)", (1024, 768)),
        ("ランドスケープ 16:9 (1280x720)", (1280, 720)),
        ("シネマティック 21:9 (1536x656)", (1536, 656)),
    ]
)
CUSTOM_ASPECT_OPTION = "カスタムサイズ"
DEFAULT_ASPECT_LABEL = next(iter(ASPECT_RATIO_PRESETS))
DEFAULT_CUSTOM_DIMENSION = 1024
MAX_IMAGES = 4


@dataclass
class GenerationResult:
    """Container for gallery data and downloadable image paths."""

    gallery_entries: List[Tuple[Image.Image, str]]
    download_paths: List[str]
    mode: str
    width: int
    height: int


def _ensure_multiple_of_eight(value: int) -> int:
    """Adjust dimension to the closest multiple of 8 (Seedream requires this)."""

    if value < 64:
        return 64
    remainder = value % 8
    if remainder == 0:
        return value
    # Round to nearest multiple of 8 to avoid large jumps.
    down = value - remainder
    up = down + 8
    if value - down < up - value:
        return down
    return up


def _prepare_dimensions(aspect_label: str, width: float | int, height: float | int) -> Tuple[int, int]:
    if aspect_label != CUSTOM_ASPECT_OPTION and aspect_label in ASPECT_RATIO_PRESETS:
        return ASPECT_RATIO_PRESETS[aspect_label]

    # Custom dimensions can be freely adjusted from the UI.
    width_int = int(width) if width else DEFAULT_CUSTOM_DIMENSION
    height_int = int(height) if height else DEFAULT_CUSTOM_DIMENSION
    return _ensure_multiple_of_eight(width_int), _ensure_multiple_of_eight(height_int)


def _download_image(url: str, index: int) -> Tuple[Image.Image, str]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(temp_file, format="PNG")
    temp_file.close()
    return image, temp_file.name


def _extract_image_urls(result_payload: dict) -> List[str]:
    images: List[str] = []
    if not result_payload:
        return images

    if isinstance(result_payload, dict):
        candidates = []
        if "images" in result_payload and isinstance(result_payload["images"], Sequence):
            candidates = result_payload["images"]
        elif "output" in result_payload and isinstance(result_payload["output"], dict):
            output_section = result_payload["output"]
            if isinstance(output_section.get("images"), Sequence):
                candidates = output_section["images"]

        for item in candidates:
            if isinstance(item, dict):
                for key in ("url", "image", "path", "src"):
                    if key in item and item[key]:
                        images.append(str(item[key]))
                        break
            elif isinstance(item, str):
                images.append(item)

    return images


def call_seedream(
    api_key: str,
    prompt: str,
    reference_image_path: Optional[str],
    aspect_label: str,
    width: float,
    height: float,
    num_images: float,
    safety_filter: bool,
    progress: gr.Progress | None = None,
) -> GenerationResult:
    if not api_key:
        raise gr.Error("先にfal.aiのAPIキーを設定してください。")

    prompt_text = (prompt or "").strip()
    if not prompt_text:
        raise gr.Error("プロンプトを入力してください。")

    resolved_width, resolved_height = _prepare_dimensions(aspect_label, width, height)
    image_count = max(1, min(int(num_images), MAX_IMAGES))

    if progress is None:
        progress = gr.Progress()

    progress(0.1, desc="Seedream 4.0へリクエストを準備中…")

    client = fal_client.SyncClient(key=api_key.strip())

    request_arguments = {
        "prompt": prompt_text,
        "num_images": image_count,
        "width": resolved_width,
        "height": resolved_height,
        # Both flags are provided because API revisions have used different keys in the past.
        "enable_safety_checker": bool(safety_filter),
        "safety_checker": "enable" if safety_filter else "disable",
    }

    endpoint = "fal-ai/bytedance/seedream/v4/text-to-image"
    mode = "text-to-image"

    if reference_image_path:
        progress(0.2, desc="参照画像をアップロード中…")
        upload_url = client.upload_file(reference_image_path)
        request_arguments.update(
            {
                "image_url": upload_url,
                # Strength balances between preserving the input and applying the prompt.
                "strength": 0.7,
            }
        )
        endpoint = "fal-ai/bytedance/seedream/v4/edit"
        mode = "image-to-image"

    progress(0.35, desc="Seedream 4.0に送信しています…")
    request_handle = client.submit(endpoint, arguments=request_arguments)

    progress(0.6, desc="生成結果を待機しています…")
    result_payload = request_handle.get()

    image_urls = _extract_image_urls(result_payload)
    if not image_urls:
        raise gr.Error("Seedream 4.0から画像URLを取得できませんでした。")

    gallery_entries: List[Tuple[Image.Image, str]] = []
    download_paths: List[str] = []

    for index, url in enumerate(image_urls):
        progress(0.7 + 0.3 * (index / max(len(image_urls), 1)), desc=f"画像 {index + 1} をダウンロード中…")
        image, file_path = _download_image(url, index)
        gallery_entries.append((image, f"{mode.title()} #{index + 1}"))
        download_paths.append(file_path)

    progress(1.0, desc="完了しました")
    return GenerationResult(
        gallery_entries=gallery_entries,
        download_paths=download_paths,
        mode=mode,
        width=resolved_width,
        height=resolved_height,
    )


def on_generate(
    api_key: str,
    prompt: str,
    reference_image_path: Optional[str],
    aspect_label: str,
    width: float,
    height: float,
    num_images: float,
    safety_filter: bool,
):
    progress = gr.Progress(track_tqdm=False)
    try:
        result = call_seedream(
            api_key=api_key,
            prompt=prompt,
            reference_image_path=reference_image_path,
            aspect_label=aspect_label,
            width=width,
            height=height,
            num_images=num_images,
            safety_filter=safety_filter,
            progress=progress,
        )
    except gr.Error:
        raise
    except Exception as exc:  # noqa: BLE001 - surfacing errors to the UI is important.
        return (
            [],
            f"⚠️ エラーが発生しました: {exc}",
            [],
            None,
            gr.update(value=None, visible=False),
        )

    status_message = (
        f"Seedream 4.0 の **{('image-to-image' if result.mode == 'image-to-image' else 'text-to-image')}** モードで"
        f" {len(result.gallery_entries)} 枚生成しました。\n"
        f"解像度: {result.width} × {result.height}px"
    )

    return (
        result.gallery_entries,
        status_message,
        result.download_paths,
        None,
        gr.update(value=None, visible=False, label="選択した画像をダウンロード"),
    )


def update_dimensions(aspect_label: str, current_width: float, current_height: float):
    if aspect_label != CUSTOM_ASPECT_OPTION and aspect_label in ASPECT_RATIO_PRESETS:
        width, height = ASPECT_RATIO_PRESETS[aspect_label]
        return (
            gr.update(value=width, interactive=False),
            gr.update(value=height, interactive=False),
        )

    width_value = int(current_width or DEFAULT_CUSTOM_DIMENSION)
    height_value = int(current_height or DEFAULT_CUSTOM_DIMENSION)
    width_value = _ensure_multiple_of_eight(width_value)
    height_value = _ensure_multiple_of_eight(height_value)
    return (
        gr.update(value=width_value, interactive=True),
        gr.update(value=height_value, interactive=True),
    )


def on_gallery_select(evt: gr.SelectData, download_paths: List[str]):
    index = evt.index if evt is not None else None
    if index is None or download_paths is None or index >= len(download_paths):
        return None, gr.update(value=None, visible=False)

    selected_path = download_paths[index]
    file_name = os.path.basename(selected_path)
    return index, gr.update(value=selected_path, visible=True, label=f"選択中: {file_name}")


def set_api_key(new_key: str):
    key = (new_key or "").strip()
    if not key:
        raise gr.Error("fal.aiで発行したAPIキーを入力してください。")

    return (
        key,
        gr.update(value="APIキーを保存しました。", visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=""),
    )


def reveal_api_key_form():
    return (
        gr.update(value="APIキーを再設定してください。", visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Seedream 4.0 Generator", theme=gr.themes.Soft()) as demo:
        api_key_state = gr.State("")
        download_state = gr.State([])
        selected_index_state = gr.State(None)

        gr.Markdown(
            """
            # Seedream 4.0 画像生成 & 編集ツール
            fal.ai の Seedream 4.0 API を使ってプロンプトから画像生成、もしくは参照画像を使った編集を行います。
            まずは fal.ai の [ダッシュボード](https://fal.ai) で発行した API キーを登録してください。
            """
        )

        with gr.Accordion("fal.ai API キー", open=True, visible=True) as api_key_accordion:
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="APIキー",
                    type="password",
                    placeholder="fal.aiのAPIキーを貼り付けてください",
                )
                save_api_key_button = gr.Button("保存", variant="primary")
        api_key_message = gr.Markdown("", visible=False)
        change_api_key_button = gr.Button("APIキーを変更する", visible=False)

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.TextArea(
                    label="プロンプト",
                    placeholder="生成したいイメージを日本語でも英語でも自由に入力してください",
                    lines=5,
                )

                with gr.Accordion("参照画像 (任意)", open=True):
                    gr.Markdown(
                        "参照画像をドロップすると自動的に Seedream 4.0 の image-to-image モードで処理されます。"
                    )
                    reference_image_input = gr.Image(
                        label="参照画像", type="filepath", image_mode="RGB", height=256
                    )

                with gr.Accordion("生成パラメータ", open=False):
                    aspect_dropdown = gr.Dropdown(
                        label="アスペクト比プリセット",
                        choices=list(ASPECT_RATIO_PRESETS.keys()) + [CUSTOM_ASPECT_OPTION],
                        value=DEFAULT_ASPECT_LABEL,
                    )
                    width_input = gr.Number(
                        label="幅 (px)",
                        value=ASPECT_RATIO_PRESETS[DEFAULT_ASPECT_LABEL][0],
                        minimum=64,
                        step=8,
                        precision=0,
                        interactive=False,
                    )
                    height_input = gr.Number(
                        label="高さ (px)",
                        value=ASPECT_RATIO_PRESETS[DEFAULT_ASPECT_LABEL][1],
                        minimum=64,
                        step=8,
                        precision=0,
                        interactive=False,
                    )
                    num_images_input = gr.Slider(
                        label="生成する枚数",
                        minimum=1,
                        maximum=MAX_IMAGES,
                        step=1,
                        value=1,
                        info="最大4枚まで同時生成できます",
                    )
                    safety_checkbox = gr.Checkbox(
                        label="Safetyオプションを有効化",
                        value=False,
                        info="デフォルトではオフです。必要に応じてチェックしてください。",
                    )

                generate_button = gr.Button("画像を生成", variant="primary")

            with gr.Column(scale=3):
                gallery_output = gr.Gallery(
                    label="生成結果",
                    columns=2,
                    height="auto",
                    allow_preview=True,
                    show_label=True,
                    show_download_button=False,
                    type="pil",
                )
                status_output = gr.Markdown("")
                download_button = gr.DownloadButton(
                    label="選択した画像をダウンロード",
                    visible=False,
                    icon="📥",
                )

        save_api_key_button.click(
            set_api_key,
            inputs=[api_key_input],
            outputs=[api_key_state, api_key_message, change_api_key_button, api_key_accordion, api_key_input],
        )
        change_api_key_button.click(
            reveal_api_key_form,
            outputs=[api_key_message, change_api_key_button, api_key_accordion],
        )

        aspect_dropdown.change(
            update_dimensions,
            inputs=[aspect_dropdown, width_input, height_input],
            outputs=[width_input, height_input],
        )

        generate_button.click(
            on_generate,
            inputs=[
                api_key_state,
                prompt_input,
                reference_image_input,
                aspect_dropdown,
                width_input,
                height_input,
                num_images_input,
                safety_checkbox,
            ],
            outputs=[
                gallery_output,
                status_output,
                download_state,
                selected_index_state,
                download_button,
            ],
        )

        gallery_output.select(
            on_gallery_select,
            inputs=[download_state],
            outputs=[selected_index_state, download_button],
        )

        gr.Markdown(
            """
            ### 使い方のヒント
            - 参照画像を指定しない場合は完全なテキストからの生成になります。
            - アスペクト比プリセットを選ぶと推奨サイズが自動で入力されます。"カスタムサイズ"を選ぶと任意の値を設定できます。
            - Safetyオプションは初期状態で無効化されています。必要に応じて切り替えてください。
            - ギャラリー内の画像をクリックするとダウンロードボタンが有効化され、個別にダウンロードできます。
            """
        )

        demo.queue()
    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
