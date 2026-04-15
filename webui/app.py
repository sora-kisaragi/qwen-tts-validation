"""
Qwen3-TTS Gradio WebUI

- 目的: FastAPI サーバーを経由して全 TTS 組み合わせを操作できる WebUI を提供する。
        4 タブ構成: Voice Clone / Custom Voice / Voice Design / プロファイル管理
- 対象: Issue #33 — Gradio WebUI 実装, Issue #47 — WebUI 話者プロファイル管理
- 関連: docs/v2-design.md — WebUI タブ設計
         api/main.py — 呼び出す API サーバー

作成者: 宗廣 颯真
作成日: 2026-04-14
最終更新者: 宗廣 颯真
最終更新日: 2026-04-15

Usage:
    # docker compose で起動（API サーバーと同時に立ち上げる）
    docker compose up api webui

    # 単体で起動（API が別途起動済みの場合）
    TTS_API_URL=http://localhost:7865 python3 webui/app.py
"""

import logging
import os
import pathlib
import sys
import tempfile

import gradio as gr
import requests

# scripts/ を sys.path に追加して audio_utils を利用する
_SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from audio_utils import ensure_wav_format  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# API サーバーの URL（環境変数で上書き可能）
API_BASE = os.environ.get("TTS_API_URL", "http://localhost:7865")

SPEAKER_PROFILES_DIR = pathlib.Path("/workspace/speaker_profiles")


# ─── API クライアントヘルパー ──────────────────────────────────────────────────


def _fetch_speakers() -> list[str]:
    """API から利用可能な話者一覧を取得する。接続失敗時は空リストを返す。"""
    try:
        resp = requests.get(f"{API_BASE}/tts/speakers", timeout=5)
        resp.raise_for_status()
        return resp.json().get("speakers", [])
    except Exception as exc:
        logger.warning("Failed to fetch speakers: %s", exc)
        return ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]


def _fetch_languages() -> list[str]:
    """API から対応言語一覧を取得する。接続失敗時はデフォルト値を返す。"""
    try:
        resp = requests.get(f"{API_BASE}/tts/languages", timeout=5)
        resp.raise_for_status()
        return resp.json().get("languages", [])
    except Exception as exc:
        logger.warning("Failed to fetch languages: %s", exc)
        return [
            "auto",
            "japanese",
            "english",
            "chinese",
            "korean",
            "french",
            "german",
            "italian",
            "portuguese",
            "russian",
            "spanish",
        ]


def _list_profiles() -> list[str]:
    """speaker_profiles/ 内の .pt ファイル名一覧を返す（起動時用: ファイルシステムを直接参照）。"""
    if not SPEAKER_PROFILES_DIR.exists():
        return []
    return sorted(p.name for p in SPEAKER_PROFILES_DIR.glob("*.pt"))


def _fetch_profile_names() -> list[str]:
    """API 経由でプロファイル名一覧を取得する（ランタイム更新用）。

    Returns:
        プロファイルファイル名のリスト。接続失敗時はファイルシステムにフォールバック。
    """
    try:
        resp = requests.get(f"{API_BASE}/tts/voice-clone/profiles", timeout=5)
        resp.raise_for_status()
        return [p["name"] for p in resp.json().get("profiles", [])]
    except Exception as exc:
        logger.warning("Failed to fetch profiles from API, falling back to filesystem: %s", exc)
        return _list_profiles()


def _fetch_profiles_with_meta() -> list[list[str]]:
    """API 経由でプロファイル一覧（名前 + 作成日時）を取得する。

    Returns:
        [[name, created_at], ...] の形式のリスト。
    """
    try:
        resp = requests.get(f"{API_BASE}/tts/voice-clone/profiles", timeout=5)
        resp.raise_for_status()
        return [[p["name"], p["created_at"]] for p in resp.json().get("profiles", [])]
    except Exception as exc:
        logger.warning("Failed to fetch profile metadata: %s", exc)
        return [[name, "—"] for name in _list_profiles()]


def _save_wav_response(resp: requests.Response) -> str:
    """WAV レスポンスを一時ファイルに保存してパスを返す。

    Args:
        resp: audio/wav を返す requests.Response。

    Returns:
        一時 WAV ファイルの絶対パス。
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(resp.content)
        return tmp.name


# ─── Tab 1: Voice Clone ───────────────────────────────────────────────────────


def voice_clone_generate(
    text: str,
    ref_audio_path: str | None,
    ref_text: str,
    profile_name: str,
    language: str,
    use_profile: bool,
) -> tuple[str | None, str]:
    """Voice Clone タブの生成ボタン処理。

    Args:
        text: 合成テキスト。
        ref_audio_path: 参照音声ファイルパス（use_profile=False 時に使用）。
        ref_text: 参照音声の書き起こし（空文字 = x-vector モード）。
        profile_name: 使用する話者プロファイル名（use_profile=True 時）。
        language: 言語指定。
        use_profile: True なら保存済みプロファイルを使用。

    Returns:
        (output_audio_path, status_message)
    """
    if not text.strip():
        return None, "エラー: テキストを入力してください。"

    try:
        if use_profile:
            if not profile_name:
                return None, "エラー: プロファイルを選択してください。"
            resp = requests.post(
                f"{API_BASE}/tts/voice-clone/profile",
                data={"text": text, "profile_name": profile_name, "language": language},
                timeout=120,
            )
        else:
            if not ref_audio_path:
                return None, "エラー: 参照音声をアップロードしてください。"
            # 推奨スペック（24kHz, mono, WAV）でない場合は API 送信前に変換する
            converted_path = ensure_wav_format(
                pathlib.Path(ref_audio_path),
                converted_dir=pathlib.Path(tempfile.gettempdir()) / "qwen_tts_converted",
            )
            with open(converted_path, "rb") as f:
                resp = requests.post(
                    f"{API_BASE}/tts/voice-clone",
                    data={
                        "text": text,
                        "ref_text": ref_text or None,
                        "language": language,
                    },
                    files={"ref_audio": ("reference.wav", f, "audio/wav")},
                    timeout=120,
                )

        if resp.status_code != 200:
            return None, f"エラー {resp.status_code}: {resp.text}"

        return _save_wav_response(resp), "生成完了"

    except requests.exceptions.ConnectionError:
        return None, f"API サーバーに接続できません ({API_BASE})。\ndocker compose up api を確認してください。"
    except Exception as exc:
        logger.exception("voice_clone_generate failed")
        return None, f"エラー: {exc}"


def _build_voice_clone_tab(languages: list[str], profiles: list[str]) -> gr.Dropdown:
    """Voice Clone タブを構築する。

    Args:
        languages: 言語選択肢。
        profiles: 起動時の話者プロファイル選択肢。

    Returns:
        プロファイル選択 Dropdown（プロファイル管理タブからの更新連携に使用）。
    """
    with gr.Tab("Voice Clone"):
        gr.Markdown(
            "### 組み合わせ #1 / #2: 参照音声 または 保存済みプロファイル\n"
            "自分の声（参照音声）から話者の声を複製して音声合成します。\n"
            "instruct は使用できません（Base モデルの制約）。"
        )
        with gr.Row():
            with gr.Column():
                vc_text = gr.Textbox(label="合成テキスト", lines=3, placeholder="ここにテキストを入力...")
                vc_language = gr.Dropdown(choices=languages, value="auto", label="言語")
                vc_use_profile = gr.Checkbox(label="保存済みプロファイルを使用", value=False)

                with gr.Group() as vc_audio_group:
                    vc_ref_audio = gr.Audio(label="参照音声 (WAV, 24kHz mono, 3〜10秒推奨)", type="filepath")
                    vc_ref_text = gr.Textbox(
                        label="参照音声の書き起こし（任意）",
                        placeholder="空欄 → x-vector モード / 入力あり → ICL モード（声質向上）",
                    )

                with gr.Group(visible=False) as vc_profile_group:
                    vc_profile = gr.Dropdown(
                        choices=profiles,
                        label="話者プロファイル",
                        info="speaker_profiles/ 内の .pt ファイル",
                    )

                vc_btn = gr.Button("生成", variant="primary")

            with gr.Column():
                vc_output = gr.Audio(label="生成音声", type="filepath")
                vc_status = gr.Textbox(label="ステータス", interactive=False)

        # プロファイル使用切り替えで表示を切り替える
        vc_use_profile.change(
            fn=lambda use: (gr.update(visible=not use), gr.update(visible=use)),
            inputs=vc_use_profile,
            outputs=[vc_audio_group, vc_profile_group],
        )

        vc_btn.click(
            fn=voice_clone_generate,
            inputs=[vc_text, vc_ref_audio, vc_ref_text, vc_profile, vc_language, vc_use_profile],
            outputs=[vc_output, vc_status],
        )

    return vc_profile


# ─── Tab 2: Custom Voice ──────────────────────────────────────────────────────


def custom_voice_generate(
    text: str,
    speaker: str,
    language: str,
    instruct: str,
) -> tuple[str | None, str]:
    """Custom Voice タブの生成ボタン処理。

    Args:
        text: 合成テキスト。
        speaker: 話者名。
        language: 言語指定。
        instruct: スタイル指示文（空文字 = なし）。

    Returns:
        (output_audio_path, status_message)
    """
    if not text.strip():
        return None, "エラー: テキストを入力してください。"
    if not speaker:
        return None, "エラー: 話者を選択してください。"

    try:
        resp = requests.post(
            f"{API_BASE}/tts/custom-voice",
            json={"text": text, "speaker": speaker, "language": language, "instruct": instruct},
            timeout=120,
        )
        if resp.status_code != 200:
            return None, f"エラー {resp.status_code}: {resp.text}"
        return _save_wav_response(resp), "生成完了"

    except requests.exceptions.ConnectionError:
        return None, f"API サーバーに接続できません ({API_BASE})。"
    except Exception as exc:
        logger.exception("custom_voice_generate failed")
        return None, f"エラー: {exc}"


def _build_custom_voice_tab(speakers: list[str], languages: list[str]) -> None:
    """Custom Voice タブを構築する。"""
    with gr.Tab("Custom Voice"):
        gr.Markdown(
            "### 組み合わせ #3 / #4: 組み込み話者（± instruct）\n"
            "事前定義された 9 種の話者から選んで音声合成します。\n"
            "`instruct` で話し方のスタイルを調整できます（空欄で省略可）。"
        )
        with gr.Row():
            with gr.Column():
                cv_text = gr.Textbox(label="合成テキスト", lines=3, placeholder="ここにテキストを入力...")
                cv_speaker = gr.Dropdown(choices=speakers, label="話者", value=speakers[0] if speakers else None)
                cv_language = gr.Dropdown(choices=languages, value="auto", label="言語")
                cv_instruct = gr.Textbox(
                    label="instruct（任意）",
                    placeholder="例: Speak in a calm, professional voice. / ゆっくり丁寧に話してください。",
                )
                cv_btn = gr.Button("生成", variant="primary")

            with gr.Column():
                cv_output = gr.Audio(label="生成音声", type="filepath")
                cv_status = gr.Textbox(label="ステータス", interactive=False)

        cv_btn.click(
            fn=custom_voice_generate,
            inputs=[cv_text, cv_speaker, cv_language, cv_instruct],
            outputs=[cv_output, cv_status],
        )


# ─── Tab 3: Voice Design ──────────────────────────────────────────────────────


def voice_design_generate(
    text: str,
    instruct: str,
    language: str,
) -> tuple[str | None, str]:
    """Voice Design タブの生成ボタン処理。

    Args:
        text: 合成テキスト。
        instruct: 声の設計指示文。
        language: 言語指定。

    Returns:
        (output_audio_path, status_message)
    """
    if not text.strip():
        return None, "エラー: テキストを入力してください。"
    if not instruct.strip():
        return None, "エラー: instruct を入力してください（Voice Design では必須です）。"

    try:
        resp = requests.post(
            f"{API_BASE}/tts/voice-design",
            json={"text": text, "instruct": instruct, "language": language},
            timeout=120,
        )
        if resp.status_code != 200:
            return None, f"エラー {resp.status_code}: {resp.text}"
        return _save_wav_response(resp), "生成完了"

    except requests.exceptions.ConnectionError:
        return None, f"API サーバーに接続できません ({API_BASE})。"
    except Exception as exc:
        logger.exception("voice_design_generate failed")
        return None, f"エラー: {exc}"


def _build_voice_design_tab(languages: list[str]) -> None:
    """Voice Design タブを構築する。"""
    with gr.Tab("Voice Design"):
        gr.Markdown(
            "### 組み合わせ #5: instruct で声を設計\n"
            "参照音声なし。自然言語で声のスタイルを記述して音声合成します。\n"
            "例: `Speak in an energetic, cheerful female voice.`"
        )
        with gr.Row():
            with gr.Column():
                vd_text = gr.Textbox(label="合成テキスト", lines=3, placeholder="ここにテキストを入力...")
                vd_instruct = gr.Textbox(
                    label="instruct（必須）",
                    lines=3,
                    placeholder=(
                        "例:\n"
                        "Speak in a calm, professional male voice with a neutral accent.\n"
                        "元気で明るい女性の声で話してください。"
                    ),
                )
                vd_language = gr.Dropdown(choices=languages, value="auto", label="言語")
                vd_btn = gr.Button("生成", variant="primary")

            with gr.Column():
                vd_output = gr.Audio(label="生成音声", type="filepath")
                vd_status = gr.Textbox(label="ステータス", interactive=False)

        vd_btn.click(
            fn=voice_design_generate,
            inputs=[vd_text, vd_instruct, vd_language],
            outputs=[vd_output, vd_status],
        )


# ─── Tab 4: プロファイル管理 ──────────────────────────────────────────────────


def profile_create(
    ref_audio_path: str | None,
    profile_name: str,
    ref_text: str,
) -> tuple[str, list[list[str]], gr.Dropdown, gr.Dropdown]:
    """プロファイル作成ボタン処理。

    Args:
        ref_audio_path: アップロードされた参照音声ファイルパス。
        profile_name: 保存するプロファイル名（拡張子なし）。
        ref_text: 参照音声の書き起こし（空文字 = x-vector モード）。

    Returns:
        (status, table_data, updated_vc_dropdown, updated_delete_dropdown)
    """
    if not ref_audio_path:
        return "エラー: 参照音声をアップロードしてください。", _fetch_profiles_with_meta(), gr.update(), gr.update()
    if not profile_name.strip():
        return "エラー: プロファイル名を入力してください。", _fetch_profiles_with_meta(), gr.update(), gr.update()

    try:
        with open(ref_audio_path, "rb") as f:
            resp = requests.post(
                f"{API_BASE}/tts/voice-clone/profiles",
                data={
                    "profile_name": profile_name.strip(),
                    "ref_text": ref_text.strip() or None,
                },
                files={"ref_audio": (pathlib.Path(ref_audio_path).name, f, "audio/wav")},
                timeout=180,
            )
        if resp.status_code not in (200, 201):
            return f"エラー {resp.status_code}: {resp.text}", _fetch_profiles_with_meta(), gr.update(), gr.update()

        created_name = resp.json().get("profile_name", "")
        new_names = _fetch_profile_names()
        table = _fetch_profiles_with_meta()
        return (
            f"作成完了: {created_name}",
            table,
            gr.update(choices=new_names),
            gr.update(choices=new_names),
        )

    except requests.exceptions.ConnectionError:
        return (
            f"API サーバーに接続できません ({API_BASE})。",
            _fetch_profiles_with_meta(),
            gr.update(),
            gr.update(),
        )
    except Exception as exc:
        logger.exception("profile_create failed")
        return f"エラー: {exc}", _fetch_profiles_with_meta(), gr.update(), gr.update()


def profile_delete(profile_name: str) -> tuple[str, list[list[str]], gr.Dropdown, gr.Dropdown]:
    """プロファイル削除ボタン処理。

    Args:
        profile_name: 削除するプロファイルファイル名（例: default.pt）。

    Returns:
        (status, table_data, updated_vc_dropdown, updated_delete_dropdown)
    """
    if not profile_name:
        return "エラー: 削除するプロファイルを選択してください。", _fetch_profiles_with_meta(), gr.update(), gr.update()

    try:
        resp = requests.delete(f"{API_BASE}/tts/voice-clone/profiles/{profile_name}", timeout=10)
        if resp.status_code == 404:
            return (
                f"プロファイルが見つかりません: {profile_name}",
                _fetch_profiles_with_meta(),
                gr.update(),
                gr.update(),
            )
        if resp.status_code != 200:
            return f"エラー {resp.status_code}: {resp.text}", _fetch_profiles_with_meta(), gr.update(), gr.update()

        new_names = _fetch_profile_names()
        table = _fetch_profiles_with_meta()
        return (
            f"削除完了: {profile_name}",
            table,
            gr.update(choices=new_names, value=None),
            gr.update(choices=new_names, value=None),
        )

    except requests.exceptions.ConnectionError:
        return (
            f"API サーバーに接続できません ({API_BASE})。",
            _fetch_profiles_with_meta(),
            gr.update(),
            gr.update(),
        )
    except Exception as exc:
        logger.exception("profile_delete failed")
        return f"エラー: {exc}", _fetch_profiles_with_meta(), gr.update(), gr.update()


def profile_refresh() -> tuple[list[list[str]], gr.Dropdown, gr.Dropdown]:
    """プロファイル一覧を再取得する。

    Returns:
        (table_data, updated_vc_dropdown, updated_delete_dropdown)
    """
    new_names = _fetch_profile_names()
    table = _fetch_profiles_with_meta()
    return table, gr.update(choices=new_names), gr.update(choices=new_names)


def _build_profile_management_tab(vc_profile_dropdown: gr.Dropdown) -> None:
    """プロファイル管理タブを構築する。

    Args:
        vc_profile_dropdown: Voice Clone タブのプロファイル選択 Dropdown（連携更新用）。
    """
    with gr.Tab("プロファイル管理"):
        gr.Markdown(
            "### 話者プロファイル作成・管理\n"
            "参照音声（WAV/MP3/M4A）からプロファイル (.pt) を生成・保存します。\n"
            "作成したプロファイルは **Voice Clone** タブで即座に使用できます。"
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### プロファイル作成")
                pm_ref_audio = gr.Audio(
                    label="参照音声（24kHz mono WAV 推奨、3〜10秒）",
                    type="filepath",
                )
                pm_ref_text = gr.Textbox(
                    label="参照音声の書き起こし（任意）",
                    placeholder="空欄 → x-vector モード / 入力あり → ICL モード（声質向上）",
                )
                pm_profile_name = gr.Textbox(
                    label="プロファイル名（英数字・_・- のみ）",
                    placeholder="例: my_voice",
                )
                pm_create_btn = gr.Button("作成する", variant="primary")
                pm_create_status = gr.Textbox(label="ステータス", interactive=False)

            with gr.Column():
                gr.Markdown("#### 保存済みプロファイル一覧")
                pm_table = gr.Dataframe(
                    headers=["プロファイル名", "作成日時 (UTC)"],
                    datatype=["str", "str"],
                    value=_fetch_profiles_with_meta(),
                    interactive=False,
                    label="保存済みプロファイル",
                )
                pm_delete_target = gr.Dropdown(
                    choices=_fetch_profile_names(),
                    label="削除するプロファイル",
                    info="一覧から選択してください",
                )
                with gr.Row():
                    pm_delete_btn = gr.Button("削除", variant="stop")
                    pm_refresh_btn = gr.Button("一覧を更新")
                pm_manage_status = gr.Textbox(label="ステータス", interactive=False)

        # 作成ボタン → テーブル・削除 Dropdown・Voice Clone Dropdown を更新する
        pm_create_btn.click(
            fn=profile_create,
            inputs=[pm_ref_audio, pm_profile_name, pm_ref_text],
            outputs=[pm_create_status, pm_table, vc_profile_dropdown, pm_delete_target],
        )

        # 削除ボタン → テーブル・削除 Dropdown・Voice Clone Dropdown を更新する
        pm_delete_btn.click(
            fn=profile_delete,
            inputs=[pm_delete_target],
            outputs=[pm_manage_status, pm_table, vc_profile_dropdown, pm_delete_target],
        )

        # 一覧更新ボタン
        pm_refresh_btn.click(
            fn=profile_refresh,
            inputs=[],
            outputs=[pm_table, vc_profile_dropdown, pm_delete_target],
        )


# ─── アプリ組み立て ────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    """Gradio アプリを組み立てて返す。

    Returns:
        gr.Blocks インスタンス。
    """
    speakers = _fetch_speakers()
    languages = _fetch_languages()
    profiles = _list_profiles()

    logger.info("Speakers: %s", speakers)
    logger.info("Languages: %s", languages)
    logger.info("Profiles: %s", profiles)

    with gr.Blocks(title="Qwen3-TTS WebUI") as demo:
        gr.Markdown("# Qwen3-TTS WebUI\nAPI: `" + API_BASE + "`")

        vc_profile_dropdown = _build_voice_clone_tab(languages, profiles)
        _build_custom_voice_tab(speakers, languages)
        _build_voice_design_tab(languages)
        _build_profile_management_tab(vc_profile_dropdown)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
