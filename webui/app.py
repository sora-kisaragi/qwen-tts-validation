"""
Qwen3-TTS Gradio WebUI

- 目的: FastAPI サーバーを経由して全 TTS 組み合わせを操作できる WebUI を提供する。
        5 タブ構成: Voice Clone / Custom Voice / Voice Design / プロファイル管理 / データ収集
- 対象: Issue #33 — Gradio WebUI 実装, Issue #46 — ファインチューニングデータ収集,
         Issue #47 — WebUI 話者プロファイル管理
- 関連: docs/v2-design.md — WebUI タブ設計
         api/main.py — 呼び出す API サーバー

作成者: 宗廣 颯真
作成日: 2026-04-14
最終更新者: 宗廣 颯真
最終更新日: 2026-04-16

Usage:
    # docker compose で起動（API サーバーと同時に立ち上げる）
    docker compose up api webui

    # 単体で起動（API が別途起動済みの場合）
    TTS_API_URL=http://localhost:7865 python3 webui/app.py
"""

import csv
import logging
import os
import pathlib
import shutil
import sys
import tempfile

import gradio as gr
import requests

# scripts/ を sys.path に追加して audio_utils を利用する
_SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from audio_utils import ensure_wav_format  # noqa: E402
from create_finetune_dataset import create_dataset as _run_create_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# API サーバーの URL（環境変数で上書き可能）
API_BASE = os.environ.get("TTS_API_URL", "http://localhost:7865")

SPEAKER_PROFILES_DIR = pathlib.Path("/workspace/speaker_profiles")

_FINETUNE_DATA_DIR = pathlib.Path("/workspace/finetune_data")
_FINETUNE_WAVS_DIR = _FINETUNE_DATA_DIR / "wavs"
_FINETUNE_OUTPUT_JSONL = _FINETUNE_DATA_DIR / "raw_data.jsonl"
_FINETUNE_TRANSCRIPT_CSV = _FINETUNE_DATA_DIR / "transcript.csv"

# ファインチューニングデータ収集用のプリセットテキスト（50 文）
_PRESET_TEXTS: list[str] = [
    # 挨拶・日常会話
    "おはようございます。今日もよろしくお願いします。",
    "こんにちは。最近いかがお過ごしですか？",
    "こんばんは。今日は遅くなってしまいました。",
    "ありがとうございました。またよろしくお願いいたします。",
    "すみません、少しよろしいですか？",
    "はじめまして。どうぞよろしくお願いします。",
    "お疲れ様でした。ゆっくり休んでください。",
    "大丈夫ですよ。心配しないでください。",
    "少々お待ちください。すぐに参ります。",
    "失礼いたします。お邪魔してもよろしいでしょうか。",
    # 天気・自然
    "今日の天気はとても良いですね。",
    "明日は雨が降るかもしれません。",
    "春になると桜がとても綺麗に咲きますね。",
    "夏の夜空に花火が上がりました。",
    "秋の紅葉はとても美しいです。",
    # 買い物・食事
    "この商品はいくらですか？",
    "レシートをいただけますか？",
    "今日の夕食は何にしましょうか。",
    "このラーメンはとても美味しいです。",
    "コーヒーを一杯いただけますか？",
    # 仕事・学業
    "会議は三時から始まります。",
    "報告書を明日までに提出してください。",
    "新しいプロジェクトが来月から始まります。",
    "試験の結果はいかがでしたか？",
    "この資料を確認していただけますか？",
    # 交通・移動
    "次の電車は何時ですか？",
    "この道をまっすぐ行くと駅があります。",
    "タクシーを呼んでいただけますか？",
    "飛行機の時間に間に合いますか？",
    "この電車は東京駅に停まりますか？",
    # 感情・表現
    "それは本当に素晴らしいですね。",
    "少し考えさせてください。",
    "とても楽しい時間を過ごしました。",
    "残念ですが、今回は参加できません。",
    "心配しないでください、大丈夫ですよ。",
    # 説明・紹介
    "こちらが新しいモデルになります。",
    "この機能について説明させてください。",
    "まず最初に、基本的な操作方法をご紹介します。",
    "詳しくはマニュアルをご参照ください。",
    "ご不明な点があれば、いつでもご連絡ください。",
    # 長めの文
    "昨日の会議では、新しいサービスの開発方針について活発な議論が行われました。",
    "東京オリンピックは多くの人々に感動を与えた歴史的なイベントでした。",
    "人工知能技術の発展により、私たちの生活は大きく変わりつつあります。",
    "環境問題は現代社会が抱える最も重要な課題の一つです。",
    "健康的な生活を送るためには、適度な運動とバランスの良い食事が大切です。",
    # ナレーション風
    "本日はお集まりいただきありがとうございます。",
    "それでは、発表を始めさせていただきます。",
    "ご静聴ありがとうございました。",
    "以上で説明を終わります。",
    "ご質問はございますか？",
]


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


# ─── Tab 5: ファインチューニングデータ収集 ──────────────────────────────────────


def _make_finetune_table(texts: list[str], audio_map: dict) -> list[list]:
    """ファインチューニングデータ収集タブ用のテーブルデータを生成する。

    Args:
        texts: テキスト一覧。
        audio_map: {row_index: audio_path} の辞書。

    Returns:
        [[No., テキスト, 登録済み], ...] 形式のリスト。
    """
    return [[i + 1, text, "✅" if i in audio_map else "—"] for i, text in enumerate(texts)]


def load_preset_texts() -> tuple[list[list], list[str]]:
    """プリセットテキスト（50 文）をテーブルと State にロードする。

    Returns:
        (table_data, texts) のタプル。
    """
    return _make_finetune_table(_PRESET_TEXTS, {}), list(_PRESET_TEXTS)


def load_csv_texts(csv_path: str | None) -> tuple[str, list[list], list[str]]:
    """CSV ファイルからテキスト一覧を読み込む。

    対応フォーマット:
        - 1 列形式（ヘッダーなし）: テキストのみ
        - 2 列形式: filename,text（先頭列は無視して末尾列をテキストとして使用）

    Args:
        csv_path: アップロードされた CSV ファイルのパス。

    Returns:
        (status, table_data, texts) のタプル。
    """
    if not csv_path:
        return "エラー: CSV ファイルをアップロードしてください。", [], []

    texts: list[str] = []
    try:
        with open(str(csv_path), encoding="utf-8") as f:
            reader = csv.reader(f)
            for lineno, row in enumerate(reader, start=1):
                # ヘッダー行を自動スキップ（末尾列が "text" 等の場合）
                if lineno == 1 and row and row[-1].strip().lower() in ("text", "テキスト", "script"):
                    continue
                if not row:
                    continue
                # 複数列の場合は末尾列をテキストとして扱う（1 列でも機能する）
                text = row[-1].strip()
                if text:
                    texts.append(text)
    except Exception as exc:
        logger.exception("load_csv_texts failed")
        return f"エラー: {exc}", [], []

    if not texts:
        return "CSV に有効なテキストが見つかりませんでした。", [], []

    return f"読み込み完了: {len(texts)} 件", _make_finetune_table(texts, {}), texts


def register_audio_entry(
    row_num: int,
    audio_path: str | None,
    texts: list[str],
    audio_map: dict,
) -> tuple[str, list[list], dict, str | None]:
    """指定した行番号に音声ファイルを登録する。

    音声は推奨スペック（24kHz mono WAV）に変換してから保存する。
    保存先: _FINETUNE_WAVS_DIR / "{row_idx:03d}.wav"

    Args:
        row_num: 登録する行番号（1 始まり）。
        audio_path: アップロードされた音声ファイルのパス。
        texts: テキスト一覧（gr.State）。
        audio_map: {row_index: audio_path} の辞書（gr.State）。

    Returns:
        (status, table_data, updated_audio_map, preview_audio_path) のタプル。
    """
    if not texts:
        return "エラー: テキストを読み込んでください。", [], audio_map, None
    if not audio_path:
        return (
            "エラー: 音声ファイルをアップロードしてください。",
            _make_finetune_table(texts, audio_map),
            audio_map,
            None,
        )

    row_idx = int(row_num) - 1  # 1 始まり → 0 始まりに変換
    if row_idx < 0 or row_idx >= len(texts):
        return (
            f"エラー: 行番号は 1〜{len(texts)} の範囲で入力してください。",
            _make_finetune_table(texts, audio_map),
            audio_map,
            None,
        )

    try:
        _FINETUNE_WAVS_DIR.mkdir(parents=True, exist_ok=True)

        # 推奨スペック（24kHz mono WAV）に変換してから保存する
        src_path = pathlib.Path(audio_path)
        converted_dir = pathlib.Path(tempfile.gettempdir()) / "qwen_tts_finetune_converted"
        wav_path = ensure_wav_format(src_path, converted_dir=converted_dir)

        dest_path = _FINETUNE_WAVS_DIR / f"{row_idx:03d}.wav"
        shutil.copy2(str(wav_path), str(dest_path))

        # 変換で中間ファイルが生成された場合は削除する
        if wav_path != src_path:
            wav_path.unlink(missing_ok=True)

        new_audio_map = dict(audio_map)
        new_audio_map[row_idx] = str(dest_path)
        table = _make_finetune_table(texts, new_audio_map)
        return f"登録完了: 行 {int(row_num)} — {texts[row_idx][:30]}…", table, new_audio_map, str(dest_path)

    except Exception as exc:
        logger.exception("register_audio_entry failed")
        return f"エラー: {exc}", _make_finetune_table(texts, audio_map), audio_map, None


def generate_finetune_dataset(
    texts: list[str],
    audio_map: dict,
    ref_audio_path: str | None,
    language: str,
) -> tuple[str, str | None]:
    """ファインチューニング用 raw JSONL データセットを生成する。

    登録済み音声とテキストから transcript.csv を作成し、
    create_finetune_dataset.create_dataset() を呼び出して JSONL を生成する。

    Args:
        texts: テキスト一覧（gr.State）。
        audio_map: {row_index: audio_path} の辞書（gr.State）。
        ref_audio_path: 話者代表音声のパス（全サンプルで共通）。
        language: 言語指定（例: japanese, auto）。

    Returns:
        (status, jsonl_file_path) のタプル。jsonl_file_path は成功時のみ非 None。
    """
    if not texts:
        return "エラー: テキストを読み込んでください。", None
    if not audio_map:
        return "エラー: 少なくとも 1 件以上の音声を登録してください。", None
    if not ref_audio_path:
        return "エラー: 話者代表音声をアップロードしてください。", None

    _FINETUNE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _FINETUNE_WAVS_DIR.mkdir(parents=True, exist_ok=True)

    # audio_map のキーは gr.State 経由で int または str になり得るため int に統一する
    registered = sorted((int(k), v) for k, v in audio_map.items())

    try:
        # 登録済み行のみ transcript.csv に書き出す
        with _FINETUNE_TRANSCRIPT_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for row_idx, audio_path in registered:
                filename = pathlib.Path(audio_path).name
                writer.writerow([filename, texts[row_idx]])

        n_written = _run_create_dataset(
            wav_dir=_FINETUNE_WAVS_DIR,
            transcript_path=_FINETUNE_TRANSCRIPT_CSV,
            ref_audio_path=pathlib.Path(ref_audio_path),
            output_path=_FINETUNE_OUTPUT_JSONL,
            language=language,
        )
        return f"生成完了: {n_written} 件 → {_FINETUNE_OUTPUT_JSONL}", str(_FINETUNE_OUTPUT_JSONL)

    except SystemExit as exc:
        # create_dataset() 内の sys.exit() を捕捉してエラーメッセージとして返す
        return f"エラー: データセット生成に失敗しました。ログを確認してください（exit code: {exc.code}）。", None
    except Exception as exc:
        logger.exception("generate_finetune_dataset failed")
        return f"エラー: {exc}", None


def _build_finetune_tab(languages: list[str]) -> None:
    """ファインチューニングデータ収集タブを構築する。

    Args:
        languages: 言語選択肢。
    """
    with gr.Tab("データ収集"):
        gr.Markdown(
            "### ファインチューニング用データ収集\n"
            "テキストと音声を対応付けて fine-tuning 用データセット（raw JSONL）を作成します。\n"
            "生成した JSONL は `finetuning/prepare_data.py` に渡して学習データに変換します。"
        )

        # テキスト一覧と登録済み音声マップを State として保持する
        ft_texts = gr.State([])
        ft_audio_map = gr.State({})

        # ── テキスト管理 ──────────────────────────────────────────────────────
        with gr.Accordion("テキスト管理", open=True):
            with gr.Row():
                ft_preset_btn = gr.Button("プリセットを読み込む（50 文）", variant="secondary")
            with gr.Row():
                ft_csv_file = gr.File(
                    label="CSV ファイル（1 列: テキスト）",
                    file_types=[".csv", ".txt"],
                )
                ft_csv_load_btn = gr.Button("CSV を読み込む", variant="secondary")
            ft_csv_status = gr.Textbox(label="読み込みステータス", interactive=False, show_label=False)
            with gr.Accordion("CSV フォーマット", open=False):
                gr.Markdown(
                    "**1 列形式**（テキストのみ）:\n"
                    "```\nおはようございます。\nこんにちは。\n```\n\n"
                    "**2 列形式**（1 列目は無視、2 列目をテキストとして使用）:\n"
                    "```\n001.wav,おはようございます。\n002.wav,こんにちは。\n```\n\n"
                    "**推奨録音スペック**: 24kHz mono WAV、1〜30 秒（推奨 3〜15 秒）"
                )

        # ── セリフ一覧 + 音声登録 ────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                ft_table = gr.Dataframe(
                    headers=["No.", "テキスト", "登録済み"],
                    datatype=["number", "str", "str"],
                    value=[],
                    interactive=False,
                    label="セリフ一覧",
                    wrap=True,
                )

            with gr.Column(scale=2):
                gr.Markdown("#### 音声を登録する")
                ft_row_num = gr.Number(
                    label="行番号 (No.)",
                    value=1,
                    precision=0,
                    minimum=1,
                    step=1,
                )
                ft_reg_audio = gr.Audio(
                    label="音声ファイル（WAV / MP3 / M4A）",
                    type="filepath",
                )
                ft_reg_btn = gr.Button("登録する", variant="primary")
                ft_reg_status = gr.Textbox(label="登録ステータス", interactive=False, show_label=False)
                ft_preview = gr.Audio(label="登録音声プレビュー", type="filepath", interactive=False)

        # ── データセット生成 ──────────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("#### データセット生成")
        with gr.Row():
            with gr.Column():
                ft_ref_audio = gr.Audio(
                    label="話者代表音声（ref_audio）",
                    info="全サンプル共通の参照音声として使用する代表音声（24kHz mono WAV 推奨、3〜10 秒）",
                    type="filepath",
                )
                ft_language = gr.Dropdown(
                    choices=languages,
                    value="japanese",
                    label="言語",
                )
                ft_generate_btn = gr.Button("JSONL を生成する", variant="primary")

            with gr.Column():
                ft_gen_status = gr.Textbox(label="生成ステータス", interactive=False)
                ft_output_file = gr.File(label="生成された JSONL ファイル（ダウンロード）", interactive=False)

        with gr.Accordion("次のステップ", open=False):
            gr.Markdown(
                "生成した `raw_data.jsonl` を公式スクリプトで音声コードに変換してからファインチューニングします。\n\n"
                "**STEP 2: 音声コードへの変換**\n"
                "```bash\n"
                "docker compose run finetune python3 finetuning/prepare_data.py \\\n"
                "    --input_jsonl finetune_data/raw_data.jsonl \\\n"
                "    --output_jsonl finetune_data/prepared_data.jsonl\n"
                "```\n\n"
                "**STEP 3: ファインチューニング**\n"
                "```bash\n"
                "docker compose run finetune python3 finetuning/sft_12hz.py \\\n"
                "    --train_jsonl finetune_data/prepared_data.jsonl \\\n"
                "    --output_model_path finetune_output/ \\\n"
                "    --speaker_name my_voice \\\n"
                "    --num_epochs 5\n"
                "```"
            )

        # ── イベントハンドラー ────────────────────────────────────────────────
        ft_preset_btn.click(
            fn=load_preset_texts,
            inputs=[],
            outputs=[ft_table, ft_texts],
        )

        ft_csv_load_btn.click(
            fn=load_csv_texts,
            inputs=[ft_csv_file],
            outputs=[ft_csv_status, ft_table, ft_texts],
        )

        ft_reg_btn.click(
            fn=register_audio_entry,
            inputs=[ft_row_num, ft_reg_audio, ft_texts, ft_audio_map],
            outputs=[ft_reg_status, ft_table, ft_audio_map, ft_preview],
        )

        ft_generate_btn.click(
            fn=generate_finetune_dataset,
            inputs=[ft_texts, ft_audio_map, ft_ref_audio, ft_language],
            outputs=[ft_gen_status, ft_output_file],
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
        _build_finetune_tab(languages)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
