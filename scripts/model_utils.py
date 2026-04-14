"""
Qwen3-TTS モデルローダーユーティリティ

- 目的: snapshot_download の不完全キャッシュ問題を回避し、
         speech_tokenizer/ サブディレクトリを含む全ファイルを確実に取得する。
         また話者プロファイルの保存・ロード機能を提供する。
- 対象: test_basic_tts.py / test_japanese.py / test_voice_cloning.py /
         create_speaker_profile.py から利用
- 関連: Issue #1 — speech_tokenizer/model.safetensors が snapshot_download で
         取得されない問題 (AutoModel.from_pretrained はトップレベルファイルのみ取得)
         Issue #21 — 話者プロファイルのテンプレート化と再利用

作成者: 宗廣 颯真
作成日: 2026-04-14
最終更新者: 宗廣 颯真
最終更新日: 2026-04-14
"""

import pathlib

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from qwen_tts import VoiceClonePromptItem

# speech_tokenizer/ 以下の必須ファイル一覧。
# snapshot_download がスキップするケースがあるため明示的に取得する。
_SPEECH_TOKENIZER_FILES = [
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
]


def save_speaker_profile(prompt: VoiceClonePromptItem, path: pathlib.Path) -> None:
    """話者プロファイルを .pt ファイルに保存する。

    VoiceClonePromptItem を torch.save でシリアライズする。
    保存後は参照音声ファイルなしで同一話者の音声合成が可能になる。

    Args:
        prompt: create_voice_clone_prompt() が返す話者プロファイル。
        path: 保存先の .pt ファイルパス。親ディレクトリが存在しない場合は作成する。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt, path)
    print(f"Speaker profile saved: {path}")


def load_speaker_profile(path: pathlib.Path, device: str = "cpu") -> VoiceClonePromptItem:
    """保存済み話者プロファイルを .pt ファイルからロードする。

    Args:
        path: 読み込む .pt ファイルパス。
        device: テンソルをマップするデバイス (例: "cpu", "cuda:0")。
                TTS モデルと同じデバイスを指定すること。

    Returns:
        VoiceClonePromptItem。generate_voice_clone(voice_clone_prompt=...) に渡せる。

    Raises:
        FileNotFoundError: 指定したパスにファイルが存在しない場合。
    """
    if not path.exists():
        raise FileNotFoundError(f"Speaker profile not found: {path}")
    # weights_only=False: VoiceClonePromptItem は dataclass のため pickle が必要
    prompt: VoiceClonePromptItem = torch.load(path, map_location=device, weights_only=False)
    print(f"Speaker profile loaded: {path} (device={device})")
    return prompt


def ensure_model_downloaded(model_id: str) -> str:
    """モデルの全ファイルを HuggingFace キャッシュに確保し、ローカルパスを返す。

    snapshot_download は不完全なキャッシュをそのまま返すことがある。
    speech_tokenizer/ 以下のファイルを hf_hub_download で個別に取得してから
    snapshot_download でスナップショットパスを解決する。

    Args:
        model_id: HuggingFace モデル ID (例: "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    Returns:
        ローカルスナップショットの絶対パス。from_pretrained に直接渡せる。
    """
    print(f"Ensuring all model files are cached: {model_id}")

    # speech_tokenizer/ の各ファイルを個別にダウンロード（ブロブとシンボリックリンクを作成）
    for filename in _SPEECH_TOKENIZER_FILES:
        hf_hub_download(repo_id=model_id, filename=filename)

    # スナップショットパスを取得（残りのトップレベルファイルも補完）
    model_path = snapshot_download(model_id)
    print(f"Model path: {model_path}")
    return model_path
