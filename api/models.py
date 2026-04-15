"""
TTS モデル管理モジュール

- 目的: Base / VoiceDesign / CustomVoice の 3 モデルを起動時に一括ロードし、
        シングルトンとして保持する。API ルートから参照して使い回す。
        ファインチューニング済みモデルは環境変数で起動時にオプションロードする。
- 対象: api/ 配下のルートモジュール
- 関連: Issue #32 — FastAPI REST API サーバー実装
         Issue #36 — ファインチューニング（参照音声 + instruct）
         Issue #39 — 推論エンジン最適化（最終目標: TensorRT / vLLM-Omni）
         docs/v2-design.md — 対応組み合わせマトリクス

作成者: 宗廣 颯真
作成日: 2026-04-14
最終更新者: 宗廣 颯真
最終更新日: 2026-04-15
"""

import logging
import os
import pathlib
import sys

import torch
from qwen_tts import Qwen3TTSModel

# scripts/ を sys.path に追加してモデルユーティリティを再利用する
_SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from model_utils import ensure_model_downloaded  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_IDS = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}

# モジュールレベルのシングルトン — 起動時に load_all_models() で初期化する
_models: dict[str, Qwen3TTSModel] = {}

# ファインチューニング済み話者名 → モデルキー のマッピング
# 例: {"my_voice": "finetuned_my_voice"}
_finetuned_speaker_map: dict[str, str] = {}


def _load_single_model(model_path: str | pathlib.Path, model_key: str, device: str) -> None:
    """1つのモデルをロードして _models に格納する共通処理。

    Args:
        model_path: ロードするモデルのディレクトリパスまたは HF モデル ID。
        model_key: _models に格納するキー名。
        device: ロード先デバイス（例: "cuda:0", "cpu"）。
    """
    wrapper = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map=device,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        max_memory={0: "60GiB"},
    )

    # torch.compile で推論グラフを最適化する（Issue #39 Step 1）。
    # 最終目標は TensorRT または vLLM-Omni への移行だが、
    # 現段階では compile が最小コストで最大の効果を得られる。
    # 初回推論時にコンパイルが走るため最初の1リクエストのみ遅延が発生する。
    if device != "cpu":
        wrapper.model = torch.compile(wrapper.model)
        logger.info("  %s: torch.compile applied.", model_key)

    _models[model_key] = wrapper
    logger.info("  %s loaded.", model_key)


def load_all_models() -> None:
    """3 種類の TTS モデルをすべてロードして _models に格納する。

    FastAPI の lifespan イベントから呼び出すこと。
    GB10 の 121 GB 統合メモリであれば 3 モデル同時保持は問題ない。

    環境変数 FINETUNE_MODEL_PATH が設定されている場合、ファインチューニング済みモデルも
    追加でロードする。FINETUNE_SPEAKER_NAME でカスタム話者名を指定すること。
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Loading all TTS models on device=%s", device)

    for model_type, model_id in MODEL_IDS.items():
        logger.info("Loading %s (%s)...", model_type, model_id)
        model_path = ensure_model_downloaded(model_id)
        _load_single_model(model_path, model_type, device)

    logger.info("All models ready.")

    # ファインチューニング済みモデルのオプションロード（Issue #36）
    _load_finetuned_model_if_configured(device)


def _load_finetuned_model_if_configured(device: str) -> None:
    """環境変数が設定されていればファインチューニング済みモデルをロードする。

    環境変数:
        FINETUNE_MODEL_PATH: チェックポイントディレクトリのパス（必須）。
        FINETUNE_SPEAKER_NAME: カスタム話者名（デフォルト: "my_voice"）。

    Args:
        device: ロード先デバイス。
    """
    ft_path = os.environ.get("FINETUNE_MODEL_PATH", "").strip()
    if not ft_path:
        return

    ft_speaker = os.environ.get("FINETUNE_SPEAKER_NAME", "my_voice").strip()
    model_key = f"finetuned_{ft_speaker}"

    checkpoint = pathlib.Path(ft_path)
    if not checkpoint.exists():
        logger.warning(
            "FINETUNE_MODEL_PATH=%s does not exist. Skipping fine-tuned model load.",
            ft_path,
        )
        return

    logger.info("Loading fine-tuned model from %s (speaker=%s)...", checkpoint, ft_speaker)
    _load_single_model(checkpoint, model_key, device)

    _finetuned_speaker_map[ft_speaker] = model_key
    logger.info("Fine-tuned speaker '%s' is now available.", ft_speaker)


def get_model(model_type: str) -> Qwen3TTSModel:
    """指定した model_type のロード済みモデルを返す。

    Args:
        model_type: "base" / "voice_design" / "custom_voice" または fine-tuned キー。

    Returns:
        ロード済み Qwen3TTSModel。

    Raises:
        RuntimeError: モデルが未初期化の場合。
    """
    if model_type not in _models:
        raise RuntimeError(f"Model '{model_type}' is not loaded. Call load_all_models() first.")
    return _models[model_type]


def get_model_for_speaker(speaker: str) -> tuple[Qwen3TTSModel, str]:
    """話者名に対応するモデルと解決済み話者名を返す。

    ファインチューニング済み話者が指定された場合は該当 fine-tuned モデルを、
    それ以外は組み込み CustomVoice モデルを返す。

    Args:
        speaker: 話者名。

    Returns:
        (Qwen3TTSModel, 解決済み話者名) のタプル。
    """
    if speaker in _finetuned_speaker_map:
        model_key = _finetuned_speaker_map[speaker]
        return get_model(model_key), speaker
    return get_model("custom_voice"), speaker


def get_supported_speakers() -> list[str]:
    """利用可能な全話者名のリストを返す（組み込み + ファインチューニング済み）。

    Returns:
        話者名のリスト（例: ["aiden", "ono_anna", ..., "my_voice"]）。
    """
    builtin = get_model("custom_voice").get_supported_speakers() or []
    finetuned = list(_finetuned_speaker_map.keys())
    return builtin + finetuned


def get_supported_languages() -> list[str]:
    """CustomVoice モデルが対応する言語名のリストを返す。

    Returns:
        言語名のリスト（例: ["auto", "japanese", "english", ...]）。
    """
    return get_model("custom_voice").get_supported_languages() or []
