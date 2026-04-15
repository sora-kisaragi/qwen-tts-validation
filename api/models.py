"""
TTS モデル管理モジュール

- 目的: Base / VoiceDesign / CustomVoice の 3 モデルを起動時に一括ロードし、
        シングルトンとして保持する。API ルートから参照して使い回す。
- 対象: api/ 配下のルートモジュール
- 関連: Issue #32 — FastAPI REST API サーバー実装
         Issue #39 — 推論エンジン最適化（最終目標: TensorRT / vLLM-Omni）
         docs/v2-design.md — 対応組み合わせマトリクス

作成者: 宗廣 颯真
作成日: 2026-04-14
最終更新者: 宗廣 颯真
最終更新日: 2026-04-14
"""

import logging
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


def load_all_models() -> None:
    """3 種類の TTS モデルをすべてロードして _models に格納する。

    FastAPI の lifespan イベントから呼び出すこと。
    GB10 の 121 GB 統合メモリであれば 3 モデル同時保持は問題ない。
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Loading all TTS models on device=%s", device)

    for model_type, model_id in MODEL_IDS.items():
        logger.info("Loading %s (%s)...", model_type, model_id)
        model_path = ensure_model_downloaded(model_id)
        wrapper = Qwen3TTSModel.from_pretrained(
            model_path,
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
            logger.info("  %s: torch.compile applied.", model_type)

        _models[model_type] = wrapper
        logger.info("  %s loaded.", model_type)

    logger.info("All models ready.")


def get_model(model_type: str) -> Qwen3TTSModel:
    """指定した model_type のロード済みモデルを返す。

    Args:
        model_type: "base" / "voice_design" / "custom_voice"

    Returns:
        ロード済み Qwen3TTSModel。

    Raises:
        RuntimeError: モデルが未初期化の場合。
        KeyError: 未知の model_type の場合。
    """
    if model_type not in _models:
        raise RuntimeError(f"Model '{model_type}' is not loaded. Call load_all_models() first.")
    return _models[model_type]


def get_supported_speakers() -> list[str]:
    """CustomVoice モデルが対応する話者名のリストを返す。

    Returns:
        話者名のリスト（例: ["aiden", "ono_anna", ...]）。
    """
    return get_model("custom_voice").get_supported_speakers() or []


def get_supported_languages() -> list[str]:
    """CustomVoice モデルが対応する言語名のリストを返す。

    Returns:
        言語名のリスト（例: ["auto", "japanese", "english", ...]）。
    """
    return get_model("custom_voice").get_supported_languages() or []
