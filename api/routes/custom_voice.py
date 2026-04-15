"""
Custom Voice ルート — 組み合わせ #3, #4

- 目的: 組み込み話者（ono_anna 等）を使って CustomVoice モデルで音声合成する。
        instruct は任意で、声のスタイル調整に使う。
- 対象: POST /tts/custom-voice
- 関連: Issue #32, docs/v2-design.md 組み合わせ #3/#4

作成者: 宗廣 颯真
作成日: 2026-04-14
最終更新者: 宗廣 颯真
最終更新日: 2026-04-14
"""

import io
import logging

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from api.models import get_model
from api.schemas import CustomVoiceRequest

router = APIRouter()
logger = logging.getLogger(__name__)


def _wav_to_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    return buf.getvalue()


@router.post(
    "/custom-voice",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}, "description": "生成した WAV 音声"}},
    summary="組み込み話者で音声合成（組み合わせ #3/#4）",
    description=(
        "事前定義された話者（aiden, ono_anna 等）を使って音声合成する。\n\n"
        "- `instruct` を省略すると話者デフォルトのスタイルで合成する（組み合わせ #3）。\n"
        "- `instruct` を指定すると声のスタイルを調整できる（組み合わせ #4）。\n"
        "利用可能な話者は `GET /tts/speakers`、言語は `GET /tts/languages` で取得できる。"
    ),
)
async def custom_voice(req: CustomVoiceRequest) -> Response:
    model = get_model("custom_voice")

    try:
        wavs, sample_rate = model.generate_custom_voice(
            text=req.text,
            speaker=req.speaker,
            language=req.language,
            instruct=req.instruct or None,
        )
    except ValueError as exc:
        # 不正な speaker / language はバリデーションエラーとして 422 を返す
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("custom_voice failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=_wav_to_bytes(wavs[0], sample_rate),
        media_type="audio/wav",
    )
