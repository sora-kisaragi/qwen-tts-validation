"""
Voice Design ルート — 組み合わせ #5

- 目的: instruct テキストで声のスタイルを設計して音声合成する。
        参照音声は不要。VoiceDesign モデルを使用。
- 対象: POST /tts/voice-design
- 関連: Issue #32, docs/v2-design.md 組み合わせ #5

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
from api.schemas import VoiceDesignRequest

router = APIRouter()
logger = logging.getLogger(__name__)


def _wav_to_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    return buf.getvalue()


@router.post(
    "/voice-design",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}, "description": "生成した WAV 音声"}},
    summary="instruct で声設計して音声合成（組み合わせ #5）",
    description=(
        "自然言語の `instruct` で声のスタイルを設計して音声合成する。参照音声は不要。\n\n"
        "例: `instruct = 'Speak in a calm, professional male voice with a neutral accent.'`\n\n"
        "利用可能な言語は `GET /tts/languages` で取得できる。"
    ),
)
async def voice_design(req: VoiceDesignRequest) -> Response:
    model = get_model("voice_design")

    try:
        wavs, sample_rate = model.generate_voice_design(
            text=req.text,
            instruct=req.instruct,
            language=req.language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("voice_design failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=_wav_to_bytes(wavs[0], sample_rate),
        media_type="audio/wav",
    )
