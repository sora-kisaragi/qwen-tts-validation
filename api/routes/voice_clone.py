"""
Voice Clone ルート — 組み合わせ #1, #2

- 目的: 参照音声ファイルまたは保存済みプロファイル (.pt) を使って
        Base モデルで音声合成する。
- 対象: POST /tts/voice-clone, POST /tts/voice-clone/profile
- 関連: Issue #32, docs/v2-design.md 組み合わせ #1/#2

作成者: 宗廣 颯真
作成日: 2026-04-14
最終更新者: 宗廣 颯真
最終更新日: 2026-04-14
"""

import io
import logging
import pathlib
import sys
import tempfile

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import Response

_SCRIPTS_DIR = pathlib.Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from audio_utils import ensure_wav_format  # noqa: E402
from model_utils import load_speaker_profile  # noqa: E402

from api.models import get_model  # noqa: E402

router = APIRouter()
logger = logging.getLogger(__name__)

_SPEAKER_PROFILES_DIR = pathlib.Path("/workspace/speaker_profiles")


def _wav_to_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    """numpy 波形を WAV バイト列に変換する。

    Args:
        wav: float32 の 1D 波形配列。
        sample_rate: サンプルレート (Hz)。

    Returns:
        WAV フォーマットのバイト列。
    """
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    return buf.getvalue()


@router.post(
    "/voice-clone",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}, "description": "生成した WAV 音声"}},
    summary="参照音声から音声合成（組み合わせ #1）",
    description=(
        "アップロードした参照音声（WAV）から話者の声を抽出し、指定テキストを合成する。\n\n"
        "- `ref_text` を省略すると x-vector モード（話者埋め込みのみ）で動作する。\n"
        "- `ref_text` を指定すると ICL モード（参照音声の音声コードも利用）で動作し、より高い声質再現度が得られる。"
    ),
)
async def voice_clone(
    text: str = Form(..., description="合成するテキスト。"),
    ref_audio: UploadFile = Form(..., description="参照音声 WAV ファイル（24kHz mono, 3〜10秒推奨）。"),
    ref_text: str | None = Form(
        default=None,
        description="参照音声の書き起こし（任意）。省略時は x-vector モード。",
    ),
    language: str = Form(default="auto", description="言語（例: auto, japanese, english）。"),
) -> Response:
    model = get_model("base")

    # 参照音声を一時ファイルに保存してからパスで渡す
    suffix = pathlib.Path(ref_audio.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await ref_audio.read())
        tmp_path = tmp.name

    # アップロードされた音声が推奨スペック（24kHz, mono, WAV）でない場合に変換する
    tmp_path_obj = pathlib.Path(tmp_path)
    converted_dir = tmp_path_obj.parent / "converted"
    wav_path = ensure_wav_format(tmp_path_obj, converted_dir=converted_dir)

    try:
        x_vector_only = ref_text is None
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(wav_path),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )
    except Exception as exc:
        logger.exception("voice_clone failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path_obj.unlink(missing_ok=True)
        # 変換が行われた場合は変換後ファイルも削除する
        if wav_path != tmp_path_obj:
            wav_path.unlink(missing_ok=True)

    return Response(
        content=_wav_to_bytes(wavs[0], sample_rate),
        media_type="audio/wav",
    )


@router.post(
    "/voice-clone/profile",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}, "description": "生成した WAV 音声"}},
    summary="保存済みプロファイルから音声合成（組み合わせ #2）",
    description=(
        "事前に `create_speaker_profile.py` で生成した話者プロファイル (.pt) を使って音声合成する。\n\n"
        "参照音声の再処理が不要なため、同一話者を繰り返し使う場合に高速。\n"
        "`profile_name` は `speaker_profiles/` ディレクトリ内のファイル名（例: `default.pt`）。"
    ),
)
async def voice_clone_profile(
    text: str = Form(..., description="合成するテキスト。"),
    profile_name: str = Form(..., description="speaker_profiles/ 以下のプロファイルファイル名。例: default.pt"),
    language: str = Form(default="auto", description="言語（例: auto, japanese, english）。"),
) -> Response:
    model = get_model("base")
    device = str(model.device)

    profile_path = _SPEAKER_PROFILES_DIR / profile_name
    if not profile_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Profile not found: {profile_name}. Check speaker_profiles/ directory.",
        )

    try:
        profile = load_speaker_profile(profile_path, device=device)
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=[profile],
        )
    except Exception as exc:
        logger.exception("voice_clone_profile failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=_wav_to_bytes(wavs[0], sample_rate),
        media_type="audio/wav",
    )
