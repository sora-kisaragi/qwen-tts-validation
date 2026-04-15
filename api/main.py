"""
Qwen3-TTS FastAPI サーバー

- 目的: Base / VoiceDesign / CustomVoice の 3 モデルを起動時にロードし、
        全対応組み合わせを REST API として提供する。
- 対象: Issue #32 — FastAPI REST API サーバー実装
- 関連: docs/v2-design.md — エンドポイント設計

作成者: 宗廣 颯真
作成日: 2026-04-14
最終更新者: 宗廣 颯真
最終更新日: 2026-04-15

Usage (inside container):
    uvicorn api.main:app --host 0.0.0.0 --port 7865

    # または docker compose
    docker compose run --service-ports qwen-tts \
        uvicorn api.main:app --host 0.0.0.0 --port 7865
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.models import get_supported_languages, get_supported_speakers, load_all_models
from api.routes import custom_voice, voice_clone, voice_design

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動時に全モデルをロードし、終了時にリソースを解放する。"""
    logger.info("Starting up: loading all TTS models...")
    load_all_models()
    logger.info("All models loaded. API is ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Qwen3-TTS API",
    description=(
        "Qwen3-TTS の全モデル・全組み合わせを提供する REST API。\n\n"
        "| エンドポイント | 組み合わせ | モデル |\n"
        "|---|---|---|\n"
        "| POST /tts/voice-clone | 参照音声アップロード | Base |\n"
        "| POST /tts/voice-clone/profile | 保存済みプロファイル | Base |\n"
        "| POST /tts/custom-voice | 組み込み話者（± instruct） | CustomVoice |\n"
        "| POST /tts/voice-design | instruct で声設計 | VoiceDesign |\n"
    ),
    version="2.1.0",
    lifespan=lifespan,
)

# ─── ルート登録 ────────────────────────────────────────────────────────────────
app.include_router(voice_clone.router, prefix="/tts", tags=["Voice Clone"])
app.include_router(custom_voice.router, prefix="/tts", tags=["Custom Voice"])
app.include_router(voice_design.router, prefix="/tts", tags=["Voice Design"])


# ─── メタ情報エンドポイント ────────────────────────────────────────────────────
@app.get("/tts/speakers", tags=["Info"], summary="利用可能な話者一覧")
async def list_speakers() -> JSONResponse:
    """CustomVoice モデルで使用できる話者名の一覧を返す。"""
    return JSONResponse({"speakers": get_supported_speakers()})


@app.get("/tts/languages", tags=["Info"], summary="対応言語一覧")
async def list_languages() -> JSONResponse:
    """各モデルが対応する言語名の一覧を返す。"""
    return JSONResponse({"languages": get_supported_languages()})


@app.get("/health", tags=["Info"], summary="ヘルスチェック")
async def health() -> JSONResponse:
    """サーバーの稼働状態を返す。"""
    return JSONResponse({"status": "ok"})
