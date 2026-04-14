"""
API リクエスト/レスポンス スキーマ定義

- 目的: FastAPI エンドポイントの入力バリデーションに使う Pydantic モデルを定義する。
- 対象: api/routes/ 配下の各ルートモジュール
- 関連: Issue #32 — FastAPI REST API サーバー実装

作成者: 宗廣 颯真
作成日: 2026-04-14
最終更新者: 宗廣 颯真
最終更新日: 2026-04-14
"""

from pydantic import BaseModel, Field


class CustomVoiceRequest(BaseModel):
    """POST /tts/custom-voice のリクエストボディ。"""

    text: str = Field(..., description="合成するテキスト。", min_length=1)
    speaker: str = Field(
        ...,
        description="話者名。/tts/speakers で取得できる値を使用する。",
    )
    language: str = Field(
        default="auto",
        description="言語。/tts/languages で取得できる値を使用する。デフォルトは 'auto'。",
    )
    instruct: str = Field(
        default="",
        description="声のスタイルを自然言語で指定する（任意）。例: 'Speak in a calm, professional voice.'",
    )


class VoiceDesignRequest(BaseModel):
    """POST /tts/voice-design のリクエストボディ。"""

    text: str = Field(..., description="合成するテキスト。", min_length=1)
    instruct: str = Field(
        ...,
        description="声をテキストで設計する指示文。例: 'Speak in an energetic female voice.'",
        min_length=1,
    )
    language: str = Field(
        default="auto",
        description="言語。デフォルトは 'auto'。",
    )
