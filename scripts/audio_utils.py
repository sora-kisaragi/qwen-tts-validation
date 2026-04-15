"""
音声ファイル変換ユーティリティ

- 目的: API / WebUI / CLI スクリプトで共通して使う音声フォーマット検証・変換処理。
        受け付けた音声が推奨スペック（24kHz, mono, WAV）でない場合に ffmpeg で自動変換する。
- 対象: api/routes/voice_clone.py, webui/app.py, scripts/create_finetune_dataset.py
- 関連: Issue #36 — ファインチューニングデータ準備

作成者: 宗廣 颯真
作成日: 2026-04-15
最終更新者: 宗廣 颯真
最終更新日: 2026-04-15
"""

import logging
import pathlib
import shutil
import subprocess

import soundfile as sf

logger = logging.getLogger(__name__)

# モデルが期待するスペック
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1


def _needs_conversion(src_path: pathlib.Path, target_sr: int, target_channels: int) -> tuple[bool, str]:
    """変換が必要かどうかを判定する。

    Args:
        src_path: 検査する音声ファイル。
        target_sr: 目標サンプルレート。
        target_channels: 目標チャンネル数。

    Returns:
        (変換要否, 理由メッセージ) のタプル。
        変換不要の場合は (False, "")。
    """
    # WAV 以外は常に変換が必要
    if src_path.suffix.lower() != ".wav":
        return True, f"format={src_path.suffix} (not WAV)"

    # WAV でも sr/channels が合わなければ変換が必要
    try:
        info = sf.info(str(src_path))
    except Exception as exc:
        # soundfile が読めない場合は ffmpeg に任せる
        return True, f"unreadable by soundfile ({exc})"

    reasons: list[str] = []
    if info.samplerate != target_sr:
        reasons.append(f"sr={info.samplerate} (expected {target_sr})")
    if info.channels != target_channels:
        reasons.append(f"channels={info.channels} (expected {target_channels})")

    if reasons:
        return True, ", ".join(reasons)
    return False, ""


def _run_ffmpeg(src_path: pathlib.Path, dst_path: pathlib.Path, target_sr: int, target_channels: int) -> None:
    """ffmpeg で音声を変換する。

    Args:
        src_path: 変換元ファイル。
        dst_path: 変換先ファイル（WAV）。
        target_sr: 目標サンプルレート。
        target_channels: 目標チャンネル数。

    Raises:
        RuntimeError: ffmpeg が見つからない、または変換に失敗した場合。
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg が見つかりません。コンテナに ffmpeg がインストールされているか確認してください。")

    cmd = [
        "ffmpeg",
        "-y",  # 上書き許可
        "-i",
        str(src_path),
        "-ar",
        str(target_sr),
        "-ac",
        str(target_channels),
        str(dst_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 変換に失敗しました: {result.stderr[-500:] if result.stderr else '(no stderr)'}")


def ensure_wav_format(
    src_path: pathlib.Path,
    target_sr: int = TARGET_SAMPLE_RATE,
    target_channels: int = TARGET_CHANNELS,
    converted_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """音声ファイルのフォーマットを確認し、必要なら ffmpeg で変換して返す。

    変換が不要な場合は src_path をそのまま返すため、呼び出し元で
    「元ファイルを削除すべきか」を判定するには戻り値 != src_path で確認できる。

    Args:
        src_path: 検査・変換対象の音声ファイル。WAV 以外（MP3, M4A 等）も受け付ける。
        target_sr: 目標サンプルレート（デフォルト: 24000）。
        target_channels: 目標チャンネル数（デフォルト: 1 = mono）。
        converted_dir: 変換後ファイルの保存先ディレクトリ。
                       None の場合は src_path.parent / "converted" を使用する。

    Returns:
        変換後のファイルパス。変換不要だった場合は src_path をそのまま返す。

    Raises:
        RuntimeError: ffmpeg が見つからない、または変換に失敗した場合。
    """
    needs, reason = _needs_conversion(src_path, target_sr, target_channels)

    if not needs:
        return src_path

    # 変換先パスを決定する
    out_dir = converted_dir if converted_dir is not None else (src_path.parent / "converted")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_path = out_dir / (src_path.stem + ".wav")

    logger.info(
        "Converting audio: %s → %s (reason: %s)",
        src_path.name,
        dst_path,
        reason,
    )
    _run_ffmpeg(src_path, dst_path, target_sr, target_channels)
    logger.info("Conversion done: %s", dst_path.name)

    return dst_path
