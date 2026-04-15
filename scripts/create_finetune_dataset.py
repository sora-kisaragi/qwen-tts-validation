"""
Fine-tuning 用データセット (raw JSONL) 作成ユーティリティ

- 目的: WAV ディレクトリ + 書き起こし CSV から、公式 prepare_data.py への
        入力 JSONL ファイルを生成する。
        推奨スペック（24kHz, mono, WAV）でない音声は ffmpeg で自動変換する。
- 対象: Issue #36 — 参照音声 + instruct 組み合わせのファインチューニング
- 関連: https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning
         scripts/audio_utils.py — 音声変換ユーティリティ
         docs/v2-design.md — 組み合わせマトリクス

作成者: 宗廣 颯真
作成日: 2026-04-15
最終更新者: 宗廣 颯真
最終更新日: 2026-04-15

Usage:
    # 1. 書き起こし CSV を用意（ヘッダーなし: filename,text）
    #    例: 001.wav,おはようございます。今日もよろしくお願いします。
    #        002.mp3,テストの音声ファイルです。  ← MP3 も自動変換される

    # 2. raw JSONL を生成（このスクリプト）
    python3 scripts/create_finetune_dataset.py \\
        --wav-dir finetune_data/wavs \\
        --transcript finetune_data/transcript.csv \\
        --ref-audio finetune_data/ref_speaker.wav \\
        --output finetune_data/raw_data.jsonl \\
        --language japanese

    # 3. 音声コードに変換（公式 prepare_data.py）
    python3 finetuning/prepare_data.py \\
        --input_jsonl finetune_data/raw_data.jsonl \\
        --output_jsonl finetune_data/prepared_data.jsonl

    # 4. ファインチューニング（公式 sft_12hz.py）
    python3 finetuning/sft_12hz.py \\
        --train_jsonl finetune_data/prepared_data.jsonl \\
        --output_model_path finetune_output/ \\
        --speaker_name my_voice \\
        --num_epochs 5
"""

import argparse
import csv
import json
import logging
import pathlib
import sys

import soundfile as sf
from audio_utils import ensure_wav_format

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 長さの許容範囲
_MIN_DURATION_SEC = 1.0
_MAX_DURATION_SEC = 30.0
_RECOMMENDED_MIN_DURATION_SEC = 3.0
_RECOMMENDED_MAX_DURATION_SEC = 15.0


def _validate_duration(path: pathlib.Path, label: str) -> float:
    """変換後の WAV ファイルの長さを検証して duration (秒) を返す。

    Args:
        path: 検証する WAV ファイルのパス（変換済みであること）。
        label: エラーメッセージに使うラベル。

    Returns:
        再生時間（秒）。

    Raises:
        SystemExit: 長さが許容範囲外の場合。
    """
    info = sf.info(str(path))
    duration = info.frames / info.samplerate

    if duration < _MIN_DURATION_SEC:
        logger.error("[%s] Duration %.2fs is too short (minimum: %.1fs).", label, duration, _MIN_DURATION_SEC)
        sys.exit(1)

    if duration > _MAX_DURATION_SEC:
        logger.error("[%s] Duration %.2fs is too long (maximum: %.1fs).", label, duration, _MAX_DURATION_SEC)
        sys.exit(1)

    if duration < _RECOMMENDED_MIN_DURATION_SEC:
        logger.warning(
            "[%s] Duration %.2fs is shorter than recommended (%.1fs).", label, duration, _RECOMMENDED_MIN_DURATION_SEC
        )

    if duration > _RECOMMENDED_MAX_DURATION_SEC:
        logger.warning(
            "[%s] Duration %.2fs is longer than recommended (%.1fs).", label, duration, _RECOMMENDED_MAX_DURATION_SEC
        )

    return duration


def _read_transcript(transcript_path: pathlib.Path) -> list[tuple[str, str]]:
    """書き起こし CSV を読み込んで (filename, text) のリストを返す。

    CSV フォーマット（ヘッダーなし）:
        filename,text
        001.wav,おはようございます。
        002.mp3,テストの音声ファイルです。

    Args:
        transcript_path: CSV ファイルのパス。

    Returns:
        (filename, text) のタプルリスト。

    Raises:
        SystemExit: ファイルが存在しない、または空の場合。
    """
    if not transcript_path.exists():
        logger.error("Transcript file not found: %s", transcript_path)
        sys.exit(1)

    entries: list[tuple[str, str]] = []
    with transcript_path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        for lineno, row in enumerate(reader, start=1):
            # ヘッダー行を自動スキップ（1行目が "filename" で始まる場合）
            if lineno == 1 and row and row[0].strip().lower() in ("filename", "file", "audio"):
                logger.info("Skipping header row: %s", row)
                continue

            if len(row) < 2:  # noqa: PLR2004
                logger.warning("Line %d: expected 2 columns, got %d — skipped.", lineno, len(row))
                continue

            filename = row[0].strip()
            text = row[1].strip()

            if not filename or not text:
                logger.warning("Line %d: empty filename or text — skipped.", lineno)
                continue

            entries.append((filename, text))

    if not entries:
        logger.error("No valid entries found in transcript: %s", transcript_path)
        sys.exit(1)

    return entries


def create_dataset(
    wav_dir: pathlib.Path,
    transcript_path: pathlib.Path,
    ref_audio_path: pathlib.Path,
    output_path: pathlib.Path,
    language: str,
) -> int:
    """raw JSONL データセットを生成する。

    推奨スペック（24kHz, mono, WAV）でない音声ファイルは ffmpeg で自動変換する。
    変換後ファイルは wav_dir/converted/ に保存される。

    Args:
        wav_dir: 音声ファイルが格納されているディレクトリ（WAV 以外も可）。
        transcript_path: 書き起こし CSV のパス。
        ref_audio_path: 話者代表音声のパス（全サンプルで共通）。
        output_path: 出力 JSONL ファイルのパス。
        language: 言語文字列（例: "japanese", "auto"）。

    Returns:
        書き込んだサンプル数。

    Raises:
        SystemExit: 入力ファイルに問題がある場合。
    """
    converted_dir = wav_dir / "converted"

    # 参照音声を変換・検証する
    logger.info("Processing ref_audio: %s", ref_audio_path)
    if not ref_audio_path.exists():
        logger.error("ref_audio not found: %s", ref_audio_path)
        sys.exit(1)
    ref_wav_path = ensure_wav_format(ref_audio_path, converted_dir=converted_dir)
    _validate_duration(ref_wav_path, "ref_audio")

    # 書き起こしを読み込む
    entries = _read_transcript(transcript_path)
    logger.info("Transcript: %d entries loaded.", len(entries))

    # 出力先のディレクトリを作成する
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for filename, text in entries:
            src_path = wav_dir / filename
            if not src_path.exists():
                logger.error("[%s] File not found: %s — skipped.", filename, src_path)
                continue

            # 推奨スペックでなければ自動変換する
            wav_path = ensure_wav_format(src_path, converted_dir=converted_dir)
            duration = _validate_duration(wav_path, filename)

            record = {
                "audio": str(wav_path.resolve()),
                "text": text,
                "ref_audio": str(ref_wav_path.resolve()),
                "language": language,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("  [%s] %.2fs — %s", wav_path.name, duration, text[:40])
            written += 1

    if written == 0:
        logger.error("No entries written. Check wav_dir and transcript.")
        sys.exit(1)

    logger.info("Done. %d entries written to %s.", written, output_path)
    return written


def main() -> None:
    """CLI エントリーポイント。"""
    parser = argparse.ArgumentParser(
        description=(
            "音声ディレクトリ + 書き起こし CSV から fine-tuning 用 raw JSONL を生成する。\n"
            "推奨スペック（24kHz, mono, WAV）でない音声は ffmpeg で自動変換する。\n"
            "生成後は公式 prepare_data.py で音声コードに変換すること。"
        )
    )
    parser.add_argument(
        "--wav-dir",
        required=True,
        type=pathlib.Path,
        help="音声ファイルが格納されているディレクトリ（WAV / MP3 / M4A 等可）。",
    )
    parser.add_argument(
        "--transcript",
        required=True,
        type=pathlib.Path,
        help="書き起こし CSV のパス（ヘッダーなし: filename,text）。",
    )
    parser.add_argument(
        "--ref-audio",
        required=True,
        type=pathlib.Path,
        help="話者代表音声のパス（全サンプルで共通の参照音声）。WAV 以外も自動変換する。",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="出力 JSONL ファイルのパス。",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="言語。例: auto, japanese, english（デフォルト: auto）。",
    )

    args = parser.parse_args()

    create_dataset(
        wav_dir=args.wav_dir,
        transcript_path=args.transcript,
        ref_audio_path=args.ref_audio,
        output_path=args.output,
        language=args.language,
    )


if __name__ == "__main__":
    main()
