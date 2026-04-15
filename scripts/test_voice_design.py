"""
VoiceDesign モデル調査スクリプト

- 目的: Qwen3-TTS-12Hz-1.7B-VoiceDesign の generate_voice_design() 機能を検証する。
        instruct パラメータによる声質・スタイル制御の効果を確認し、
        language パラメータによる日本語発音改善の効果も合わせて検証する。
- 対象: Issue #30, #31 の調査担当者
- 関連: Issue #30 — voice_design モデル調査・instruct 機能検証
         Issue #31 — language パラメータによる日本語発音改善の検証

作成者: 宗廣 颯真
作成日: 2026-04-14
最終更新者: 宗廣 颯真
最終更新日: 2026-04-14

Usage:
    # VoiceDesign 全ケース実行
    docker compose run qwen-tts python3 scripts/test_voice_design.py

    # language テストのみスキップ（instruct テストのみ）
    docker compose run qwen-tts python3 scripts/test_voice_design.py --skip-language

    # instruct テストのみスキップ（language テストのみ）
    docker compose run qwen-tts python3 scripts/test_voice_design.py --skip-instruct
"""

import argparse
import datetime
import logging
import pathlib
import sys
import time

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    logger.error("torch not found. Are you running inside the Docker container?")
    sys.exit(1)

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    logger.error("qwen_tts not found. Run: pip install qwen-tts")
    sys.exit(1)

from model_utils import ensure_model_downloaded  # noqa: E402

VOICE_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# ─── テストケース定義 ─────────────────────────────────────────────────────────

# TC-VD-01〜04: instruct による声質制御（英語テキスト）
INSTRUCT_TEST_CASES = [
    {
        "id": "TC-VD-01",
        "text": "Welcome. This is a test of voice design using natural language instructions.",
        "instruct": "",
        "language": None,
        "description": "instruct なし（デフォルト）",
    },
    {
        "id": "TC-VD-02",
        "text": "Welcome. This is a test of voice design using natural language instructions.",
        "instruct": "Speak in a calm, professional male voice with a neutral accent.",
        "language": None,
        "description": "落ち着いた男性プロフェッショナル",
    },
    {
        "id": "TC-VD-03",
        "text": "Welcome. This is a test of voice design using natural language instructions.",
        "instruct": "Speak in an energetic, cheerful female voice, as if presenting an exciting product.",
        "language": None,
        "description": "元気な女性プレゼンター",
    },
    {
        "id": "TC-VD-04",
        "text": "Welcome. This is a test of voice design using natural language instructions.",
        "instruct": "Speak slowly and clearly, like a language learning teacher reading an example sentence.",
        "language": None,
        "description": "ゆっくり丁寧な教師",
    },
]

# TC-VD-05〜07: language パラメータ検証（日本語テキスト）
# 漢字読みの中国語混入が language="ja" で改善するかを確認する (Issue #31)
LANGUAGE_TEST_CASES = [
    {
        "id": "TC-VD-05",
        "text": "音声合成の技術は急速に発展しています。自然な日本語の発音と抑揚が再現できるか確認します。",
        "instruct": "Speak in a natural Japanese voice.",
        "language": None,  # Auto
        "description": "日本語漢字 language=Auto（デフォルト）",
    },
    {
        "id": "TC-VD-06",
        "text": "音声合成の技術は急速に発展しています。自然な日本語の発音と抑揚が再現できるか確認します。",
        "instruct": "Speak in a natural Japanese voice.",
        "language": "japanese",
        "description": "日本語漢字 language=japanese",
    },
    {
        "id": "TC-VD-07",
        "text": "今日はQwen TTSのテストをしています。This is a mixed language test combining Japanese and English.",
        "instruct": "Speak naturally, switching between Japanese and English as appropriate.",
        "language": "japanese",
        "description": "日英混在 language=japanese",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Qwen3-TTS VoiceDesign model with various instruct and language settings.",
    )
    parser.add_argument(
        "--skip-instruct",
        action="store_true",
        help="Skip instruct test cases (TC-VD-01 to TC-VD-04).",
    )
    parser.add_argument(
        "--skip-language",
        action="store_true",
        help="Skip language parameter test cases (TC-VD-05 to TC-VD-07).",
    )
    return parser.parse_args()


def run_test_case(
    model: Qwen3TTSModel,
    tc: dict,
    output_dir: pathlib.Path,
) -> dict:
    """1 テストケースを実行し結果を返す。

    Args:
        model: ロード済み Qwen3TTSModel インスタンス。
        tc: テストケース定義辞書。
        output_dir: WAV 出力先ディレクトリ。

    Returns:
        テスト結果の辞書 (id, description, passed, duration_sec, latency_sec, max_amplitude, error)。
    """
    tc_id = tc["id"]
    logger.info("Running %s: %s", tc_id, tc["description"])
    logger.info("  text    : %s", tc["text"][:80])
    logger.info("  instruct: %s", tc["instruct"][:80] if tc["instruct"] else "(none)")
    logger.info("  language: %s", tc["language"] or "Auto")

    result: dict = {
        "id": tc_id,
        "description": tc["description"],
        "passed": False,
        "duration_sec": None,
        "latency_sec": None,
        "max_amplitude": None,
        "error": None,
    }

    try:
        t0 = time.perf_counter()
        wavs, sample_rate = model.generate_voice_design(
            text=tc["text"],
            instruct=tc["instruct"],
            language=tc["language"],
        )
        latency = time.perf_counter() - t0

        wav = wavs[0]
        duration = len(wav) / sample_rate
        max_amp = float(np.max(np.abs(wav)))

        output_path = output_dir / f"{tc_id.lower()}.wav"
        sf.write(str(output_path), wav, sample_rate)

        result["latency_sec"] = round(latency, 2)
        result["duration_sec"] = round(duration, 2)
        result["max_amplitude"] = round(max_amp, 4)
        result["passed"] = max_amp > 0.01 and duration >= 0.5

        status = "PASS" if result["passed"] else "FAIL"
        logger.info(
            "  %s — duration=%.2fs, latency=%.2fs, max_amp=%.4f → %s",
            tc_id,
            duration,
            latency,
            max_amp,
            status,
        )

    except Exception as exc:
        result["error"] = str(exc)
        logger.error("  %s FAIL — %s", tc_id, exc)

    return result


def print_summary(results: list[dict]) -> None:
    """テスト結果サマリーを標準出力に表示する。

    Args:
        results: run_test_case が返す辞書のリスト。
    """
    print("\n" + "=" * 70)
    print("VoiceDesign テスト結果サマリー")
    print("=" * 70)
    header = f"{'TC':<12} {'説明':<30} {'結果':<6} {'秒数':>7} {'遅延':>7} {'最大振幅':>10}"
    print(header)
    print("-" * 70)

    passed_count = 0
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            passed_count += 1
        dur = f"{r['duration_sec']:.2f}s" if r["duration_sec"] is not None else "—"
        lat = f"{r['latency_sec']:.2f}s" if r["latency_sec"] is not None else "—"
        amp = f"{r['max_amplitude']:.4f}" if r["max_amplitude"] is not None else "—"
        desc = r["description"][:28]
        print(f"{r['id']:<12} {desc:<30} {status:<6} {dur:>7} {lat:>7} {amp:>10}")
        if r["error"]:
            print(f"  └ ERROR: {r['error']}")

    print("-" * 70)
    total = len(results)
    print(f"総合: {passed_count}/{total} PASS")
    print("=" * 70)


def main() -> None:
    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = pathlib.Path("/workspace/output") / f"voice_design_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    logger.info("Loading model: %s", VOICE_DESIGN_MODEL_ID)
    model_path = ensure_model_downloaded(VOICE_DESIGN_MODEL_ID)
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        max_memory={0: "60GiB"},
    )
    logger.info("Model loaded (tts_model_type=%s).", model.model.tts_model_type)

    # サポート言語を表示（モデルが情報を持つ場合）
    supported_langs = model.get_supported_languages()
    if supported_langs:
        logger.info("Supported languages: %s", supported_langs)
    else:
        logger.info("Supported languages: (model does not restrict — any language accepted)")

    all_results: list[dict] = []

    if not args.skip_instruct:
        logger.info("\n--- instruct テスト (TC-VD-01〜04) ---")
        for tc in INSTRUCT_TEST_CASES:
            result = run_test_case(model, tc, output_dir)
            all_results.append(result)

    if not args.skip_language:
        logger.info("\n--- language パラメータ テスト (TC-VD-05〜07) ---")
        for tc in LANGUAGE_TEST_CASES:
            result = run_test_case(model, tc, output_dir)
            all_results.append(result)

    print_summary(all_results)

    all_passed = all(r["passed"] for r in all_results)
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
