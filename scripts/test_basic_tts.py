"""
TC-01, TC-02: Basic TTS test (English)

Tests text-to-speech synthesis with English input using Qwen3-TTS.
Saves output WAV files to /workspace/output/basic_tts_<timestamp>/

Note: Qwen3-TTS-12Hz-1.7B-Base requires a reference audio for all synthesis.
      Place a 3-10 second mono 24kHz WAV at /workspace/sample_audio/reference.wav
      before running this script.

      ffmpeg -y -i input.mp3 -ss 3 -t 10 -ac 1 -ar 24000 sample_audio/reference.wav

Usage:
    docker compose run qwen-tts python3 scripts/test_basic_tts.py
"""

import pathlib
import sys
import time
from datetime import datetime
from typing import Any

try:
    import numpy as np
    import torch
except ImportError:
    print("ERROR: torch/numpy not found. Are you running inside the Docker container?")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile not found. Run: pip install soundfile")
    sys.exit(1)

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("ERROR: qwen_tts not found. Run: pip install qwen-tts")
    sys.exit(1)

from model_utils import ensure_model_downloaded

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
OUTPUT_BASE = pathlib.Path("/workspace/output")
REFERENCE_AUDIO = pathlib.Path("/workspace/sample_audio/reference.wav")

TEST_CASES = [
    {
        "id": "TC-01",
        "name": "Short English sentence",
        "text": "Hello, this is a test of the Qwen TTS system running on DGX Spark.",
    },
    {
        "id": "TC-02",
        "name": "Medium English paragraph",
        "text": (
            "Qwen3-TTS is an open-source text-to-speech model developed by Alibaba Cloud. "
            "It features a 1.7 billion parameter architecture and supports multiple languages "
            "including Japanese and English. The model is capable of voice cloning from just "
            "a few seconds of reference audio."
        ),
    },
]


def check_environment() -> None:
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"SM capability   : sm_{cap[0]}{cap[1]}")
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory      : {mem_total:.1f} GB")
    else:
        print("WARNING: CUDA not available — inference will be slow on CPU")
    print()


def synthesize(
    model: Any,
    text: str,
    output_path: pathlib.Path,
    ref_audio: str,
    ref_text: str,
) -> dict[str, Any]:
    """テキストを音声合成し、WAV ファイルへ保存する。

    Args:
        model: Qwen3TTSModel インスタンス。
        text: 合成対象のテキスト。
        output_path: 出力先 WAV ファイルパス。
        ref_audio: 参照音声ファイルパス（24kHz mono WAV）。
        ref_text: 参照音声の書き起こしテキスト。

    Returns:
        出力パス・再生時間・推論時間・最大振幅・サンプルレートを含む辞書。
    """
    start = time.time()
    wavs, sample_rate = model.generate_voice_clone(
        text,
        ref_audio=ref_audio,
        ref_text=ref_text if ref_text else None,
        # ref_text なしの場合は speaker embedding のみで合成（x_vector_only_mode）
        x_vector_only_mode=not bool(ref_text),
    )
    elapsed = time.time() - start

    audio = wavs[0]
    sf.write(str(output_path), audio, sample_rate)

    max_amplitude = float(np.abs(audio).max())
    duration = len(audio) / sample_rate

    return {
        "output_path": output_path,
        "duration_sec": duration,
        "inference_sec": elapsed,
        "max_amplitude": max_amplitude,
        "sample_rate": sample_rate,
    }


def run_tests() -> None:
    check_environment()

    if not REFERENCE_AUDIO.exists():
        print(f"ERROR: Reference audio not found: {REFERENCE_AUDIO}")
        print()
        print("The Base model requires a reference audio for all synthesis.")
        print("To prepare one, run:")
        print("  ffmpeg -y -i input.mp3 -ss 3 -t 10 -ac 1 -ar 24000 sample_audio/reference.wav")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / f"basic_tts_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Reference audio : {REFERENCE_AUDIO}")
    print()

    print(f"Loading model: {MODEL_ID}")
    model_path = ensure_model_downloaded(MODEL_ID)
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        max_memory={0: "60GiB"},
    )
    print("Model loaded.\n")

    # Use empty string as ref_text to trigger x-vector only mode (speaker embedding only).
    # This avoids requiring a transcript of the reference audio for basic TTS tests.
    ref_text = ""

    results = []
    for tc in TEST_CASES:
        print(f"[{tc['id']}] {tc['name']}")
        print(f"  Input: {tc['text'][:80]}{'...' if len(tc['text']) > 80 else ''}")

        output_path = output_dir / f"{tc['id'].lower().replace('-', '_')}.wav"
        result = synthesize(model, tc["text"], output_path, str(REFERENCE_AUDIO), ref_text)

        passed = result["max_amplitude"] > 0.01
        status = "PASS" if passed else "FAIL"

        print(f"  Output  : {result['output_path'].name}")
        print(f"  Duration: {result['duration_sec']:.2f}s")
        print(f"  Latency : {result['inference_sec']:.2f}s")
        print(f"  Max amp : {result['max_amplitude']:.4f}")
        print(f"  Status  : {status}")
        print()

        results.append({**tc, **result, "passed": passed})

    print("=" * 60)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"Results: {passed_count}/{len(results)} passed")
    print("=" * 60)

    if passed_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
