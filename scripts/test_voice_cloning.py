"""
TC-06: Voice cloning test

Clones a speaker's voice from a reference WAV file (3-10 seconds, 24kHz mono)
and synthesizes new speech in the cloned voice.

Prerequisites:
    Place reference WAV files in /workspace/sample_audio/
    Audio must be clean, mono, at 24kHz sample rate.

    To convert any audio:
        ffmpeg -y -i input.mp3 -ss 3 -t 10 -ac 1 -ar 24000 sample_audio/reference.wav

Usage:
    docker compose run qwen-tts python3 scripts/test_voice_cloning.py
"""

import pathlib
import subprocess
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
    print("ERROR: soundfile not found.")
    sys.exit(1)

try:
    import whisper
except ImportError:
    print("ERROR: openai-whisper not found. Run: pip install openai-whisper")
    sys.exit(1)

try:
    from qwen_tts import QwenTTS
except ImportError:
    print("ERROR: qwen_tts not found. Run: pip install qwen-tts")
    sys.exit(1)


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SAMPLE_AUDIO_DIR = pathlib.Path("/workspace/sample_audio")
OUTPUT_BASE = pathlib.Path("/workspace/output")

TARGET_TEXTS = [
    "This is my cloned voice speaking a new sentence.",
    "音声クローニングのテストです。うまく声が複製できているか確認してください。",
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
    # Whisper runs on CPU on ARM64 (triton not available)
    print(f"Whisper device  : {'cuda' if torch.cuda.is_available() else 'cpu'} (transcription)")
    print()


def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg not found. Install with: apt-get install ffmpeg")
        return False


def prepare_audio(source_path: pathlib.Path, output_path: pathlib.Path) -> bool:
    """任意の音声ファイルを 24kHz mono WAV へ変換する。

    Args:
        source_path: 変換元の音声ファイルパス。
        output_path: 変換後の WAV 出力先パス。

    Returns:
        変換成功なら True、失敗なら False。
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "24000",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  ERROR: ffmpeg conversion failed: {result.stderr.decode()}")
        return False
    return True


def transcribe_audio(audio_path: pathlib.Path) -> str:
    """Whisper で参照音声を文字起こしする。

    Args:
        audio_path: 文字起こし対象の WAV ファイルパス。

    Returns:
        文字起こしテキスト。
    """
    print("  Transcribing reference audio with Whisper (base model)...")
    # ARM64: triton not available, Whisper uses CPU
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path))
    text = str(result["text"]).strip()
    print(f"  Transcript: {text}")
    return text


def clone_and_synthesize(
    tts_model: Any,
    reference_audio: pathlib.Path,
    reference_text: str,
    target_text: str,
    output_path: pathlib.Path,
) -> dict[str, Any]:
    """参照音声の声質でターゲットテキストを音声合成する。

    Args:
        tts_model: QwenTTS モデルインスタンス。
        reference_audio: 参照音声 WAV ファイルパス（24kHz mono）。
        reference_text: 参照音声の文字起こしテキスト。
        target_text: 合成対象のテキスト。
        output_path: 出力先 WAV ファイルパス。

    Returns:
        出力パス・再生時間・推論時間・最大振幅・サンプルレートを含む辞書。
    """
    start = time.time()
    # QwenTTS voice cloning API: pass reference audio path and transcript
    audio, sample_rate = tts_model(
        target_text,
        reference_audio=str(reference_audio),
        reference_text=reference_text,
    )
    elapsed = time.time() - start

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

    if not check_ffmpeg():
        sys.exit(1)

    # Discover reference audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    sample_files = [
        f for f in SAMPLE_AUDIO_DIR.iterdir() if f.suffix.lower() in audio_extensions and f.name != ".gitkeep"
    ]

    if not sample_files:
        print("ERROR: No reference audio files found in /workspace/sample_audio/")
        print()
        print("To prepare a reference audio file, run:")
        print("  ffmpeg -y -i input.mp3 -ss 3 -t 10 -ac 1 -ar 24000 sample_audio/reference.wav")
        print()
        print("Then re-run this test.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / f"voice_cloning_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Found {len(sample_files)} reference audio file(s).\n")

    print(f"Loading model: {MODEL_ID}")
    tts_model = QwenTTS(MODEL_ID)
    print("Model loaded.\n")

    results: list[dict[str, Any]] = []
    for ref_file in sample_files:
        print(f"[TC-06] Voice cloning — reference: {ref_file.name}")

        # Convert reference audio to 24kHz mono WAV
        prepared_path = output_dir / f"prepared_{ref_file.stem}.wav"
        print("  Preparing audio (24kHz mono WAV)...")
        if not prepare_audio(ref_file, prepared_path):
            print(f"  FAIL: Could not prepare {ref_file.name}")
            results.append({"ref": ref_file.name, "passed": False})
            continue

        # Transcribe
        try:
            reference_text = transcribe_audio(prepared_path)
        except RuntimeError as e:
            print(f"  FAIL: Transcription error: {e}")
            results.append({"ref": ref_file.name, "passed": False})
            continue

        # Synthesize each target text
        for i, target_text in enumerate(TARGET_TEXTS):
            print(f"  Target [{i + 1}]: {target_text}")
            output_path = output_dir / f"cloned_{ref_file.stem}_{i + 1}.wav"

            try:
                result = clone_and_synthesize(tts_model, prepared_path, reference_text, target_text, output_path)
                passed = result["max_amplitude"] > 0.01
                status = "PASS" if passed else "FAIL"
                print(f"  Output  : {result['output_path'].name}")
                print(f"  Duration: {result['duration_sec']:.2f}s")
                print(f"  Latency : {result['inference_sec']:.2f}s")
                print(f"  Max amp : {result['max_amplitude']:.4f}")
                print(f"  Status  : {status}")
                results.append({"ref": ref_file.name, "target": i + 1, "passed": passed, **result})
            except RuntimeError as e:
                print(f"  FAIL: Synthesis error: {e}")
                results.append({"ref": ref_file.name, "target": i + 1, "passed": False})

        print()

    print("=" * 60)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"Results: {passed_count}/{len(results)} passed")
    print("Note: Audio quality (voice similarity) requires manual subjective evaluation.")
    print("=" * 60)

    if passed_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
