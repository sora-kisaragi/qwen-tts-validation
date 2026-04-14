"""
話者プロファイル生成ユーティリティ

- 目的: 参照音声 WAV から VoiceClonePromptItem を生成し .pt ファイルに保存する
- 対象: ユーザーが話者テンプレートを事前作成する際に使用
- 関連: Issue #21 — 話者プロファイルのテンプレート化と再利用

作成者: 宗廣 颯真
作成日: 2026-04-14
最終更新者: 宗廣 颯真
最終更新日: 2026-04-14

Usage:
    docker compose run qwen-tts python3 scripts/create_speaker_profile.py \\
        --ref-audio sample_audio/reference.wav \\
        --output speaker_profiles/default.pt
"""

import argparse
import pathlib
import sys

try:
    import torch
except ImportError:
    print("ERROR: torch not found. Are you running inside the Docker container?")
    sys.exit(1)

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("ERROR: qwen_tts not found. Run: pip install qwen-tts")
    sys.exit(1)

from model_utils import ensure_model_downloaded, save_speaker_profile

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a speaker profile (.pt) from a reference WAV file.")
    parser.add_argument(
        "--ref-audio",
        type=pathlib.Path,
        default=pathlib.Path("/workspace/sample_audio/reference.wav"),
        help="Reference WAV file (24kHz mono, 3-10s). Default: /workspace/sample_audio/reference.wav",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("/workspace/speaker_profiles/default.pt"),
        help="Output .pt file path. Default: /workspace/speaker_profiles/default.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ref_audio.exists():
        print(f"ERROR: Reference audio not found: {args.ref_audio}")
        print("Prepare with: ffmpeg -y -i input.mp3 -ac 1 -ar 24000 sample_audio/reference.wav")
        sys.exit(1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device         : {device}")
    print(f"Reference audio: {args.ref_audio}")
    print(f"Output profile : {args.output}")
    print()

    print(f"Loading model: {MODEL_ID}")
    model_path = ensure_model_downloaded(MODEL_ID)
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        max_memory={0: "60GiB"},
    )
    print("Model loaded.\n")

    print("Extracting speaker embedding...")
    # x_vector_only_mode=True: 参照音声のトランスクリプト不要（話者埋め込みのみ使用）
    prompts = model.create_voice_clone_prompt(
        ref_audio=str(args.ref_audio),
        x_vector_only_mode=True,
    )
    # create_voice_clone_prompt はリストを返す
    profile = prompts[0] if isinstance(prompts, list) else prompts

    save_speaker_profile(profile, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
