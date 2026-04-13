# CLAUDE.md — Claude Code Guidance for qwen-tts-validation

このファイルは Claude Code がこのリポジトリで作業する際のガイドラインを提供します。

---

## Project Purpose

NVIDIA DGX Spark (GB10 GPU, CUDA 13.0 driver, ARM64 Ubuntu 24.04) 上で
Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-1.7B-Base, 1.7B params, Apache 2.0) を
Docker 環境で動作させ、以下を検証するプロジェクト:

- 基本的な TTS 動作（テキスト → 音声）
- ボイスクローニング（参照音声 3〜10秒 → 声の複製）
- 日本語対応の品質確認

参考記事: https://zenn.dev/karaage0703/articles/97f8a01cbb9c49

---

## Docker Conventions

<!-- TODO: ユーザーの Docker 規約をここに記載してください -->
<!-- 例: イメージ命名規則、volume の扱い、GPU パススルーの方法など -->

[PLACEHOLDER — Docker の個人規約・手順書を後で提供してください]

---

## Python Conventions

<!-- TODO: ユーザーの Python 規約をここに記載してください -->
<!-- 例: コーディングスタイル、パッケージ管理、型ヒントの方針など -->

[PLACEHOLDER — Python の個人規約・手順書を後で提供してください]

---

## Key Technical Context

| 項目 | 値 |
|------|-----|
| Base image | `nvcr.io/nvidia/pytorch:25.03-py3` |
| PyTorch | 2.7.0a0 (CUDA 12.8 toolkit 収録) |
| GPU | NVIDIA GB10 (Blackwell, sm_121) |
| CUDA driver | 13.0 (580.126.09) — forward-compat 経由で動作 |
| Python | 3.12.3 |
| Architecture | ARM64 (aarch64) |
| qwen-tts | 0.1.1 (transformers==4.57.3 に依存 → base image の 5.3.0 からダウングレード) |
| openai-whisper | ARM64 では triton 不可 → CPU で文字起こし実行 |

**なぜ `nvcr.io/nvidia/pytorch:25.03-py3` を使うか:**
記事では PyTorch 2.9.1 cu130 を使用しているが、この NGC イメージは
GB10 (sm_121) の PTX から JIT コンパイル済みで CUDA 13.0 ドライバとの
forward-compatibility が確認済みのため、同等以上の動作が得られる。

---

## File Map

| ファイル | 役割 |
|---------|------|
| `Dockerfile` | イメージ定義 (NGC base + qwen-tts + whisper) |
| `docker-compose.yml` | GPU パススルー + volume マウント定義 |
| `requirements.txt` | pip インストール対象パッケージ一覧 |
| `scripts/test_basic_tts.py` | 基本 TTS テスト (英語) |
| `scripts/test_japanese.py` | 日本語対応テスト |
| `scripts/test_voice_cloning.py` | ボイスクローニングテスト |
| `sample_audio/` | 参照音声 WAV を置く場所 (gitignore 対象) |
| `output/` | 生成音声の出力先 (gitignore 対象、bind mount) |
| `docs/validation-plan.md` | テスト計画と期待結果 |

---

## Running Tests

```bash
# イメージビルド
docker compose build

# 基本 TTS テスト
docker compose run qwen-tts python3 scripts/test_basic_tts.py

# 日本語テスト
docker compose run qwen-tts python3 scripts/test_japanese.py

# ボイスクローニングテスト (sample_audio/ に WAV を置いてから実行)
docker compose run qwen-tts python3 scripts/test_voice_cloning.py

# インタラクティブシェル
docker compose run qwen-tts bash
```

## Voice Cloning Setup

```bash
# 参照音声を準備 (3〜10秒, 24kHz mono WAV に変換)
ffmpeg -y -i input.mp3 -ss 3 -t 10 -ac 1 -ar 24000 sample_audio/reference.wav
```
