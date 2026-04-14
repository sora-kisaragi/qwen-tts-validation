# qwen-tts-validation

NVIDIA DGX Spark (GB10 GPU, CUDA 13.0, ARM64) 上で [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) を Docker 環境で動作させるための検証プロジェクト。

参考記事: [Qwen3 TTSで声を複製してみた](https://zenn.dev/karaage0703/articles/97f8a01cbb9c49)

---

## 検証結果サマリー

**全 TC PASS (8/8)** — 2026-04-14 検証完了

| TC | 内容 | 結果 | Duration | Latency |
|----|------|------|----------|---------|
| TC-01 | 英語短文 TTS | ✅ PASS | 5.84s | 20.29s |
| TC-02 | 英語段落 TTS | ✅ PASS | 19.68s | 15.00s |
| TC-03 | 日本語ひらがな/カタカナ | ✅ PASS | 6.40s | 19.87s |
| TC-04 | 日本語漢字混じり | ✅ PASS | 8.48s | 6.58s |
| TC-05 | 日英混在テキスト | ✅ PASS | 8.72s | 6.58s |
| TC-06 | ボイスクローニング (英語) | ✅ PASS | 3.12s | 3.39s |
| TC-06 | ボイスクローニング (日本語) | ✅ PASS | 6.24s | 4.86s |
| TC-07 | GPU 利用確認 | ✅ PASS | — | — |

詳細は [docs/validation-plan.md](docs/validation-plan.md) を参照。

---

## Qwen3-TTS について

- **モデル**: Qwen/Qwen3-TTS-12Hz-1.7B-Base
- **パラメータ数**: 1.7B
- **ライセンス**: Apache 2.0
- **開発**: Alibaba Cloud Qwen Team
- **特徴**:
  - 多言語対応（日本語含む）
  - わずか 3〜10 秒の参照音声からボイスクローニングが可能

---

## 検証環境

| 項目 | 値 |
|------|-----|
| マシン | NVIDIA DGX Spark |
| GPU | NVIDIA GB10 (Blackwell, sm_121) |
| CUDA ドライバ | 580.126.09 (CUDA 13.0) |
| RAM | 121.7 GB |
| OS | Ubuntu 24.04 LTS (ARM64) |
| Docker | 29.1.3 |
| Python | 3.12.3 |
| ベースイメージ | `ubuntu:24.04` |
| PyTorch | 2.9.1+cu130 |

---

## ファイル構成

```
qwen-tts-validation/
├── Dockerfile                      # ubuntu:24.04 + torch 2.9.1+cu130
├── docker-compose.yml              # GPU パススルー + volume 定義
├── requirements.txt                # qwen-tts 0.1.1 他
├── scripts/
│   ├── model_utils.py              # モデルローダー・話者プロファイル保存/ロード
│   ├── test_basic_tts.py           # TC-01, TC-02: 英語 TTS
│   ├── test_japanese.py            # TC-03〜TC-05: 日本語 TTS
│   ├── test_voice_cloning.py       # TC-06: ボイスクローニング
│   └── create_speaker_profile.py  # 話者プロファイル (.pt) 生成ユーティリティ
├── sample_audio/                   # 参照音声 WAV を置く場所 (.gitignore 対象)
├── speaker_profiles/               # 話者プロファイル .pt (.gitignore 対象)
├── output/                         # 生成音声の出力先 (.gitignore 対象)
└── docs/
    └── validation-plan.md          # テスト計画・検証結果
```

---

## クイックスタート

### 前提条件

- Docker 29.x + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU ドライバ 580+
- (オプション) `HF_TOKEN` 環境変数（公開モデルは不要）

### ビルド

```bash
docker compose build
```

### テスト実行

```bash
# 基本 TTS テスト (英語)
docker compose run qwen-tts python3 scripts/test_basic_tts.py

# 日本語テスト
docker compose run qwen-tts python3 scripts/test_japanese.py
```

ボイスクローニングテストは参照音声の準備が必要です（下記「ボイスクローニングの準備」参照）。

### 全テスト一括実行

参照音声を準備した上で:

```bash
docker compose run qwen-tts bash -c "
  python3 scripts/test_basic_tts.py &&
  python3 scripts/test_japanese.py &&
  python3 scripts/test_voice_cloning.py \
    --ref-audio sample_audio/reference.wav \
    --ref-text '参照音声の内容'
"
```

### インタラクティブシェル

```bash
docker compose run qwen-tts bash
```

---

## ボイスクローニングの準備

### 参照音声の変換

```bash
# 任意の音声ファイルを 24kHz mono WAV に変換
ffmpeg -y -i input.mp3 -ac 1 -ar 24000 sample_audio/reference.wav
```

**推奨仕様:** 3〜10 秒、モノラル、24kHz、ノイズ少なめ

> **注意**: `sample_audio/*.wav` は `.gitignore` に含まれており、リポジトリにコミットされません。

### 実行モード

**モード (A): テキスト手動入力**（Whisper 不要・ICL モード・推奨）

```bash
docker compose run qwen-tts python3 scripts/test_voice_cloning.py \
  --ref-audio sample_audio/reference.wav \
  --ref-text "参照音声で話している内容をここに入力"
```

**モード (B): Whisper 自動文字起こし**（参照音声の内容が不明な場合）

```bash
docker compose run qwen-tts python3 scripts/test_voice_cloning.py \
  --ref-audio sample_audio/reference.wav
```

**モード (C): 話者プロファイル再利用**（最速・同じ話者を繰り返し使う場合）

```bash
# プロファイルを一度だけ生成
docker compose run qwen-tts python3 scripts/create_speaker_profile.py \
  --ref-audio sample_audio/reference.wav \
  --output speaker_profiles/default.pt

# 以降はプロファイルを再利用（参照音声の再処理なし）
docker compose run qwen-tts python3 scripts/test_voice_cloning.py \
  --profile speaker_profiles/default.pt
```

> **注意**: `speaker_profiles/*.pt` は `.gitignore` に含まれており、リポジトリにコミットされません。

---

## 生成音声の確認

生成された WAV ファイルは `output/` ディレクトリに保存されます（bind mount でホストと同期）。

```bash
# ホスト上で再生
aplay output/basic_tts_*/tc_01.wav
# または
ffplay output/basic_tts_*/tc_01.wav
```

---

## トラブルシューティング

### CUDA がコンテナ内で使えない

```
RuntimeError: CUDA not available
```

NVIDIA Container Toolkit が正しく設定されているか確認:

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

### GB10 GPU の警告が出る

```
UserWarning: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

torch 2.9.1 の公式サポートは sm_120 までだが、PTX JIT フォールバックにより sm_121 (GB10) でも正常動作する。警告は無視して問題ない（推論は GPU で実行される）。

### モデルダウンロードが失敗する

ネットワーク接続または HuggingFace のレート制限を確認。必要に応じて:

```bash
export HF_TOKEN=<your_token>
docker compose run qwen-tts python3 scripts/test_basic_tts.py
```

### speech_tokenizer/ のファイルが見つからない

```
OSError: Can't load feature extractor ... preprocessor_config.json
```

`snapshot_download` はサブディレクトリのキャッシュが不完全な場合がある。`ensure_model_downloaded()` (model_utils.py) が `hf_hub_download` で各ファイルを個別に取得するため、通常は自動回復する。手動で解消する場合:

```bash
docker compose run qwen-tts python3 -c "
from model_utils import ensure_model_downloaded
ensure_model_downloaded('Qwen/Qwen3-TTS-12Hz-1.7B-Base')
"
```

### sox 関連のエラー

Dockerfile に `libsox-dev` が含まれているか確認し、イメージを再ビルド:

```bash
docker compose build --no-cache
```

---

## ライセンス

- **Qwen3-TTS モデル**: [Apache 2.0](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- **このリポジトリ**: MIT
