# qwen-tts-validation

NVIDIA DGX Spark (GB10 GPU, CUDA 13.0, ARM64) 上で [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) を Docker 環境で動作させるための検証プロジェクト。

参考記事: [Qwen3 TTSで声を複製してみた](https://zenn.dev/karaage0703/articles/97f8a01cbb9c49)

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
| RAM | 121 GB |
| OS | Ubuntu 24.04 LTS (ARM64) |
| Docker | 29.1.3 |
| Python | 3.12.3 |
| ベースイメージ | `nvcr.io/nvidia/pytorch:25.03-py3` |
| PyTorch | 2.7.0 (CUDA 12.8 toolkit) |

### なぜ記事と異なるベースイメージを使うのか

参考記事では `PyTorch 2.9.1 cu130` を使用しているが、本プロジェクトでは
`nvcr.io/nvidia/pytorch:25.03-py3` を採用している。この NGC イメージは
GB10 (sm_121) 向けの PTX から JIT コンパイル済みで、CUDA 13.0 ドライバとの
forward-compatibility が DGX Spark 上で確認済みのため。

---

## 検証内容

詳細は [docs/validation-plan.md](docs/validation-plan.md) を参照。

| TC | 内容 |
|----|------|
| TC-01 | 英語短文 TTS |
| TC-02 | 英語段落 TTS |
| TC-03 | 日本語ひらがな/カタカナ |
| TC-04 | 日本語漢字混じり |
| TC-05 | 日英混在テキスト |
| TC-06 | ボイスクローニング |
| TC-07 | GPU 利用確認 |

---

## クイックスタート

### 前提条件

- Docker 29.x + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU ドライバ 580+
- (オプション) `HF_TOKEN` 環境変数 (公開モデルは不要)

### ビルド

```bash
docker compose build
```

初回ビルドは NGC ベースイメージのダウンロードが含まれるため時間がかかります。

### テスト実行

```bash
# 基本 TTS テスト (英語)
docker compose run qwen-tts python3 scripts/test_basic_tts.py

# 日本語テスト
docker compose run qwen-tts python3 scripts/test_japanese.py

# ボイスクローニングテスト (sample_audio/ に WAV を配置してから実行)
docker compose run qwen-tts python3 scripts/test_voice_cloning.py
```

### 全テスト一括実行

```bash
docker compose run qwen-tts bash -c "
  python3 scripts/test_basic_tts.py &&
  python3 scripts/test_japanese.py &&
  python3 scripts/test_voice_cloning.py
"
```

### インタラクティブシェル

```bash
docker compose run qwen-tts bash
```

---

## ボイスクローニングの準備

`sample_audio/` ディレクトリに参照音声を配置してください。

```bash
# 任意の音声ファイルを 24kHz mono WAV に変換 (3〜10秒)
ffmpeg -y -i input.mp3 -ss 3 -t 10 -ac 1 -ar 24000 sample_audio/reference.wav
```

**推奨仕様:**
- 長さ: 3〜10 秒（10秒がより高品質）
- チャンネル: モノラル
- サンプルレート: 24kHz
- ノイズ: 少ないほど良い

> **注意**: `sample_audio/*.wav` は `.gitignore` に含まれており、個人の音声はリポジトリにコミットされません。

---

## 生成音声の確認

生成された WAV ファイルは `output/` ディレクトリに保存されます（Docker volume と bind mount で同期）。

```bash
# ホスト上で再生
aplay output/basic_tts_*/tc_01_*.wav
# または
ffplay output/basic_tts_*/tc_01_*.wav
```

---

## トラブルシューティング

### CUDA がコンテナ内で使えない

```
RuntimeError: CUDA not available
```

NVIDIA Container Toolkit が正しく設定されているか確認:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### モデルダウンロードが失敗する

ネットワーク接続または HuggingFace のレート制限を確認。必要に応じて:

```bash
export HF_TOKEN=<your_token>
docker compose run qwen-tts python3 scripts/test_basic_tts.py
```

### sox 関連のエラー

Dockerfile に `libsox-dev` が含まれているか確認し、イメージを再ビルド:

```bash
docker compose build --no-cache
```

### torchaudio バージョン不一致

base image の PyTorch 2.7.0 と torchaudio 2.7.0 が一致しているか `requirements.txt` を確認。

---

## 開発フロー（GitHub Flow）

このリポジトリは **GitHub Flow** を採用しています。

### 基本ルール

- `main` ブランチは常にクリーンな状態を保つ（直接コミット禁止）
- 作業は Issue に対応したフィーチャーブランチで行う
- すべての変更は Pull Request 経由でマージする

### ブランチ命名

```
<type>/issue-<N>-<短い説明>
```

例: `feature/issue-1-fix-torchaudio-cuda` / `fix/issue-2-transformers-warning` / `docs/issue-12-validation-results`

### 作業手順

```bash
# 1. main から作業ブランチを作成
git switch -c feature/issue-N-description

# 2. 作業・コミット
git add <files>
git commit -m "fix: torchaudio cuda compatibility"

# 3. PR を作成（Issue を自動クローズ）
gh pr create --title "..." --body "Closes #N"

# 4. レビュー・CI 確認後にマージ → ブランチ削除
```

詳細なルールは [CLAUDE.md](CLAUDE.md) の「Git 運用戦略」セクションを参照。

---

## ライセンス

- **Qwen3-TTS モデル**: [Apache 2.0](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- **このリポジトリ**: MIT
