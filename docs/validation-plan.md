# 検証計画: Qwen3-TTS on DGX Spark

## 目的

Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-1.7B-Base, 1.7B パラメータ, Apache 2.0 ライセンス) を
Docker 環境上で NVIDIA DGX Spark (GB10 GPU) で安定して動作させることを確認する。

参考記事: https://zenn.dev/karaage0703/articles/97f8a01cbb9c49

---

## 検証対象環境

| 項目 | 値 |
|------|-----|
| マシン | NVIDIA DGX Spark |
| GPU | NVIDIA GB10 (Blackwell, sm_121) |
| CUDA ドライバ | 580.126.09 (CUDA 13.0 表示) |
| コンテナ base image | nvcr.io/nvidia/pytorch:25.03-py3 |
| PyTorch | 2.7.0a0 (CUDA 12.8 toolkit) |
| Python | 3.12.3 |
| アーキテクチャ | ARM64 (aarch64) |
| OS | Ubuntu 24.04 LTS |
| RAM | 121 GB |

### 記事との環境の差異

| 項目 | 記事 | 本検証 |
|------|------|--------|
| PyTorch | 2.9.1 cu130 | 2.7.0 (NGC base image) |
| インストール方法 | uv + venv | Docker |
| CUDA toolkit | 13.0 | 12.8 (driver 13.0 forward-compat) |

---

## テストケース

### TC-01: 英語短文 TTS

- **スクリプト**: `scripts/test_basic_tts.py`
- **入力**: "Hello, this is a test of the Qwen TTS system running on DGX Spark."
- **期待結果**: WAV ファイル生成、非無音、エラーなし
- **合否基準**: `max_amplitude > 0.01`、例外なし
- **出力先**: `output/basic_tts_<timestamp>/tc_01.wav`

### TC-02: 英語段落 TTS

- **スクリプト**: `scripts/test_basic_tts.py`
- **入力**: Qwen3-TTS に関する3文の英語段落
- **期待結果**: 5秒以上の WAV、クリッピングなし
- **合否基準**: `duration_sec >= 5.0`、`max_amplitude > 0.01`
- **出力先**: `output/basic_tts_<timestamp>/tc_02.wav`

### TC-03: 日本語 ひらがな/カタカナ

- **スクリプト**: `scripts/test_japanese.py`
- **入力**: "こんにちは、これはQwen音声合成システムのテストです。よろしくお願いします。"
- **期待結果**: 日本語として認識できる音声出力
- **合否基準**: `max_amplitude > 0.01`、`duration_sec >= 1.0`、例外なし
- **出力先**: `output/japanese_<timestamp>/tc_03.wav`

### TC-04: 日本語 漢字混じり

- **スクリプト**: `scripts/test_japanese.py`
- **入力**: "音声合成の技術は急速に発展しています。自然な日本語の発音と抑揚が再現できるか確認します。"
- **期待結果**: 自然なイントネーションの日本語音声
- **合否基準**: `max_amplitude > 0.01`、例外なし（音質は主観評価）
- **出力先**: `output/japanese_<timestamp>/tc_04.wav`

### TC-05: 日英混在テキスト

- **スクリプト**: `scripts/test_japanese.py`
- **入力**: "今日はQwen TTSのテストをしています。This is a mixed language test combining Japanese and English."
- **期待結果**: 日本語・英語ともに出力
- **合否基準**: `max_amplitude > 0.01`、例外なし
- **出力先**: `output/japanese_<timestamp>/tc_05.wav`

### TC-06: ボイスクローニング

- **スクリプト**: `scripts/test_voice_cloning.py`
- **事前準備**: `sample_audio/` に参照音声 WAV を配置
  ```bash
  ffmpeg -y -i input.mp3 -ss 3 -t 10 -ac 1 -ar 24000 sample_audio/reference.wav
  ```
- **処理フロー**:
  1. ffmpeg で参照音声を 24kHz mono WAV に変換
  2. Whisper (base model, CPU) で文字起こし
  3. 参照音声 + 文字起こしで声を複製し、新しいテキストを合成
- **ターゲットテキスト**:
  - "This is my cloned voice speaking a new sentence."
  - "音声クローニングのテストです。うまく声が複製できているか確認してください。"
- **合否基準**: `max_amplitude > 0.01`（音声類似度は主観評価）
- **出力先**: `output/voice_cloning_<timestamp>/cloned_reference_1.wav` 等

### TC-07: GPU 利用確認

- **確認方法**: 各テストスクリプト実行中に別ターミナルで `nvidia-smi` を実行
- **期待結果**: GPU メモリ使用量 > 1GB (モデルロード後)
- **コード確認**: `torch.cuda.memory_allocated()` が推論中に 0 より大きい値を示す

---

## 既知の制約事項

| 制約 | 内容 | 対処 |
|------|------|------|
| triton (ARM64) | ARM64 で triton 不可 | Whisper は CPU で動作（許容範囲） |
| transformers ダウングレード | qwen-tts 0.1.1 が 4.57.3 を要求 | Dockerfile で明示的にピン留め |
| 音声品質評価 | 自動指標なし | WAV 再生による主観評価 |
| ボイスクローニング参照音声 | ユーザーが用意する必要あり | sample_audio/ に配置手順を README に記載 |

---

## 成功基準

- TC-01 〜 TC-05: 全テストが自動的に PASS
- TC-06: WAV が非無音（音声類似度は主観評価で合格判定）
- TC-07: GPU メモリが推論中に使用されていることを確認

---

## 検証結果記録

テスト実行後、以下の表を埋めてください。

| TC | テスト名 | 実行日時 | 結果 | 出力ファイル | 備考 |
|----|---------|---------|------|------------|------|
| TC-01 | 英語短文 TTS | | | | |
| TC-02 | 英語段落 TTS | | | | |
| TC-03 | 日本語ひらがな/カタカナ | | | | |
| TC-04 | 日本語漢字混じり | | | | |
| TC-05 | 日英混在 | | | | |
| TC-06 | ボイスクローニング | | | | |
| TC-07 | GPU 利用確認 | | | | |
