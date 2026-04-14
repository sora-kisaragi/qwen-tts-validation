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
| CUDA ドライバ | 580.126.09 (CUDA 13.0) |
| コンテナ base image | ubuntu:24.04 |
| PyTorch | 2.9.1+cu130 |
| Python | 3.12.3 |
| アーキテクチャ | ARM64 (aarch64) |
| OS | Ubuntu 24.04 LTS |
| RAM | 121.7 GB |

### 記事との環境の差異

| 項目 | 記事 | 本検証 |
|------|------|--------|
| PyTorch | 2.9.1 cu130 | 2.9.1+cu130（同一） |
| インストール方法 | uv + venv | Docker |
| CUDA toolkit | 13.0 | cu130 wheels（libcudart.so.13 同梱） |

> **注意**: 当初 NGC 25.03 ベースイメージ（PyTorch 2.7.0, CUDA 12.8）を使用しようとしたが、
> ARM64 向け torchaudio cu130 wheels が libcudart.so.13 をリンクするため ABI 不一致が発生した（Issue #1）。
> ubuntu:24.04 + torch 2.9.1+cu130 に切り替えることで解消した（PR #19）。

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
  ffmpeg -y -i input.mp3 -ac 1 -ar 24000 sample_audio/reference.wav
  ```
- **実行モード** (いずれか選択):

  | モード | コマンド | Whisper |
  |--------|---------|---------|
  | (A) テキスト手動入力 | `--ref-audio <wav> --ref-text "<text>"` | 不要 |
  | (B) Whisper 自動文字起こし | `--ref-audio <wav>` | 必要 |
  | (C) 事前プロファイル再利用 | `--profile speaker_profiles/default.pt` | 不要 |

- **話者プロファイル事前生成**:
  ```bash
  python3 scripts/create_speaker_profile.py \
    --ref-audio sample_audio/reference.wav \
    --output speaker_profiles/default.pt
  ```
- **ターゲットテキスト**:
  - "This is my cloned voice speaking a new sentence."
  - "音声クローニングのテストです。うまく声が複製できているか確認してください。"
- **合否基準**: `max_amplitude > 0.01`（音声類似度は主観評価）
- **出力先**: `output/voice_cloning_<timestamp>/cloned_<name>_1.wav` 等

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
| **日本語漢字の中国語読み混入** | 多言語モデルのため CJK 共通漢字（例: 音声合成）が中国語発音になる場合がある。自動合否には影響しないが発音品質として不適切 | 即時対応は見送り。長期対策として pykakasi による漢字→ひらがな前処理、`language="ja"` 明示指定などを検討（Issue #26） |

---

## 成功基準

- TC-01 〜 TC-05: 全テストが自動的に PASS
- TC-06: WAV が非無音（音声類似度は主観評価で合格判定）
- TC-07: GPU メモリが推論中に使用されていることを確認

---

## 検証結果記録

### 実行環境（最終）

| 項目 | 値 |
|------|-----|
| 実行日 | 2026-04-14 |
| base image | ubuntu:24.04 |
| PyTorch | 2.9.1+cu130 |
| GPU | NVIDIA GB10 (sm_121, 121.7 GB) |

### テスト結果

| TC | テスト名 | 結果 | Duration | Latency | Max Amp | 備考 |
|----|---------|------|----------|---------|---------|------|
| TC-01 | 英語短文 TTS | ✅ PASS | 5.84s | 20.29s | 0.9680 | |
| TC-02 | 英語段落 TTS | ✅ PASS | 19.68s | 15.00s | 0.9970 | |
| TC-03 | 日本語ひらがな/カタカナ | ✅ PASS | 6.40s | 19.87s | 0.2152 | |
| TC-04 | 日本語漢字混じり | ✅ PASS | 8.48s | 6.58s | 0.1823 | |
| TC-05 | 日英混在 | ✅ PASS | 8.72s | 6.58s | 0.1781 | |
| TC-06 | ボイスクローニング (英語) | ✅ PASS | 3.12s | 3.39s | 0.1432 | 主観評価: 声質類似確認 |
| TC-06 | ボイスクローニング (日本語) | ✅ PASS | 6.24s | 4.86s | 0.1530 | 主観評価: 声質類似確認 |
| TC-07 | GPU 利用確認 | ✅ PASS | — | — | — | CUDA available, sm_121, 121.7 GB |

### 総合結果

**全 TC PASS** (8/8)

### 発見事項・トラブルシュート

| Issue | 内容 | 対処 | PR |
|-------|------|------|----|
| #1 | torchaudio CUDA 非互換 (libcudart.so.13) | ubuntu:24.04 + cu130 wheels に切り替え | #19 |
| #2 | TRANSFORMERS_CACHE 非推奨警告 | docker-compose.yml から削除 | #20 |
| #3 | GB10 GPU "may not yet be supported" 警告 | NGC 切り替えで解消。PTX JIT により sm_121 動作確認 | — |
| #21 | 参照音声の毎回再処理が非効率 | 話者プロファイル (.pt) のテンプレート化を実装 | #22 |
