# 公式 GitHub 調査レポート

調査日: 2026-04-14
対象: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

---

## 1. ファインチューニング（Issue #36 に直結）

### 公式サポート状況

| 項目 | 内容 |
|------|------|
| 対象モデル | `Qwen3-TTS-12Hz-1.7B/0.6B-Base` |
| 現在の制約 | 単一話者のみ |
| ロードマップ | マルチ話者対応を予定 |
| スクリプト | `finetuning/prepare_data.py`, `sft_12hz.py`, `dataset.py` |

### なぜファインチューニングで「参照音声 + instruct」が実現できるのか

```
通常の使い方:
  参照音声 → generate_voice_clone() → instruct は使えない

ファインチューニング後:
  自分の声の学習データ → Base モデルを SFT → カスタム話者として登録
                                              ↓
                          generate_custom_voice(speaker="my_voice", instruct="...")
                          → instruct が使える！
```

### 学習データ形式（JSONL）

```jsonl
{"audio": "data/001.wav", "text": "話している内容のテキスト", "ref_audio": "ref/speaker.wav"}
{"audio": "data/002.wav", "text": "2番目の発話テキスト", "ref_audio": "ref/speaker.wav"}
```

- `ref_audio` は全サンプルで同一を推奨（話者一貫性のため）
- `audio` は WAV 形式

### 学習スクリプトの技術詳細（`sft_12hz.py` 解析）

| 項目 | 内容 |
|------|------|
| 学習対象パラメータ | talker モデル + code predictor |
| 非学習 | speaker encoder（推論には使用するが重みは更新しない） |
| 話者注入方法 | `ref_mels` から speaker embedding を抽出し codec embedding の位置 6 に注入 |
| 損失関数 | `outputs.loss + 0.3 * sub_talker_loss` |
| 推奨ハイパーパラメータ | batch=2〜32, lr=2e-6〜2e-5, epochs=3〜10 |

---

## 2. 公式 Gradio デモ（`qwen-tts-demo`）

`pip install qwen-tts` でインストールされる組み込みコマンド。

```bash
# CustomVoice モデルで起動
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000

# HTTPS 対応（リモートアクセス時）
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --ip 0.0.0.0 --port 8000 \
  --ssl-certfile cert.pem --ssl-keyfile key.pem
```

**本プロジェクトとの違い**: 公式デモは 1 モデルずつ起動。本プロジェクトの WebUI (#33) は
API 経由で全 3 モデルを 1 画面に統合している。

---

## 3. DashScope API（Issue #40）

Alibaba Cloud が提供するマネージド API。ローカル GPU 不要・ストリーミング対応。

| エンドポイント | 内容 |
|---|---|
| CustomVoice Real-time API | 組み込み話者でのストリーミング合成 |
| Voice Clone Real-time API | 参照音声からのボイスクローニング |
| VoiceDesign Real-time API | instruct による声設計 |

→ 国内（中国）・国際の 2 エンドポイント。API キーが必要（有料）。

---

## 4. vLLM-Omni 統合（Issue #39）

| 項目 | 内容 |
|------|------|
| 現状 | オフライン推論のみサポート |
| 予定 | オンラインサービング（`vllm serve`）は近日対応予定 |
| 用途 | FastAPI サーバーのバックエンドとして統合しスループット改善 |

---

## 5. 未リリースモデル（Issue #41）

公式 README: 「テクニカルレポートに記載の他のモデルを近日公開予定」

- 内容は非公開
- 「参照音声 + instruct」を直接サポートするモデルが含まれる可能性
- 公開後に本プロジェクトへの統合を検討

---

## フェーズへの影響

| 発見事項 | フェーズ | Issue |
|---------|---------|-------|
| ファインチューニングで参照音声+instruct 実現 | v2.1 | #36（更新済み） |
| vLLM-Omni オフライン推論 | v2.1 | #39（新規） |
| DashScope API（クラウドバックエンド） | v2.1 | #40（新規） |
| 未リリースモデルの追跡 | 公開後 | #41（新規） |
