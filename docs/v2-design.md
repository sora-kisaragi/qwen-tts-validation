# v2.0.0 設計ドキュメント: API / WebUI / モデル対応マトリクス

作成日: 2026-04-14

---

## 対応する組み合わせマトリクス

qwen-tts 0.1.1 で実行可能な全組み合わせを以下に示す。

| # | 声の元 | instruct | モデル | メソッド | v2.0 対応 |
|---|--------|----------|--------|---------|:---------:|
| 1 | 参照音声（自分の声） | なし | Base | `generate_voice_clone(ref_audio)` | ✅ |
| 2 | 参照音声 → プロファイル (.pt) | なし | Base | `generate_voice_clone(voice_clone_prompt)` | ✅ |
| 3 | 組み込み話者（9種） | なし | CustomVoice | `generate_custom_voice(speaker)` | ✅ |
| 4 | 組み込み話者（9種） | あり | CustomVoice | `generate_custom_voice(speaker, instruct)` | ✅ |
| 5 | なし（声をテキストで設計） | あり | VoiceDesign | `generate_voice_design(instruct)` | ✅ |
| 6 | 参照音声 + instruct | あり | — | — | ❌ **現 API に存在しない** |

> **#6 について**: アーキテクチャ上は `model.generate()` に両パラメータを渡すことは可能だが、
> Base モデルは instruct で学習されておらず、VoiceDesign モデルは参照音声で学習されていない。
> 実用的な品質は得られないため、専用モデルの学習またはハック検証を v2.1 以降で扱う。
> → Issue #36 参照

---

## 利用可能モデル一覧

| モデル ID | tts_model_type | 話者 | 対応言語 |
|-----------|---------------|------|---------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `base` | 参照音声で決まる | Auto のみ推奨 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `voice_design` | instruct で設計 | auto / ja / en / zh 等 11言語 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `custom_voice` | 9種固定 | auto / ja / en / zh 等 11言語 |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `base` | 参照音声で決まる | 軽量版（未検証） |

### CustomVoice 組み込み話者

`aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu`, `vivian`

### 対応言語（CustomVoice / VoiceDesign）

`auto`, `chinese`, `english`, `french`, `german`, `italian`, `japanese`, `korean`,
`portuguese`, `russian`, `spanish`

---

## REST API 設計（FastAPI）

### エンドポイント一覧

| メソッド | パス | 対応組み合わせ | モデル |
|---------|------|-------------|--------|
| `POST` | `/tts/voice-clone` | #1: 参照音声アップロード | Base |
| `POST` | `/tts/voice-clone/profile` | #2: 保存済みプロファイル使用 | Base |
| `POST` | `/tts/custom-voice` | #3, #4: 組み込み話者 | CustomVoice |
| `POST` | `/tts/voice-design` | #5: instruct で声設計 | VoiceDesign |
| `GET` | `/tts/speakers` | 利用可能話者一覧 | CustomVoice |
| `GET` | `/tts/languages` | 対応言語一覧 | — |
| `GET` | `/health` | ヘルスチェック | — |

### リクエスト/レスポンス概要

```
POST /tts/voice-clone
  Request : multipart/form-data
    ref_audio : WAV file (24kHz mono, 3-10s)
    ref_text  : string (optional, ICL モード用)
    text      : string (合成テキスト)
    language  : string (optional, default="Auto")
  Response: audio/wav

POST /tts/voice-clone/profile
  Request : multipart/form-data
    profile_name : string (speaker_profiles/ 以下のファイル名)
    text         : string
    language     : string (optional)
  Response: audio/wav

POST /tts/custom-voice
  Request : application/json
    speaker  : string (aiden / ono_anna 等)
    text     : string
    language : string (optional, default="Auto")
    instruct : string (optional)
  Response: audio/wav

POST /tts/voice-design
  Request : application/json
    text     : string
    instruct : string (必須)
    language : string (optional, default="Auto")
  Response: audio/wav
```

### 実装方針

- 起動時に 3 モデルをすべてメモリへロード（GB10 の 121 GB 統合メモリで十分）
- レスポンスは `audio/wav` バイト列をそのまま返す（ファイル保存はしない）
- `speaker_profiles/` はコンテナ外のバインドマウントを参照
- `scripts/model_utils.py` の `ensure_model_downloaded` / `load_speaker_profile` を再利用

---

## WebUI 設計（Gradio）

### タブ構成

```
┌─────────────────────────────────────────────────────┐
│  Qwen3-TTS WebUI                                    │
├──────────┬──────────────┬──────────────┬────────────┤
│ Voice    │ Custom Voice │ Voice Design │            │
│ Clone    │              │              │            │
└──────────┴──────────────┴──────────────┴────────────┘
```

#### Tab 1: Voice Clone（組み合わせ #1, #2）

```
[参照音声] ファイルアップロード or プロファイル名入力
[参照テキスト] テキストボックス（空 = x-vector モード）
[言語] ドロップダウン（Auto / ja / en / zh ...）
[合成テキスト] テキストエリア
[生成] ボタン
→ [再生] + [ダウンロード]
```

#### Tab 2: Custom Voice（組み合わせ #3, #4）

```
[話者] ドロップダウン（aiden / dylan / eric / ono_anna / ryan / serena / sohee / uncle_fu / vivian）
[言語] ドロップダウン
[instruct] テキストボックス（任意）
[合成テキスト] テキストエリア
[生成] ボタン
→ [再生] + [ダウンロード]
```

#### Tab 3: Voice Design（組み合わせ #5）

```
[instruct] テキストエリア（例: "Speak in a calm, professional male voice"）
[言語] ドロップダウン
[合成テキスト] テキストエリア
[生成] ボタン
→ [再生] + [ダウンロード]
```

---

## ファイル構成（追加分）

```
qwen-tts-validation/
├── api/
│   ├── main.py          # FastAPI アプリ本体
│   ├── models.py        # モデルロード・管理（起動時に全モデルを初期化）
│   ├── routes/
│   │   ├── voice_clone.py
│   │   ├── custom_voice.py
│   │   └── voice_design.py
│   └── schemas.py       # Pydantic リクエスト/レスポンス定義
├── webui/
│   └── app.py           # Gradio アプリ（API を呼び出す、または直接モデルを使う）
├── docker-compose.yml   # 既存（api / webui サービスを追加）
└── requirements.txt     # fastapi / uvicorn / gradio を追加
```

---

## フェーズ計画

### v2.0.0（現フェーズ）

- [ ] Issue #30: VoiceDesign / CustomVoice モデル調査・検証
- [ ] Issue #31: `language="ja"` パラメータ効果検証
- [ ] Issue #32: FastAPI 実装（上記 4 エンドポイント）
- [ ] Issue #33: Gradio WebUI 実装（上記 3 タブ）
- [ ] Issue #34: instruct 制御の統合（CustomVoice + VoiceDesign）

### v2.1.0（次フェーズ候補）

- [ ] Issue #36: 「参照音声 + instruct」の実験的検証
  - 低レベル API ハックによる動作確認
  - 品質評価（使用に耐えるか）
  - 将来的なファインチューニング方針の整理

---

## 既知の制約

| 制約 | 内容 | フェーズ |
|------|------|---------|
| 参照音声 + instruct 不可 | いずれのモデルもこの組み合わせで学習されていない | v2.1 で実験的に検証 |
| 日本語漢字の中国語読み混入 | Base モデルで CJK 漢字が中国語発音になる場合がある（Issue #26） | language="ja" 効果を v2.0 で検証 |
| triton (ARM64) 不可 | Whisper は CPU で動作（許容範囲） | — |
