# Qwen3-TTS API 仕様書

Qwen3-TTS FastAPI サーバーのエンドポイント仕様。
他プロジェクトからの呼び出しや、クライアント実装の参照用途として使用する。

- **Base URL**: `http://<host>:7865`
- **レスポンス形式**: 音声エンドポイントは `audio/wav`、情報エンドポイントは `application/json`
- **認証**: なし
- **Swagger UI**: `http://<host>:7865/docs`
- **OpenAPI JSON**: `http://<host>:7865/openapi.json`

---

## エンドポイント一覧

| メソッド | パス | 説明 | 使用モデル |
|---------|------|------|-----------|
| GET | `/health` | ヘルスチェック | — |
| GET | `/tts/speakers` | 利用可能な話者一覧 | — |
| GET | `/tts/languages` | 対応言語一覧 | — |
| POST | `/tts/voice-clone` | 参照音声から音声合成 | Base |
| POST | `/tts/voice-clone/profile` | 保存済みプロファイルから音声合成 | Base |
| GET | `/tts/voice-clone/profiles` | 保存済みプロファイル一覧 | — |
| POST | `/tts/voice-clone/profiles` | 話者プロファイル作成 | Base |
| DELETE | `/tts/voice-clone/profiles/{profile_name}` | 話者プロファイル削除 | — |
| POST | `/tts/custom-voice` | 組み込み話者で音声合成 | CustomVoice |
| POST | `/tts/voice-design` | instruct で声を設計して音声合成 | VoiceDesign |

---

## 情報エンドポイント

### GET `/health`

サーバーの稼働状態を返す。

**レスポンス**

```json
{"status": "ok"}
```

---

### GET `/tts/speakers`

CustomVoice モデルが対応する話者名の一覧を返す。

**レスポンス**

```json
{
  "speakers": ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
}
```

---

### GET `/tts/languages`

全モデルが対応する言語名の一覧を返す。

**レスポンス**

```json
{
  "languages": ["auto", "chinese", "english", "french", "german", "italian", "japanese", "korean", "portuguese", "russian", "spanish"]
}
```

> **注意**: `language` パラメータには ISO コード (`"ja"`) ではなく完全名 (`"japanese"`) を使うこと。

---

## 音声合成エンドポイント

全エンドポイントの成功レスポンスは `Content-Type: audio/wav` の WAV バイナリ。

### POST `/tts/voice-clone` — 参照音声アップロード（組み合わせ #1）

アップロードした参照音声から話者の声を抽出し、指定テキストを合成する。

- `ref_text` **あり** → ICL モード。参照音声の音声コードも利用するため声質再現度が高い。**推奨**。
- `ref_text` **なし** → x-vector モード。話者埋め込みのみ使用。

**リクエスト**: `multipart/form-data`

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| `text` | string | ✅ | 合成するテキスト |
| `ref_audio` | file (WAV) | ✅ | 参照音声ファイル。24kHz mono, 3〜10秒推奨 |
| `ref_text` | string | — | 参照音声の書き起こし。省略時は x-vector モード |
| `language` | string | — | 言語。デフォルト `"auto"` |

**curl 例**

```bash
# ICL モード（ref_text あり・推奨）
curl -X POST http://localhost:7865/tts/voice-clone \
  -F "text=こんにちは、これはテストです。" \
  -F "ref_audio=@sample_audio/reference.wav" \
  -F "ref_text=参照音声で話している内容" \
  -F "language=japanese" \
  --output output/result.wav

# x-vector モード（ref_text なし）
curl -X POST http://localhost:7865/tts/voice-clone \
  -F "text=Hello, this is a test." \
  -F "ref_audio=@sample_audio/reference.wav" \
  --output output/result.wav
```

---

### POST `/tts/voice-clone/profile` — 保存済みプロファイル（組み合わせ #2）

`create_speaker_profile.py` で事前生成した話者プロファイル (`.pt`) を使って合成する。
参照音声の再処理が不要なため、同一話者を繰り返し使う場合に高速。

**リクエスト**: `multipart/form-data`

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| `text` | string | ✅ | 合成するテキスト |
| `profile_name` | string | ✅ | `speaker_profiles/` 以下のファイル名。例: `default.pt` |
| `language` | string | — | 言語。デフォルト `"auto"` |

**curl 例**

```bash
curl -X POST http://localhost:7865/tts/voice-clone/profile \
  -F "text=おはようございます。" \
  -F "profile_name=default.pt" \
  -F "language=japanese" \
  --output output/result.wav
```

> **ヒント**: WebUI の「プロファイル管理」タブを使うと CLI なしでプロファイルを作成・管理できます。

---

### GET `/tts/voice-clone/profiles` — プロファイル一覧

`speaker_profiles/` 内の `.pt` ファイルを一覧する。

**レスポンス**

```json
{
  "profiles": [
    {"name": "default.pt", "created_at": "2026-04-15T10:00:00Z"},
    {"name": "my_voice.pt", "created_at": "2026-04-15T12:30:00Z"}
  ]
}
```

---

### POST `/tts/voice-clone/profiles` — 話者プロファイル作成

参照音声から話者プロファイル (`.pt`) を生成し `speaker_profiles/` に保存する。

- `ref_text` を省略すると **x-vector モード**（高速・簡易）
- `ref_text` を指定すると **ICL モード**（より高い声質再現度）
- `profile_name` に `.pt` 拡張子は不要（自動付与）

**リクエスト**: `multipart/form-data`

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| `ref_audio` | file (WAV) | ✅ | 参照音声ファイル。24kHz mono, 3〜10秒推奨 |
| `profile_name` | string | ✅ | 保存名。英数字・アンダースコア・ハイフンのみ（`.pt` は自動付与） |
| `ref_text` | string | — | 参照音声の書き起こし。省略時は x-vector モード |

**レスポンス** (201 Created)

```json
{"profile_name": "my_voice.pt", "message": "created"}
```

**curl 例**

```bash
# ICL モード（推奨）
curl -X POST http://localhost:7865/tts/voice-clone/profiles \
  -F "ref_audio=@sample_audio/reference.wav" \
  -F "profile_name=my_voice" \
  -F "ref_text=参照音声で話されている内容"

# x-vector モード（ref_text なし）
curl -X POST http://localhost:7865/tts/voice-clone/profiles \
  -F "ref_audio=@sample_audio/reference.wav" \
  -F "profile_name=my_voice"
```

---

### DELETE `/tts/voice-clone/profiles/{profile_name}` — 話者プロファイル削除

指定した `.pt` ファイルを `speaker_profiles/` から削除する。

**パスパラメータ**

| パラメータ | 説明 |
|-----------|------|
| `profile_name` | 削除するファイル名。例: `my_voice.pt` |

**レスポンス** (200 OK)

```json
{"profile_name": "my_voice.pt", "message": "deleted"}
```

**curl 例**

```bash
curl -X DELETE http://localhost:7865/tts/voice-clone/profiles/my_voice.pt
```

---

### POST `/tts/custom-voice` — 組み込み話者（組み合わせ #3/#4）

事前定義された話者を使って合成する。`instruct` でスタイルを調整できる。

- `instruct` **なし** → 話者デフォルトのスタイルで合成（組み合わせ #3）
- `instruct` **あり** → 声のスタイルを調整（組み合わせ #4）

**リクエスト**: `application/json`

| フィールド | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| `text` | string | ✅ | — | 合成するテキスト |
| `speaker` | string | ✅ | — | 話者名。`GET /tts/speakers` で取得できる値 |
| `language` | string | — | `"auto"` | 言語。`GET /tts/languages` で取得できる値 |
| `instruct` | string | — | `""` | 声のスタイル指定（任意）。例: `"Speak slowly and gently."` |

**curl 例**

```bash
# instruct なし（組み合わせ #3）
curl -X POST http://localhost:7865/tts/custom-voice \
  -H "Content-Type: application/json" \
  -d '{"text": "こんにちは。", "speaker": "ono_anna", "language": "japanese"}' \
  --output output/result.wav

# instruct あり（組み合わせ #4）
curl -X POST http://localhost:7865/tts/custom-voice \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to the presentation.",
    "speaker": "aiden",
    "language": "english",
    "instruct": "Speak in a calm, professional voice."
  }' \
  --output output/result.wav
```

**利用可能な話者（9名）**

| 話者名 | 特徴 |
|--------|------|
| `aiden` | 男性・英語向け |
| `dylan` | 男性・英語向け |
| `eric` | 男性・英語向け |
| `ono_anna` | 女性・日本語向け |
| `ryan` | 男性・英語向け |
| `serena` | 女性・英語向け |
| `sohee` | 女性・韓国語向け |
| `uncle_fu` | 男性・中国語向け |
| `vivian` | 女性・英語向け |

---

### POST `/tts/voice-design` — instruct で声設計（組み合わせ #5）

参照音声なしで、自然言語の `instruct` だけで声のスタイルを設計して合成する。

**リクエスト**: `application/json`

| フィールド | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| `text` | string | ✅ | — | 合成するテキスト |
| `instruct` | string | ✅ | — | 声の設計指示文。例: `"Speak in an energetic female voice with a slight Japanese accent."` |
| `language` | string | — | `"auto"` | 言語。`GET /tts/languages` で取得できる値 |

**curl 例**

```bash
curl -X POST http://localhost:7865/tts/voice-design \
  -H "Content-Type: application/json" \
  -d '{
    "text": "良い一日をお過ごしください。",
    "instruct": "Speak in a warm, friendly female voice with a calm tone.",
    "language": "japanese"
  }' \
  --output output/result.wav
```

---

## エラーレスポンス

| HTTP ステータス | 原因 |
|----------------|------|
| `400 Bad Request` | 不正なプロファイル名（禁止文字を含む） |
| `404 Not Found` | 指定したプロファイルファイルが存在しない |
| `422 Unprocessable Entity` | 不正なパラメータ（存在しない `speaker` / `language` など） |
| `500 Internal Server Error` | モデル推論中の予期しないエラー |

エラーレスポンスの形式:

```json
{"detail": "エラーの説明文"}
```

---

## Python クライアント例

```python
import requests

BASE_URL = "http://localhost:7865"

# 組み込み話者で合成
resp = requests.post(
    f"{BASE_URL}/tts/custom-voice",
    json={"text": "こんにちは。", "speaker": "ono_anna", "language": "japanese"},
    timeout=120,
)
resp.raise_for_status()

with open("output.wav", "wb") as f:
    f.write(resp.content)
```

---

## 関連ドキュメント

- [docs/v2-design.md](v2-design.md) — API 設計の背景と組み合わせマトリクス
- [docs/official-research.md](official-research.md) — 公式 GitHub 調査結果
- [docs/validation-plan.md](validation-plan.md) — テスト計画と検証結果
