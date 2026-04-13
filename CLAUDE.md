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

### 基本方針

- **Docker Desktop は使用しない**（商用ライセンス問題のため）。Linux では Docker Engine を直接使用。
- `docker compose`（スペースあり）を使用する（旧 `docker-compose` は使用しない）。
- `sudo` なしで `docker` を実行できる設定（`docker` グループへの追加）を前提とする。

### Compose ファイル構成

```
project/
  ├─ docker-compose.yml          # 共通設定（ベース）
  ├─ docker-compose.override.yml # 開発環境の上書き設定（自動適用）
  └─ docker-compose.prod.yml     # 本番環境の上書き設定
```

- 開発環境: `docker compose up -d`（override.yml が自動適用）
- 本番環境: `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`

### データ永続化

- **Named Volume を推奨**（コンテナとホストを疎結合に保つ）。
- Bind Mount はソースコード共有・ホットリロード目的に限定。
- `docker compose down -v` はボリューム削除を伴うため、DB データ消失に注意。

### 機密情報

- パスワード等は `.env` ファイルに定義し、`docker-compose.yml` から参照する。
- `.env` は `.gitignore` に追加してリポジトリに含めない。

### キャッシュ管理

- 定期的に `docker system df` で使用量を確認する。
- 不要リソースの削除: `docker system prune`（ボリュームも含める場合は `--volumes` を付与、要注意）。

### このプロジェクト固有のルール

- GPU パススルー: `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=all`（NVIDIA Container Toolkit 前提）。
- モデルキャッシュ（~3.4GB）は named volume (`hf_cache`, `torch_cache`) で永続化する。
- 生成音声 (`output/`) は bind mount でホストから即座に確認できるようにする。

---

## Python Conventions

### コードスタイル

- インデント: スペース4（タブ禁止）
- 行の長さ: 120文字以内（black / ruff で統一）
- 文字コード: UTF-8
- import 順序: 標準ライブラリ → サードパーティ → アプリ内部
- ツール: `ruff`（Lint）、`black`（フォーマット）、`mypy`（型チェック）

### 命名規則

| 対象 | 規則 | 例 |
|------|------|----|
| 変数・関数 | snake_case | `fetch_user`, `user_name` |
| クラス | PascalCase | `UserService` |
| 定数 | UPPER_SNAKE_CASE | `DEFAULT_TIMEOUT` |
| ファイル名 | snake_case | `test_basic_tts.py` |

### 型ヒント

- すべての関数に引数・返却値の型を付与する。
- 戻り値がない場合は `-> None` を明記する。
- Union は `|` 記法（Python 3.10+）: `str | None`

### ファイルヘッダー（モジュール Docstring）

`src/` 配下のアプリコードには以下のヘッダーを `import` より前に記載する:

```python
"""
<ファイルの概要を1行で>

- 目的: <このファイルの責務>
- 対象: <主な利用者>
- 関連: <設計書ID / チケットID など>

作成者: 宗廣 颯真
作成日: YYYY-MM-DD
最終更新者: 宗廣 颯真
最終更新日: YYYY-MM-DD
"""
```

### Docstring（Google スタイル）

- 1行目: 何をするか簡潔に記述。
- セクション順: `Args:` → `Returns:` → `Raises:` → `Examples:`
- 型は型ヒントに一本化し、Docstring に重複記載しない。
- `dict[str, Any]` 等で抽象的になる場合は期待するキー・構造を説明する。

```python
def fetch_user(user_id: int) -> User:
    """ユーザーIDを指定してユーザー情報を取得する。

    Args:
        user_id: 対象ユーザーID。正の整数であること。

    Returns:
        ユーザーエンティティ。

    Raises:
        UserNotFound: 指定したIDのユーザーが存在しない場合。
    """
```

### コメント

- コメントは "Why（なぜそうするか）" を書く（"What" はコードから読み取れるため不要）。
- 処理段階が複数ある関数では、各ブロック前に処理単位コメントを入れる。
- `""" """` は Docstring 専用。コメント目的での使用禁止。

### エラーハンドリング

- 空の `except:` 禁止。`except Exception:` の乱用禁止。
- ログを書かずに例外を無視する行為禁止。
- 想定する例外型を捕捉し、コンテキストをログに残してから再 raise する。

### ログ

- `print` デバッグ禁止。`logging` または `structlog` を使用。
- `user_id` / `trace_id` を含める。JSON 形式を推奨。

### セキュリティ

- `eval` / `exec` 禁止。
- Secrets（APIキー・パスワード等）の直書き禁止。
- `requests` は `timeout` を必須設定。

### 禁止事項

- 未使用変数・未使用 import
- マジックナンバー
- 例外の握りつぶし
- 過度な if-else ネスト（3段以内）
- グローバル変数の乱用
- 過度な抽象化・不要なクラス化

### このプロジェクトでの環境構築

- Docker コンテナ内で pip を使用（ローカルに venv は作らない）。
- 依存関係は `requirements.txt` にピン留め（`package==version` 形式）。
- pyenv / poetry / uv はコンテナ内では使用しない。

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

## GitHub Issues 構造

リポジトリ: https://github.com/sora-kisaragi/qwen-tts-validation

### ラベル体系

| ラベル | 用途 |
|--------|------|
| `epic` | フェーズ全体を管理する親チケット |
| `phase-1` 〜 `phase-4` | 各フェーズへの紐付け |
| `priority:high/medium/low` | 優先度 |
| `bug` / `documentation` | 種別 |

### フェーズ構成

| Epic | Issue | 内容 |
|------|-------|------|
| [Epic] Phase 1: Docker 環境の安定化 | #15 | **現在対応中** |
| └ torchaudio CUDA 非互換問題の解決 | #1 | ブロッカー: `libcudart.so.13` not found |
| └ TRANSFORMERS_CACHE 非推奨警告対応 | #2 | 低優先度 |
| └ GB10 GPU 警告の調査 | #3 | 中優先度 |
| [Epic] Phase 2: 基本TTS・日本語対応検証 | #16 | Phase 1 完了後 |
| └ TC-01: 英語短文 TTS | #4 | |
| └ TC-02: 英語段落 TTS | #5 | |
| └ TC-03: 日本語ひらがな/カタカナ | #6 | |
| └ TC-04: 日本語漢字混じり | #7 | |
| └ TC-05: 日英混在テキスト | #8 | |
| └ TC-07: GPU 利用確認 | #9 | |
| [Epic] Phase 3: ボイスクローニング検証 | #17 | Phase 2 完了後 |
| └ 参照音声の準備 | #10 | |
| └ TC-06: ボイスクローニング実行 | #11 | |
| [Epic] Phase 4: 検証結果ドキュメント化 | #18 | Phase 3 完了後 |
| └ validation-plan.md に結果記入 | #12 | |
| └ README.md トラブルシューティング更新 | #13 | |
| └ 検証完了レポート作成 | #14 | |

### 運用ルール

- Phase は順次進行（前フェーズの Epic 完了が次フェーズの開始条件）
- 作業開始時に issue を `In Progress` に、完了時に Close する
- トラブル・発見事項は該当 issue のコメントに記録する

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
