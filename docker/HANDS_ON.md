# tobira Docker ハンズオン

tobira の MTA プラグイン（Haraka / rspamd / SpamAssassin）が
API サーバーと連携して動作する様子を、手元の Docker 環境で体験するガイドです。

## 前提条件

- Docker Engine 20.10+
- Docker Compose v2+

```bash
# バージョン確認
docker --version
docker compose version
```

## 2 つのモード

| モード | API サーバー | ML モデル | 起動時間 | 用途 |
|---|---|---|---|---|
| **mock** (デフォルト) | ルールベース分類器 | 不要 | 数秒 | プラグイン連携の動作確認・CI |
| **real** | 本物の tobira + Ollama | gemma2:2b (~2GB) | 初回数分 | ML推論の体験・本番相当テスト |

## アーキテクチャ

### mock モード

```
┌──────────┐  SMTP   ┌──────────┐  HTTP POST   ┌─────────────┐
│  あなた   │───────→│  Haraka   │────────────→│  tobira-api │
└──────────┘ :2525   └──────────┘ /v1/predict  │  (mock)     │
                                                │  :8000      │
┌──────────┐  HTTP   ┌──────────┐  HTTP POST   │  キーワード  │
│  あなた   │───────→│  rspamd  │────────────→│  マッチで    │
└──────────┘ :11333  └──────────┘ /v1/predict  │  分類       │
                                                │             │
┌──────────┐  spamc  ┌──────────────┐ HTTP POST│             │
│  あなた   │───────→│ SpamAssassin │─────────→│             │
└──────────┘ :783    └──────────────┘/v1/predict└─────────────┘
```

### real モード

```
┌──────────┐  SMTP   ┌──────────┐              ┌─────────────┐     ┌──────────┐
│  あなた   │───────→│  Haraka   │─────────────│  tobira-api │────→│  Ollama  │
└──────────┘ :2525   └──────────┘  /v1/predict │  (real)     │     │  :11434  │
                                                │  :8000      │     │ gemma2:2b│
┌──────────┐  HTTP   ┌──────────┐              │             │     └──────────┘
│  あなた   │───────→│  rspamd  │─────────────│  ML推論で   │
└──────────┘ :11333  └──────────┘  /v1/predict │  分類       │
                                                │             │
┌──────────┐  spamc  ┌──────────────┐          │             │
│  あなた   │───────→│ SpamAssassin │──────────│             │
└──────────┘ :783    └──────────────┘/v1/predict└─────────────┘
```

---

## Step 1: サービスの起動

### mock モードで起動（推奨：まずこちらから）

```bash
cd docker
docker compose up --build -d
```

### real モードで起動（ML モデルを使う場合）

```bash
cd docker
docker compose -f docker-compose.real.yml up --build -d
```

> **初回のみ**: Ollama の gemma2:2b モデル (~2GB) のダウンロードに数分かかります。
> `ollama-init` サービスがモデルをプルし、完了後に tobira-api が起動します。

起動状態を確認:

```bash
# mock モード
docker compose ps

# real モード
docker compose -f docker-compose.real.yml ps
```

すべてのサービスが `running` (tobira-api は `healthy`) になっていれば OK です。

> **Note**: 以降のステップでは mock モードの `docker compose` コマンドで記載しています。
> real モードの場合は、すべてのコマンドで `-f docker-compose.real.yml` を追加してください。

---

## Step 2: tobira API を直接テスト

まず API サーバー単体の動作を確認します。

### ヘルスチェック

```bash
curl -s http://localhost:8000/v1/health | python3 -m json.tool
```

期待される出力:

```json
{
    "status": "ok"
}
```

### ham（正常メール）を判定

```bash
curl -s -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hi, can we schedule our weekly meeting for Thursday?"}' \
  | python3 -m json.tool
```

**mock モードの場合**（スパムキーワードなし → ham 判定）:

```json
{
    "label": "ham",
    "score": 0.91,
    "labels": {
        "spam": 0.09,
        "ham": 0.91
    }
}
```

**real モードの場合**（LLM が文脈を判断）:

```json
{
    "label": "ham",
    "score": 0.95,
    "labels": {
        "spam": 0.05,
        "ham": 0.95
    }
}
```

> real モードではスコアが LLM の推論結果に基づくため、毎回微妙に異なります。

### spam を判定

```bash
curl -s -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Buy now! Free offer! Click here for your lottery winner prize!"}' \
  | python3 -m json.tool
```

**mock モードの場合**（スパムキーワード 4 個 → spam 判定、score=0.95）:

```json
{
    "label": "spam",
    "score": 0.95,
    "labels": {
        "spam": 0.95,
        "ham": 0.05
    }
}
```

> **mock モードのポイント**: キーワードマッチで分類します。
> `buy now`, `free offer`, `click here`, `lottery`, `winner` 等のキーワードが
> 3 個以上あると score=0.95、2 個で 0.80、1 個で 0.60 になります。

> **real モードのポイント**: LLM がメール文面の意味を理解して分類します。
> キーワードがなくても、詐欺的な文面であれば spam 判定されます。

---

## Step 3: Haraka（SMTP）経由でテスト

Haraka は SMTP サーバーとして動作し、`data_post` フックで tobira API を呼び出します。

### ham メールを送信

```bash
python3 -c "
import smtplib
from email.message import EmailMessage

msg = EmailMessage()
msg['From'] = 'alice@example.com'
msg['To'] = 'bob@example.com'
msg['Subject'] = 'Meeting tomorrow'
msg.set_content('Hi Bob, can we meet at 3pm tomorrow to discuss the project?')

try:
    with smtplib.SMTP('localhost', 2525, timeout=10) as s:
        s.send_message(msg)
    print('Result: accepted')
except smtplib.SMTPRecipientsRefused:
    print('Result: processed (rejected by rcpt_to, not by tobira)')
except smtplib.SMTPDataError as e:
    print(f'Result: rejected by tobira - {e}')
"
```

### spam メールを送信

```bash
python3 -c "
import smtplib
from email.message import EmailMessage

msg = EmailMessage()
msg['From'] = 'spammer@example.com'
msg['To'] = 'victim@example.com'
msg['Subject'] = 'You are a winner!'
msg.set_content('Buy now! Free offer! Click here for your lottery winner prize! Act now!')

try:
    with smtplib.SMTP('localhost', 2525, timeout=10) as s:
        s.send_message(msg)
    print('Result: accepted (spam was not blocked)')
except smtplib.SMTPRecipientsRefused:
    print('Result: processed (rejected by rcpt_to)')
except smtplib.SMTPDataError as e:
    print(f'Result: REJECTED by tobira plugin! - {e}')
"
```

> **注意**: テスト環境では `rcpt_to` プラグインが未設定のため、
> 宛先不明で reject されることがあります。これは tobira プラグインの
> reject ではありません。Haraka のログで tobira の動作を確認してください。

### Haraka ログを確認

```bash
docker compose logs haraka --tail 20
```

`[plugins] loading tobira` や SMTP トランザクションのログが見えます。

---

## Step 4: rspamd 経由でテスト

rspamd は HTTP API (`/checkv2`) でメールをスキャンし、
tobira プラグインが `TOBIRA_*` シンボルを挿入します。

### ham メールをスキャン

```bash
curl -s -X POST http://localhost:11333/checkv2 --data-binary \
'From: alice@example.com
To: bob@example.com
Subject: Normal email
Content-Type: text/plain; charset=utf-8

Hi, this is a perfectly normal business email about our quarterly review.' \
  | python3 -m json.tool
```

レスポンスの `symbols` セクションに `TOBIRA_HAM` が含まれていれば成功です:

```json
{
    "TOBIRA_HAM": {
        "name": "TOBIRA_HAM",
        "score": -3.0,
        "options": ["score=0.1000"]
    }
}
```

### spam メールをスキャン

```bash
curl -s -X POST http://localhost:11333/checkv2 --data-binary \
'From: spammer@example.com
To: victim@example.com
Subject: Winner!
Content-Type: text/plain; charset=utf-8

Buy now! Free offer! Click here for your lottery winner prize! Act now! Urgent discount casino!' \
  | python3 -m json.tool
```

`TOBIRA_SPAM_HIGH` が含まれていれば成功です:

```json
{
    "TOBIRA_SPAM_HIGH": {
        "name": "TOBIRA_SPAM_HIGH",
        "score": 8.0,
        "options": ["score=0.9500"]
    }
}
```

### rspamd ログを確認

```bash
docker compose logs rspamd | grep tobira
```

`tobira: label=ham score=...` や `tobira: label=spam score=...` のログが確認できます。

### シンボルとスコアの対応

| シンボル | weight | 条件 |
|---|---|---|
| `TOBIRA_SPAM_HIGH` | +8.0 | spam_score >= 0.9 |
| `TOBIRA_SPAM_MED` | +5.0 | spam_score >= 0.7 |
| `TOBIRA_SPAM_LOW` | +2.0 | spam_score >= 0.5 |
| `TOBIRA_HAM` | -3.0 | spam_score < 0.3 |
| `TOBIRA_FAIL` | 0.0 | API エラー時（fail-open） |

---

## Step 5: SpamAssassin 経由でテスト

SpamAssassin はコンテナ内の `spamc` コマンドでテストします。

### ham メールをスキャン

```bash
echo 'Subject: Meeting tomorrow
From: alice@example.com

Hi Bob, can we meet at 3pm tomorrow?' \
  | docker compose exec -T spamassassin spamc -R
```

出力にルール一覧が表示されます。`TOBIRA_CHECK` が含まれていれば
tobira プラグインが動作しています。

### spam メールをスキャン

```bash
echo 'Subject: You are a winner!
From: spammer@example.com

Buy now! Free offer! Click here for your lottery winner prize! Act now! Urgent discount casino!' \
  | docker compose exec -T spamassassin spamc -R
```

出力に `TOBIRA_SPAM_HIGH` (8.0 点) が含まれていれば成功です。

### SpamAssassin ログを確認

```bash
docker compose logs spamassassin | grep "tobira:"
```

`tobira: label=spam score=0.95` のようなログが確認できます。

---

## Step 6: ログをまとめて確認

### 全サービスのログをリアルタイム表示

```bash
docker compose logs -f
```

別のターミナルからメールを送信すると、各サービスのログがリアルタイムで流れます。
`Ctrl+C` で停止。

### tobira-api に届いたリクエストを確認

```bash
docker compose logs tobira-api | grep "/v1/predict"
```

各プラグインからの HTTP リクエストが `POST /v1/predict HTTP/1.1 200 OK` として記録されています。

---

## Step 7: 自動テストを実行

すべてのサービスの動作を一括検証するテストスクリプトがあります:

```bash
./test.sh
```

すべて PASS になれば、全プラグインの連携が正常に動作しています。

> **Note**: `test.sh` は mock / real どちらのモードでも動作します。
> real モードでは LLM の推論結果に基づくため、スコアの値は異なります。

---

## クリーンアップ

```bash
# mock モード
docker compose down

# real モード
docker compose -f docker-compose.real.yml down
```

ビルドキャッシュも含めて完全に削除する場合:

```bash
# mock モード
docker compose down --rmi local --volumes

# real モード（Ollama のモデルキャッシュも削除）
docker compose -f docker-compose.real.yml down --rmi local --volumes
```

---

## トラブルシューティング

### サービスが起動しない

```bash
# 各サービスのログを確認
docker compose logs tobira-api
docker compose logs haraka
docker compose logs rspamd
docker compose logs spamassassin

# real モードの場合、Ollama のログも確認
docker compose -f docker-compose.real.yml logs ollama
docker compose -f docker-compose.real.yml logs ollama-init
```

### real モードで tobira-api が起動しない

Ollama のモデルダウンロードが完了しているか確認してください:

```bash
docker compose -f docker-compose.real.yml logs ollama-init
```

`success` と表示されていれば OK です。タイムアウトした場合は再実行:

```bash
docker compose -f docker-compose.real.yml restart ollama-init
```

### real モードで predict が遅い

初回リクエストは LLM のウォームアップのため数秒かかります。
2 回目以降は高速化されます。GPU が利用できる環境では Ollama が
自動的に GPU を使用します。

### rspamd の TOBIRA シンボルが出ない

rspamd は起動直後にプラグインのロードに数秒かかることがあります。
10 秒ほど待ってから再度スキャンしてください。

### ポートが競合する

他のサービスが 8000, 2525, 11333, 11334, 783 を使用している場合、
compose ファイルの `ports` を変更してください:

```yaml
ports:
  - "18000:8000"  # 例: ホスト側を 18000 に変更
```
