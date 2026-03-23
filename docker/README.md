# Docker Integration Test Environment

各プラグイン（Haraka / Rspamd / SpamAssassin）の動作確認用 Docker 環境です。

## 構成

| サービス | 説明 | ポート |
|---|---|---|
| `tobira-api` | モック API サーバー（ルールベース分類） | 8000 |
| `haraka` | Haraka MTA + tobira プラグイン | 2525 (SMTP) |
| `rspamd` | Rspamd + tobira プラグイン | 11333, 11334 |
| `spamassassin` | SpamAssassin + tobira プラグイン | 783 |

## 使い方

### 全サービス起動

```bash
cd docker
docker compose up --build -d
```

### 個別サービス起動

```bash
# API + Haraka のみ
docker compose up --build -d tobira-api haraka

# API + Rspamd のみ
docker compose up --build -d tobira-api rspamd

# API + SpamAssassin のみ
docker compose up --build -d tobira-api spamassassin
```

### テスト実行

```bash
# 全サービス起動後
./test.sh
```

### 手動テスト

```bash
# API 直接テスト
curl -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Buy now! Free offer! Click here!"}'

# Haraka (SMTP)
python3 -c "
import smtplib
from email.message import EmailMessage
msg = EmailMessage()
msg['From'] = 'test@example.com'
msg['To'] = 'dest@example.com'
msg['Subject'] = 'Test'
msg.set_content('Hello world')
with smtplib.SMTP('localhost', 2525) as s:
    s.send_message(msg)
"

# Rspamd
curl -X POST http://localhost:11333/checkv2 \
  -d 'Buy now! Free offer! Click here!'

# SpamAssassin (spamc が必要)
echo "Subject: Test\n\nBuy now! Free offer!" | spamc -d localhost -p 783
```

### 停止

```bash
docker compose down
```

## モック API サーバー

テスト用にルールベースの分類器を使用します（ML モデル不要）。

- スパムキーワード 3 個以上 → `spam` (score=0.95)
- スパムキーワード 2 個 → `spam` (score=0.80)
- スパムキーワード 1 個 → `spam` (score=0.60)
- キーワードなし → `ham` (score=0.81〜1.00)

キーワード: `buy now`, `free offer`, `click here`, `viagra`, `lottery`, `winner`, `urgent`, `act now`, `limited time`, `casino`, `cheap`, `discount`
