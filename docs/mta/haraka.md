# Haraka Integration

This guide covers integrating tobira with Haraka using the Node.js plugin.

## Prerequisites

- Haraka mail server
- Node.js >= 18
- tobira API server running and accessible

## Installation

1. Install the tobira npm package in your Haraka installation:

    ```bash
    cd /path/to/haraka
    npm install tobira
    ```

2. Copy the plugin file:

    ```bash
    cp node_modules/tobira/../integrations/haraka/plugins/tobira.js plugins/
    ```

    Or copy from the tobira repository:

    ```bash
    cp integrations/haraka/plugins/tobira.js /path/to/haraka/plugins/
    ```

3. Create the configuration file `config/tobira.ini`:

    ```ini
    [main]
    url = http://127.0.0.1:8000
    timeout = 5000
    threshold = 0.5
    reject_spam = true
    send_headers = false
    ```

4. Enable the plugin by adding `tobira` to `config/plugins`:

    ```
    tobira
    ```

5. Restart Haraka:

    ```bash
    systemctl restart haraka
    ```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `url` | `http://127.0.0.1:8000` | tobira API server base URL |
| `timeout` | `5000` | HTTP request timeout in milliseconds |
| `threshold` | `0.5` | Spam score threshold for rejection |
| `reject_spam` | `true` | Reject emails above threshold |
| `send_headers` | `false` | Send email headers for analysis |

## How It Works

The plugin hooks into Haraka's `data_post` event:

1. Collects the email body text from the transaction
2. Optionally extracts email headers (SPF, DKIM, DMARC, From, Reply-To, Received, Content-Type)
3. Sends a POST request to the tobira `/predict` endpoint
4. Adds results to the transaction using `transaction.results.add()`
5. If `reject_spam` is enabled and the score exceeds the threshold, the message is rejected with a 550 response

## Transaction Results

The plugin adds results to `transaction.results` under the `tobira` namespace:

```javascript
{
  pass: "score=0.12",    // if classified as ham
  fail: "score=0.95",    // if classified as spam
  score: 0.95,
  label: "spam"
}
```

Other plugins can access these results:

```javascript
const tobiraResults = connection.transaction.results.get("tobira");
```

## Header Analysis

Set `send_headers = true` to include email authentication results in the classification request. The plugin extracts:

- SPF result from `Received-SPF` header
- DKIM result from `DKIM-Signature` header
- DMARC result from `Authentication-Results` header
- From, Reply-To, Received, and Content-Type headers

## Verification

Check the Haraka log for tobira plugin activity:

```bash
# Look for tobira results in the log
grep tobira /var/log/haraka/haraka.log
```

## Troubleshooting

**Plugin not loading**: Ensure `tobira` is listed in `config/plugins` and the `tobira` npm package is installed.

**Connection refused**: Verify the API server URL in `config/tobira.ini` and check that the server is running.

**Timeout errors**: Increase the `timeout` value in `config/tobira.ini`.
