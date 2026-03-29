# tobira

**JavaScript client for the tobira spam detection API.**

A lightweight Node.js SDK for integrating with tobira's ML-powered email spam classification service.

[![npm version](https://img.shields.io/npm/v/tobira)](https://www.npmjs.com/package/tobira)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js 18+](https://img.shields.io/badge/node-18%2B-green.svg)](https://nodejs.org/)

**[Documentation](https://velocitylabo.github.io/tobira)** | **[GitHub](https://github.com/velocitylabo/tobira)** | **[Python Package](https://pypi.org/project/tobira/)**

## Install

```bash
npm install tobira
```

## Quick Start

```javascript
const { TobiraClient } = require("tobira");

const client = new TobiraClient("http://127.0.0.1:8000");

const result = await client.predict("Congratulations! You've won a free iPhone...");
console.log(result.label); // "spam"
console.log(result.score); // 0.98
```

## API

### `new TobiraClient(baseUrl, options?)`

Create a client instance.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `timeout` | `number` | `5000` | Request timeout in ms |
| `apiKey` | `string` | — | Bearer token for authentication |

```javascript
const client = new TobiraClient("http://127.0.0.1:8000", {
  timeout: 3000,
  apiKey: "your-token",
});
```

### `client.predict(text, options?)`

Classify text as spam or ham.

```javascript
const result = await client.predict("Check out this deal!", {
  headers: {
    spf: "pass",
    dkim: "pass",
    dmarc: "pass",
    from: "sender@example.com",
  },
  explain: true,
});

// result:
// {
//   label: "spam",
//   score: 0.95,
//   labels: { spam: 0.95, ham: 0.05 },
//   header_score: 0.1,
//   explanations: { ... }
// }
```

### `client.feedback(text, label, source?)`

Submit a correction to improve future predictions.

```javascript
await client.feedback(
  "This was actually legitimate",
  "ham",       // "spam" | "ham"
  "haraka-01"  // optional: identifier for your MTA instance
);
```

### `client.health()`

Check the API server status.

```javascript
const status = await client.health();
// { status: "ok" }
```

## Use with Haraka

tobira includes a ready-to-use Haraka plugin. See the [Haraka integration guide](https://velocitylabo.github.io/tobira/mta/haraka/) for setup instructions.

## Requirements

- Node.js >= 18.0.0
- A running tobira API server (see [tobira on PyPI](https://pypi.org/project/tobira/))

## Documentation

Full documentation is available at **[velocitylabo.github.io/tobira](https://velocitylabo.github.io/tobira)**, including server setup, MTA integration tutorials, and API reference.

## License

MIT
