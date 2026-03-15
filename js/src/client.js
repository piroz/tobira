"use strict";

/**
 * TobiraClient — lightweight HTTP client for the tobira predict API.
 *
 * @example
 *   const { TobiraClient } = require("tobira");
 *   const client = new TobiraClient("http://127.0.0.1:8000");
 *   const result = await client.predict("some email text");
 *   console.log(result.label, result.score);
 */
class TobiraClient {
  /**
   * @param {string} baseUrl  Base URL of the tobira API (e.g. "http://127.0.0.1:8000").
   * @param {object} [options]
   * @param {number} [options.timeout=5000]  Request timeout in milliseconds.
   */
  constructor(baseUrl, options = {}) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.timeout = options.timeout ?? 5000;
  }

  /**
   * Call POST /predict.
   *
   * @param {string} text  The text to classify.
   * @returns {Promise<{label: string, score: number, labels: Object<string, number>}>}
   */
  async predict(text) {
    const url = `${this.baseUrl}/predict`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });

      if (!res.ok) {
        throw new Error(`tobira API error: ${res.status} ${res.statusText}`);
      }

      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Call POST /feedback.
   *
   * @param {string} text   The email text to report.
   * @param {string} label  "spam" or "ham".
   * @param {string} [source="unknown"]  Reporting MTA identifier.
   * @returns {Promise<{status: string, id: string}>}
   */
  async feedback(text, label, source = "unknown") {
    const url = `${this.baseUrl}/feedback`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, label, source }),
        signal: controller.signal,
      });

      if (!res.ok) {
        throw new Error(`tobira API error: ${res.status} ${res.statusText}`);
      }

      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Call GET /health.
   *
   * @returns {Promise<{status: string}>}
   */
  async health() {
    const url = `${this.baseUrl}/health`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const res = await fetch(url, {
        signal: controller.signal,
      });

      if (!res.ok) {
        throw new Error(`tobira API error: ${res.status} ${res.statusText}`);
      }

      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }
}

module.exports = { TobiraClient };
