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
   * Call POST /v1/predict.
   *
   * @param {string} text  The text to classify.
   * @param {object} [options]
   * @param {object} [options.headers]  Optional email headers for header-based classification.
   * @returns {Promise<{label: string, score: number, labels: Object<string, number>, header_score?: number}>}
   */
  async predict(text, options = {}) {
    const url = `${this.baseUrl}/v1/predict`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    const body = { text };
    if (options.headers) {
      body.headers = options.headers;
    }

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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
   * Call POST /v1/feedback.
   *
   * @param {string} text   The email text to report.
   * @param {string} label  "spam" or "ham".
   * @param {string} [source="unknown"]  Reporting MTA identifier.
   * @returns {Promise<{status: string, id: string}>}
   */
  async feedback(text, label, source = "unknown") {
    const url = `${this.baseUrl}/v1/feedback`;
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
   * Call GET /v1/health.
   *
   * @returns {Promise<{status: string}>}
   */
  async health() {
    const url = `${this.baseUrl}/v1/health`;
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
