"use strict";

const { describe, it, beforeEach, afterEach, mock } = require("node:test");
const assert = require("node:assert/strict");
const { TobiraClient } = require("../src/client");

describe("TobiraClient", () => {
  let originalFetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  describe("constructor", () => {
    it("should strip trailing slashes from baseUrl", () => {
      const client = new TobiraClient("http://localhost:8000///");
      assert.equal(client.baseUrl, "http://localhost:8000");
    });

    it("should use default timeout of 5000ms", () => {
      const client = new TobiraClient("http://localhost:8000");
      assert.equal(client.timeout, 5000);
    });

    it("should accept custom timeout", () => {
      const client = new TobiraClient("http://localhost:8000", { timeout: 3000 });
      assert.equal(client.timeout, 3000);
    });
  });

  describe("predict", () => {
    it("should POST to /predict and return result", async () => {
      const expected = { label: "ham", score: 0.95, labels: { ham: 0.95, spam: 0.05 } };
      globalThis.fetch = mock.fn(async () => ({
        ok: true,
        json: async () => expected,
      }));

      const client = new TobiraClient("http://localhost:8000");
      const result = await client.predict("hello world");

      assert.deepEqual(result, expected);

      const call = globalThis.fetch.mock.calls[0];
      assert.equal(call.arguments[0], "http://localhost:8000/predict");
      const opts = call.arguments[1];
      assert.equal(opts.method, "POST");
      assert.equal(opts.headers["Content-Type"], "application/json");
      assert.equal(opts.body, JSON.stringify({ text: "hello world" }));
    });

    it("should throw on non-ok response", async () => {
      globalThis.fetch = mock.fn(async () => ({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
      }));

      const client = new TobiraClient("http://localhost:8000");
      await assert.rejects(
        () => client.predict("test"),
        { message: "tobira API error: 500 Internal Server Error" }
      );
    });

    it("should propagate network errors", async () => {
      globalThis.fetch = mock.fn(async () => {
        throw new TypeError("fetch failed");
      });

      const client = new TobiraClient("http://localhost:8000");
      await assert.rejects(
        () => client.predict("test"),
        { name: "TypeError", message: "fetch failed" }
      );
    });

    it("should propagate abort errors on timeout", async () => {
      globalThis.fetch = mock.fn(async (_url, opts) => {
        return new Promise((_resolve, reject) => {
          opts.signal.addEventListener("abort", () => {
            reject(new DOMException("The operation was aborted", "AbortError"));
          });
        });
      });

      const client = new TobiraClient("http://localhost:8000", { timeout: 1 });
      await assert.rejects(
        () => client.predict("test"),
        { name: "AbortError" }
      );
    });
  });

  describe("feedback", () => {
    it("should POST to /feedback and return result", async () => {
      const expected = { status: "accepted", id: "abc-123" };
      globalThis.fetch = mock.fn(async () => ({
        ok: true,
        json: async () => expected,
      }));

      const client = new TobiraClient("http://localhost:8000");
      const result = await client.feedback("spam email", "spam", "rspamd");

      assert.deepEqual(result, expected);

      const call = globalThis.fetch.mock.calls[0];
      assert.equal(call.arguments[0], "http://localhost:8000/feedback");
      const opts = call.arguments[1];
      assert.equal(opts.method, "POST");
      assert.equal(opts.headers["Content-Type"], "application/json");
      const body = JSON.parse(opts.body);
      assert.equal(body.text, "spam email");
      assert.equal(body.label, "spam");
      assert.equal(body.source, "rspamd");
    });

    it("should use default source when not provided", async () => {
      globalThis.fetch = mock.fn(async () => ({
        ok: true,
        json: async () => ({ status: "accepted", id: "abc-123" }),
      }));

      const client = new TobiraClient("http://localhost:8000");
      await client.feedback("text", "ham");

      const body = JSON.parse(globalThis.fetch.mock.calls[0].arguments[1].body);
      assert.equal(body.source, "unknown");
    });

    it("should throw on non-ok response", async () => {
      globalThis.fetch = mock.fn(async () => ({
        ok: false,
        status: 422,
        statusText: "Unprocessable Entity",
      }));

      const client = new TobiraClient("http://localhost:8000");
      await assert.rejects(
        () => client.feedback("text", "invalid"),
        { message: "tobira API error: 422 Unprocessable Entity" }
      );
    });

    it("should propagate abort errors on timeout", async () => {
      globalThis.fetch = mock.fn(async (_url, opts) => {
        return new Promise((_resolve, reject) => {
          opts.signal.addEventListener("abort", () => {
            reject(new DOMException("The operation was aborted", "AbortError"));
          });
        });
      });

      const client = new TobiraClient("http://localhost:8000", { timeout: 1 });
      await assert.rejects(
        () => client.feedback("text", "spam"),
        { name: "AbortError" }
      );
    });
  });

  describe("health", () => {
    it("should GET /health and return result", async () => {
      const expected = { status: "ok" };
      globalThis.fetch = mock.fn(async () => ({
        ok: true,
        json: async () => expected,
      }));

      const client = new TobiraClient("http://localhost:8000");
      const result = await client.health();

      assert.deepEqual(result, expected);

      const call = globalThis.fetch.mock.calls[0];
      assert.equal(call.arguments[0], "http://localhost:8000/health");
    });

    it("should throw on non-ok response", async () => {
      globalThis.fetch = mock.fn(async () => ({
        ok: false,
        status: 503,
        statusText: "Service Unavailable",
      }));

      const client = new TobiraClient("http://localhost:8000");
      await assert.rejects(
        () => client.health(),
        { message: "tobira API error: 503 Service Unavailable" }
      );
    });

    it("should send GET request without body", async () => {
      globalThis.fetch = mock.fn(async () => ({
        ok: true,
        json: async () => ({ status: "ok" }),
      }));

      const client = new TobiraClient("http://localhost:8000");
      await client.health();

      const call = globalThis.fetch.mock.calls[0];
      const opts = call.arguments[1];
      assert.equal(opts.method, undefined);
      assert.equal(opts.body, undefined);
    });

    it("should propagate network errors", async () => {
      globalThis.fetch = mock.fn(async () => {
        throw new TypeError("fetch failed");
      });

      const client = new TobiraClient("http://localhost:8000");
      await assert.rejects(
        () => client.health(),
        { name: "TypeError", message: "fetch failed" }
      );
    });

    it("should propagate abort errors on timeout", async () => {
      globalThis.fetch = mock.fn(async (_url, opts) => {
        return new Promise((_resolve, reject) => {
          opts.signal.addEventListener("abort", () => {
            reject(new DOMException("The operation was aborted", "AbortError"));
          });
        });
      });

      const client = new TobiraClient("http://localhost:8000", { timeout: 1 });
      await assert.rejects(
        () => client.health(),
        { name: "AbortError" }
      );
    });
  });
});
