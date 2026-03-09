"use strict";

const { TobiraClient } = require("tobira");

exports.register = function () {
  this.cfg = this.config.get("tobira.ini", {
    booleans: ["+main.reject_spam"],
  });

  const main = this.cfg.main || {};
  const url = main.url || "http://127.0.0.1:8000";
  const timeout = main.timeout !== undefined ? Number(main.timeout) : 5000;
  const threshold = main.threshold !== undefined ? Number(main.threshold) : 0.5;
  const rejectSpam = main.reject_spam !== undefined ? main.reject_spam : true;

  this.client = new TobiraClient(url, { timeout });
  this.threshold = threshold;
  this.rejectSpam = rejectSpam;

  this.register_hook("data_post", "check_spam");
};

exports.check_spam = function (next, connection) {
  const transaction = connection.transaction;
  if (!transaction) {
    return next();
  }

  const bodyText = collect_body(transaction);
  if (!bodyText) {
    return next();
  }

  this.client
    .predict(bodyText)
    .then((result) => {
      transaction.results.add(this, {
        pass: result.label !== "spam" ? `score=${result.score}` : undefined,
        fail: result.label === "spam" ? `score=${result.score}` : undefined,
        score: result.score,
        label: result.label,
      });

      if (result.label === "spam" && result.score >= this.threshold) {
        if (this.rejectSpam) {
          return next(DENY, "Message detected as spam");
        }
      }

      return next();
    })
    .catch((err) => {
      connection.logerror(this, `tobira API error: ${err.message}`);
      transaction.results.add(this, {
        fail: "api_error",
        err: err.message,
      });
      return next();
    });
};

function collect_body(transaction) {
  const body = transaction.body;
  if (!body) {
    return "";
  }

  const parts = [];
  collect_text_parts(body, parts);
  return parts.join("\n");
}

function collect_text_parts(node, parts) {
  const ct = (node.ct || "").toLowerCase();
  if (ct.startsWith("text/")) {
    const decoded = node.bodytext || "";
    if (decoded) {
      parts.push(decoded);
    }
  }

  if (node.children && node.children.length > 0) {
    for (const child of node.children) {
      collect_text_parts(child, parts);
    }
  }
}
