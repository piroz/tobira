"""Web dashboard for tobira monitoring and statistics.

Provides API endpoints for statistics data and a self-contained HTML dashboard
that visualizes spam detection metrics, score distributions, drift detection
status, and backend health.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _read_log_records(log_path: str) -> list[dict[str, Any]]:
    """Read prediction log records from a JSONL file.

    Returns an empty list if the file does not exist.
    """
    p = Path(log_path)
    if not p.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _compute_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from prediction log records."""
    now = datetime.now(timezone.utc)
    total = len(records)

    def _count_in_window(seconds: int) -> dict[str, int]:
        spam = 0
        ham = 0
        other = 0
        for r in records:
            ts = r.get("timestamp")
            if not isinstance(ts, str):
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                continue
            if (now - dt).total_seconds() <= seconds:
                label = r.get("label")
                if label == "spam":
                    spam += 1
                elif label == "ham":
                    ham += 1
                else:
                    other += 1
        return {"spam": spam, "ham": ham, "other": other}

    last_24h = _count_in_window(86400)
    last_7d = _count_in_window(86400 * 7)

    avg_latency = 0.0
    latency_count = 0
    for r in records:
        lat = r.get("latency_ms")
        if isinstance(lat, (int, float)):
            avg_latency += lat
            latency_count += 1
    if latency_count > 0:
        avg_latency = round(avg_latency / latency_count, 2)

    return {
        "total_predictions": total,
        "last_24h": last_24h,
        "last_7d": last_7d,
        "avg_latency_ms": avg_latency,
    }


def _compute_distribution(
    records: list[dict[str, Any]],
    bins: int = 20,
) -> dict[str, Any]:
    """Compute score distribution histogram."""
    scores: list[float] = []
    for r in records:
        s = r.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))

    if not scores:
        return {"bins": [], "counts": [], "total": 0}

    bin_width = 1.0 / bins
    counts = [0] * bins
    for s in scores:
        idx = min(int(s / bin_width), bins - 1)
        if idx < 0:
            idx = 0
        counts[idx] += 1

    bin_labels = [round(i * bin_width, 3) for i in range(bins)]
    return {"bins": bin_labels, "counts": counts, "total": len(scores)}


def _compute_drift(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute drift detection results from prediction records."""
    scores: list[float] = []
    for r in records:
        s = r.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))

    min_samples = 30
    if len(scores) < min_samples * 2:
        return {
            "status": "insufficient_data",
            "message": f"Need at least {min_samples * 2} samples, have {len(scores)}.",
            "psi": None,
            "ks": None,
        }

    mid = len(scores) // 2
    reference = scores[:mid]
    current = scores[mid:]

    try:
        from tobira.monitoring.analyzer import compute_ks, compute_psi

        psi_result = compute_psi(reference, current)
        ks_result = compute_ks(reference, current)
        return {
            "status": psi_result.level,
            "psi": {"value": psi_result.psi, "level": psi_result.level},
            "ks": {
                "statistic": ks_result.statistic,
                "significant": ks_result.significant,
            },
        }
    except Exception as e:
        logger.debug("Drift computation failed: %s", e)
        return {"status": "error", "message": str(e), "psi": None, "ks": None}


def _get_recent_predictions(
    records: list[dict[str, Any]],
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return the most recent prediction records for the feedback UI.

    Only returns metadata (timestamp, label, score); never exposes raw text.
    """
    recent = records[-limit:] if len(records) > limit else list(records)
    recent.reverse()
    result: list[dict[str, Any]] = []
    for idx, r in enumerate(recent):
        result.append({
            "index": len(records) - 1 - idx,
            "timestamp": r.get("timestamp", ""),
            "label": r.get("label", ""),
            "score": r.get("score", 0.0),
            "latency_ms": r.get("latency_ms"),
        })
    return result


def _compute_feedback_stats(feedback_path: str) -> dict[str, Any]:
    """Compute feedback statistics from the feedback JSONL file."""
    p = Path(feedback_path)
    if not p.exists():
        return {"total": 0, "spam_reports": 0, "ham_reports": 0}

    total = 0
    spam_reports = 0
    ham_reports = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            label = record.get("label", "")
            if label == "spam":
                spam_reports += 1
            elif label == "ham":
                ham_reports += 1
    return {"total": total, "spam_reports": spam_reports, "ham_reports": ham_reports}


def _get_backend_status(backend: Any) -> dict[str, Any]:
    """Get backend health and metadata."""
    backend_type = type(backend).__name__
    return {
        "type": backend_type,
        "status": "ok",
    }


def get_dashboard_html(*, feedback_enabled: bool = False) -> str:
    """Return the self-contained HTML dashboard page.

    Args:
        feedback_enabled: When *True*, the feedback UI section is included.
    """
    if feedback_enabled:
        return _DASHBOARD_HTML.replace(
            "<!-- FEEDBACK_SECTION -->", _FEEDBACK_SECTION_HTML
        ).replace(
            "// FEEDBACK_JS", _FEEDBACK_JS
        )
    return _DASHBOARD_HTML


def register_dashboard_routes(
    app: Any,
    log_path: str = "/var/lib/tobira/predictions.jsonl",
    feedback_path: str | None = None,
) -> None:
    """Register dashboard endpoints on a FastAPI app.

    Args:
        app: A FastAPI application instance.
        log_path: Path to the prediction log JSONL file.
        feedback_path: Optional path to the feedback JSONL file.
            When provided, feedback UI and endpoints are enabled.
    """
    _get_fastapi()
    from fastapi.responses import HTMLResponse, JSONResponse

    @app.get("/dashboard", response_class=HTMLResponse, tags=["dashboard"])
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(
            content=get_dashboard_html(feedback_enabled=feedback_path is not None),
        )

    @app.get(
        "/api/stats/summary",
        response_class=JSONResponse,
        tags=["dashboard"],
    )
    async def stats_summary() -> JSONResponse:
        records = _read_log_records(log_path)
        return JSONResponse(content=_compute_summary(records))

    @app.get(
        "/api/stats/distribution",
        response_class=JSONResponse,
        tags=["dashboard"],
    )
    async def stats_distribution() -> JSONResponse:
        records = _read_log_records(log_path)
        return JSONResponse(content=_compute_distribution(records))

    @app.get(
        "/api/stats/drift",
        response_class=JSONResponse,
        tags=["dashboard"],
    )
    async def stats_drift() -> JSONResponse:
        records = _read_log_records(log_path)
        return JSONResponse(content=_compute_drift(records))

    @app.get(
        "/api/stats/backends",
        response_class=JSONResponse,
        tags=["dashboard"],
    )
    async def stats_backends() -> JSONResponse:
        backend = app.state.backend
        return JSONResponse(content=_get_backend_status(backend))

    @app.get(
        "/api/predictions/recent",
        response_class=JSONResponse,
        tags=["dashboard"],
    )
    async def predictions_recent() -> JSONResponse:
        records = _read_log_records(log_path)
        return JSONResponse(content=_get_recent_predictions(records))

    if feedback_path is not None:
        from tobira.serving._feedback_routes import register_feedback_routes

        register_feedback_routes(app, feedback_path)


def _get_fastapi() -> Any:
    """Import FastAPI lazily."""
    try:
        import fastapi
    except ImportError:
        raise ImportError(
            "serving dependencies are not installed. "
            "Install them with: pip install tobira[serving]"
        ) from None
    return fastapi


_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>tobira Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    background: #f5f5f5; color: #333; line-height: 1.6;
  }
  header {
    background: #1a1a2e; color: #fff; padding: 1rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
  }
  header h1 { font-size: 1.4rem; font-weight: 600; }
  header .refresh-info { font-size: 0.85rem; opacity: 0.7; }
  .container { max-width: 1200px; margin: 0 auto; padding: 1.5rem; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem; margin-bottom: 1.5rem;
  }
  .card {
    background: #fff; border-radius: 8px; padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }
  .card h2 {
    font-size: 1rem; color: #666; margin-bottom: 1rem;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .stat-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
  .stat-label { font-size: 0.85rem; color: #888; }
  .stat-row {
    display: flex; justify-content: space-between;
    padding: 0.5rem 0; border-bottom: 1px solid #eee;
  }
  .stat-row:last-child { border-bottom: none; }
  .badge {
    display: inline-block; padding: 0.2rem 0.6rem;
    border-radius: 4px; font-size: 0.8rem; font-weight: 600;
  }
  .badge-ok { background: #d4edda; color: #155724; }
  .badge-warning { background: #fff3cd; color: #856404; }
  .badge-alert { background: #f8d7da; color: #721c24; }
  .badge-info { background: #d1ecf1; color: #0c5460; }
  .chart-container {
    width: 100%; height: 200px; position: relative;
    margin-top: 0.5rem;
  }
  .bar-chart {
    display: flex; align-items: flex-end; height: 180px;
    gap: 2px; padding-top: 10px;
  }
  .bar {
    flex: 1; background: #4a90d9; border-radius: 2px 2px 0 0;
    min-width: 4px; transition: height 0.3s;
  }
  .bar:hover { background: #2c6fbb; }
  .chart-labels {
    display: flex; justify-content: space-between;
    font-size: 0.7rem; color: #888; margin-top: 4px;
  }
  .wide { grid-column: 1 / -1; }
  .error-msg { color: #dc3545; font-style: italic; }
  #loading { text-align: center; padding: 2rem; color: #888; }
  .pred-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  .pred-table th, .pred-table td {
    padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #eee;
  }
  .pred-table th { color: #666; font-weight: 600; }
  .fb-btn {
    padding: 0.25rem 0.6rem; border: 1px solid #ccc; border-radius: 4px;
    background: #fff; cursor: pointer; font-size: 0.8rem;
  }
  .fb-btn:hover { background: #f0f0f0; }
  .fb-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .fb-spam { color: #dc3545; border-color: #dc3545; }
  .fb-ham { color: #28a745; border-color: #28a745; }
</style>
</head>
<body>
<header>
  <h1>tobira Dashboard</h1>
  <span class="refresh-info" id="refresh-info">Auto-refresh: 30s</span>
</header>
<div class="container">
  <div id="loading">Loading dashboard data...</div>
  <div id="content" style="display:none;">

    <div class="grid">
      <div class="card">
        <h2>Total Predictions</h2>
        <div class="stat-value" id="total-predictions">-</div>
        <div class="stat-label">all time</div>
      </div>
      <div class="card">
        <h2>Last 24 Hours</h2>
        <div class="stat-row">
          <span>Spam</span><span id="spam-24h">-</span>
        </div>
        <div class="stat-row">
          <span>Ham</span><span id="ham-24h">-</span>
        </div>
        <div class="stat-row">
          <span>Other</span><span id="other-24h">-</span>
        </div>
      </div>
      <div class="card">
        <h2>Last 7 Days</h2>
        <div class="stat-row">
          <span>Spam</span><span id="spam-7d">-</span>
        </div>
        <div class="stat-row">
          <span>Ham</span><span id="ham-7d">-</span>
        </div>
        <div class="stat-row">
          <span>Other</span><span id="other-7d">-</span>
        </div>
      </div>
      <div class="card">
        <h2>Average Latency</h2>
        <div class="stat-value" id="avg-latency">-</div>
        <div class="stat-label">milliseconds</div>
      </div>
    </div>

    <div class="grid">
      <div class="card wide">
        <h2>Score Distribution</h2>
        <div class="chart-container">
          <div class="bar-chart" id="score-chart"></div>
          <div class="chart-labels">
            <span>0.0</span><span>0.25</span><span>0.5</span>
            <span>0.75</span><span>1.0</span>
          </div>
        </div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Drift Detection</h2>
        <div id="drift-status">-</div>
      </div>
      <div class="card">
        <h2>Backend Status</h2>
        <div id="backend-status">-</div>
      </div>
    </div>

    <!-- FEEDBACK_SECTION -->

  </div>
</div>
<script>
(function() {
  var REFRESH_INTERVAL = 30000;

  function fetchJSON(url) {
    return fetch(url).then(function(r) { return r.json(); });
  }

  function badgeHTML(level, text) {
    var cls = "badge-info";
    if (level === "ok") cls = "badge-ok";
    else if (level === "warning") cls = "badge-warning";
    else if (level === "alert") cls = "badge-alert";
    return '<span class="badge ' + cls + '">' + text + '</span>';
  }

  function updateSummary(data) {
    document.getElementById("total-predictions").textContent =
      data.total_predictions.toLocaleString();
    document.getElementById("spam-24h").textContent = data.last_24h.spam;
    document.getElementById("ham-24h").textContent = data.last_24h.ham;
    document.getElementById("other-24h").textContent = data.last_24h.other;
    document.getElementById("spam-7d").textContent = data.last_7d.spam;
    document.getElementById("ham-7d").textContent = data.last_7d.ham;
    document.getElementById("other-7d").textContent = data.last_7d.other;
    document.getElementById("avg-latency").textContent = data.avg_latency_ms;
  }

  function updateDistribution(data) {
    var chart = document.getElementById("score-chart");
    if (!data.counts || data.counts.length === 0) {
      chart.innerHTML = '<span class="error-msg">No data available</span>';
      return;
    }
    var max = Math.max.apply(null, data.counts);
    if (max === 0) max = 1;
    var html = "";
    for (var i = 0; i < data.counts.length; i++) {
      var pct = (data.counts[i] / max) * 100;
      html += '<div class="bar" style="height:' + pct +
        '%" title="' + data.bins[i].toFixed(2) + ': ' +
        data.counts[i] + '"></div>';
    }
    chart.innerHTML = html;
  }

  function updateDrift(data) {
    var el = document.getElementById("drift-status");
    if (data.status === "insufficient_data") {
      el.innerHTML = badgeHTML("info", "Insufficient Data") +
        '<p style="margin-top:0.5rem;font-size:0.9rem;">' +
        data.message + '</p>';
      return;
    }
    if (data.status === "error") {
      el.innerHTML = '<span class="error-msg">' + data.message + '</span>';
      return;
    }
    var html = badgeHTML(data.status,
      data.status.charAt(0).toUpperCase() + data.status.slice(1));
    if (data.psi) {
      html += '<div class="stat-row"><span>PSI</span><span>' +
        data.psi.value.toFixed(4) + '</span></div>';
    }
    if (data.ks) {
      html += '<div class="stat-row"><span>KS Statistic</span><span>' +
        data.ks.statistic.toFixed(4) + '</span></div>';
      html += '<div class="stat-row"><span>KS Significant</span><span>' +
        (data.ks.significant ? "Yes" : "No") + '</span></div>';
    }
    el.innerHTML = html;
  }

  function updateBackends(data) {
    var el = document.getElementById("backend-status");
    var html = '<div class="stat-row"><span>Type</span><span>' +
      data.type + '</span></div>';
    html += '<div class="stat-row"><span>Status</span><span>' +
      badgeHTML("ok", data.status) + '</span></div>';
    el.innerHTML = html;
  }

  function refresh() {
    Promise.all([
      fetchJSON("/api/stats/summary"),
      fetchJSON("/api/stats/distribution"),
      fetchJSON("/api/stats/drift"),
      fetchJSON("/api/stats/backends")
    ]).then(function(results) {
      updateSummary(results[0]);
      updateDistribution(results[1]);
      updateDrift(results[2]);
      updateBackends(results[3]);
      document.getElementById("loading").style.display = "none";
      document.getElementById("content").style.display = "block";
    }).catch(function(err) {
      console.error("Dashboard refresh failed:", err);
    });
  }

  // FEEDBACK_JS

  refresh();
  setInterval(refresh, REFRESH_INTERVAL);
})();
</script>
</body>
</html>
"""

_FEEDBACK_SECTION_HTML = """\
    <div class="grid">
      <div class="card">
        <h2>Feedback Stats</h2>
        <div class="stat-row">
          <span>Total Reports</span><span id="fb-total">-</span>
        </div>
        <div class="stat-row">
          <span>Spam Reports</span><span id="fb-spam">-</span>
        </div>
        <div class="stat-row">
          <span>Ham Reports</span><span id="fb-ham">-</span>
        </div>
      </div>
    </div>

    <div class="grid">
      <div class="card wide">
        <h2>Recent Predictions</h2>
        <div id="predictions-table-wrap">
          <table class="pred-table" id="predictions-table">
            <thead>
              <tr>
                <th>Time</th><th>Label</th><th>Score</th>
                <th>Latency</th><th>Feedback</th>
              </tr>
            </thead>
            <tbody id="predictions-body"></tbody>
          </table>
        </div>
      </div>
    </div>
"""

_FEEDBACK_JS = """\

  function updateFeedbackStats(data) {
    document.getElementById("fb-total").textContent = data.total;
    document.getElementById("fb-spam").textContent = data.spam_reports;
    document.getElementById("fb-ham").textContent = data.ham_reports;
  }

  function updatePredictions(data) {
    var tbody = document.getElementById("predictions-body");
    if (!data || data.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="error-msg">' +
        'No predictions</td></tr>';
      return;
    }
    var html = "";
    for (var i = 0; i < data.length; i++) {
      var p = data[i];
      var ts = p.timestamp ? new Date(p.timestamp).toLocaleString() : "-";
      var lat = p.latency_ms != null ? p.latency_ms.toFixed(1) + " ms" : "-";
      var spamBtn = '<button class="fb-btn fb-spam" data-idx="' + p.index +
        '" onclick="sendFeedback(this, \'spam\')">Spam</button>';
      var hamBtn = '<button class="fb-btn fb-ham" data-idx="' + p.index +
        '" onclick="sendFeedback(this, \'ham\')">Not Spam</button>';
      html += "<tr><td>" + ts + "</td><td>" + p.label + "</td><td>" +
        (typeof p.score === "number" ? p.score.toFixed(4) : "-") +
        "</td><td>" + lat + "</td><td>" + spamBtn + " " + hamBtn +
        "</td></tr>";
    }
    tbody.innerHTML = html;
  }

  window.sendFeedback = function(btn, correctLabel) {
    var idx = btn.getAttribute("data-idx");
    var row = btn.closest("tr");
    var cells = row.getElementsByTagName("td");
    btn.disabled = true;
    fetch("/api/feedback", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        text: "prediction-index:" + idx,
        correct_label: correctLabel,
        source: "dashboard"
      })
    }).then(function(r) { return r.json(); }).then(function(data) {
      if (data.status === "accepted") {
        cells[4].innerHTML = '<span class="badge badge-ok">Reported: ' +
          correctLabel + '</span>';
        refreshFeedbackStats();
      }
    }).catch(function() {
      btn.disabled = false;
    });
  };

  function refreshFeedbackStats() {
    fetchJSON("/api/feedback/stats").then(updateFeedbackStats).catch(function(){});
  }

  var origRefresh = refresh;
  refresh = function() {
    origRefresh();
    fetchJSON("/api/predictions/recent").then(updatePredictions).catch(function(){});
    refreshFeedbackStats();
  };
"""
