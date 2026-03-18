"""Web dashboard for tobira monitoring and statistics.

Provides API endpoints for statistics data and a self-contained HTML dashboard
that visualizes spam detection metrics, score distributions, drift detection
status, and backend health.  The dashboard uses Chart.js for interactive charts
and a traffic-light status indicator for at-a-glance system health.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
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


def _compute_trend(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute daily spam/ham trend for time-series charts.

    Returns dates, spam counts, and ham counts sorted chronologically.
    """
    daily: dict[str, dict[str, int]] = defaultdict(
        lambda: {"spam": 0, "ham": 0, "other": 0}
    )
    for r in records:
        ts = r.get("timestamp")
        if not isinstance(ts, str):
            continue
        day = ts[:10]
        label = r.get("label")
        if label == "spam":
            daily[day]["spam"] += 1
        elif label == "ham":
            daily[day]["ham"] += 1
        else:
            daily[day]["other"] += 1

    sorted_days = sorted(daily.keys())
    return {
        "dates": sorted_days,
        "spam": [daily[d]["spam"] for d in sorted_days],
        "ham": [daily[d]["ham"] for d in sorted_days],
        "other": [daily[d]["other"] for d in sorted_days],
    }


def _get_recent_predictions(
    records: list[dict[str, Any]],
    page: int = 1,
    per_page: int = 20,
) -> dict[str, Any]:
    """Get paginated recent predictions, most recent first.

    Args:
        records: All prediction log records.
        page: Page number (1-based).
        per_page: Number of records per page.

    Returns:
        Dict with items, total count, page, per_page, and total_pages.
    """
    sorted_records = sorted(
        records,
        key=lambda r: r.get("timestamp", ""),
        reverse=True,
    )
    total = len(sorted_records)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page

    items: list[dict[str, Any]] = []
    for r in sorted_records[start:end]:
        items.append({
            "timestamp": r.get("timestamp", ""),
            "label": r.get("label", ""),
            "score": r.get("score"),
            "latency_ms": r.get("latency_ms"),
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
    }


def _compute_health_status(
    records: list[dict[str, Any]],
    backend: Any,
) -> dict[str, Any]:
    """Compute traffic-light health status for the system.

    Checks API health (backend availability), model load state, and drift
    detection status.  Each component is rated green/yellow/red.

    The overall status is the worst of the individual statuses.

    Returns:
        Dict with overall status and per-component details.
    """
    components: list[dict[str, str]] = []

    # API health (backend availability)
    try:
        backend_type = type(backend).__name__
        components.append({
            "name": "API / Backend",
            "status": "green",
            "detail": f"{backend_type} loaded",
        })
    except Exception:
        components.append({
            "name": "API / Backend",
            "status": "red",
            "detail": "Backend unavailable",
        })

    # Model load state (based on whether we have recent predictions)
    now = datetime.now(timezone.utc)
    recent_count = 0
    for r in records:
        ts = r.get("timestamp")
        if not isinstance(ts, str):
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            continue
        if (now - dt).total_seconds() <= 3600:
            recent_count += 1

    if recent_count > 0:
        components.append({
            "name": "Model Activity",
            "status": "green",
            "detail": f"{recent_count} predictions in last hour",
        })
    elif len(records) > 0:
        components.append({
            "name": "Model Activity",
            "status": "yellow",
            "detail": "No predictions in last hour",
        })
    else:
        components.append({
            "name": "Model Activity",
            "status": "yellow",
            "detail": "No prediction data yet",
        })

    # Drift detection status
    drift = _compute_drift(records)
    drift_status = drift.get("status", "unknown")
    if drift_status == "ok":
        components.append({
            "name": "Drift Detection",
            "status": "green",
            "detail": "No drift detected",
        })
    elif drift_status == "warning":
        components.append({
            "name": "Drift Detection",
            "status": "yellow",
            "detail": "Score distribution shift detected",
        })
    elif drift_status == "alert":
        components.append({
            "name": "Drift Detection",
            "status": "red",
            "detail": "Significant score distribution shift",
        })
    else:
        components.append({
            "name": "Drift Detection",
            "status": "yellow",
            "detail": drift.get("message", "Insufficient data"),
        })

    # Overall status: worst of all components
    priority = {"red": 0, "yellow": 1, "green": 2}
    overall = "green"
    for c in components:
        if priority.get(c["status"], 2) < priority.get(overall, 2):
            overall = c["status"]

    return {
        "overall": overall,
        "components": components,
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
        "/api/stats/trend",
        response_class=JSONResponse,
        tags=["dashboard"],
    )
    async def stats_trend() -> JSONResponse:
        records = _read_log_records(log_path)
        return JSONResponse(content=_compute_trend(records))

    @app.get(
        "/api/stats/recent",
        response_class=JSONResponse,
        tags=["dashboard"],
    )
    async def stats_recent(page: int = 1, per_page: int = 20) -> JSONResponse:
        records = _read_log_records(log_path)
        return JSONResponse(
            content=_get_recent_predictions(records, page=page, per_page=per_page)
        )

    @app.get(
        "/api/stats/health",
        response_class=JSONResponse,
        tags=["dashboard"],
    )
    async def stats_health() -> JSONResponse:
        records = _read_log_records(log_path)
        backend = app.state.backend
        return JSONResponse(content=_compute_health_status(records, backend))

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
  header .header-right { display: flex; align-items: center; gap: 1rem; }
  header .refresh-info { font-size: 0.85rem; opacity: 0.7; }
  .traffic-light {
    display: inline-flex; align-items: center; gap: 0.5rem;
    font-size: 0.85rem;
  }
  .traffic-dot {
    width: 14px; height: 14px; border-radius: 50%;
    display: inline-block; border: 2px solid rgba(255,255,255,0.3);
  }
  .traffic-dot.green { background: #28a745; }
  .traffic-dot.yellow { background: #ffc107; }
  .traffic-dot.red { background: #dc3545; }
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
  .badge-ok, .badge-green { background: #d4edda; color: #155724; }
  .badge-warning, .badge-yellow { background: #fff3cd; color: #856404; }
  .badge-alert, .badge-red { background: #f8d7da; color: #721c24; }
  .badge-info { background: #d1ecf1; color: #0c5460; }
  .chart-container {
    width: 100%; height: 250px; position: relative;
    margin-top: 0.5rem;
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
  .health-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }
  .health-item {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.75rem; border-radius: 6px; background: #f8f9fa;
  }
  .health-dot {
    width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0;
  }
  .health-dot.green { background: #28a745; }
  .health-dot.yellow { background: #ffc107; }
  .health-dot.red { background: #dc3545; }
  .health-text { font-size: 0.85rem; }
  .health-text .name { font-weight: 600; }
  .health-text .detail { color: #666; }
  table.log-table {
    width: 100%; border-collapse: collapse; font-size: 0.85rem;
  }
  table.log-table th {
    text-align: left; padding: 0.5rem; border-bottom: 2px solid #dee2e6;
    color: #666; text-transform: uppercase; font-size: 0.75rem;
    letter-spacing: 0.05em;
  }
  table.log-table td {
    padding: 0.5rem; border-bottom: 1px solid #eee;
  }
  .pagination {
    display: flex; justify-content: center; align-items: center;
    gap: 0.5rem; margin-top: 1rem;
  }
  .pagination button {
    padding: 0.3rem 0.8rem; border: 1px solid #dee2e6;
    border-radius: 4px; background: #fff; cursor: pointer;
    font-size: 0.85rem;
  }
  .pagination button:hover { background: #e9ecef; }
  .pagination button:disabled { opacity: 0.5; cursor: default; }
  .pagination .page-info { font-size: 0.85rem; color: #666; }
</style>
</head>
<body>
<header>
  <h1>tobira Dashboard</h1>
  <div class="header-right">
    <div class="traffic-light" id="header-traffic">
      <span class="traffic-dot" id="header-dot"></span>
      <span id="header-status-text">Loading...</span>
    </div>
    <span class="refresh-info" id="refresh-info">Auto-refresh: 30s</span>
  </div>
</header>
<div class="container">
  <div id="loading">Loading dashboard data...</div>
  <div id="content" style="display:none;">

    <div class="grid">
      <div class="card wide">
        <h2>System Health</h2>
        <div class="health-grid" id="health-grid"></div>
      </div>
    </div>

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
        <h2>Spam / Ham Trend</h2>
        <div class="chart-container">
          <canvas id="trend-chart"></canvas>
        </div>
      </div>
    </div>

    <div class="grid">
      <div class="card wide">
        <h2>Score Distribution</h2>
        <div class="chart-container">
          <canvas id="score-chart"></canvas>
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

    <div class="grid">
      <div class="card wide">
        <h2>Recent Predictions</h2>
        <table class="log-table">
          <thead>
            <tr>
              <th>Timestamp</th><th>Label</th><th>Score</th><th>Latency (ms)</th>
            </tr>
          </thead>
          <tbody id="recent-tbody"></tbody>
        </table>
        <div class="pagination" id="pagination"></div>
      </div>
    </div>

    <!-- FEEDBACK_SECTION -->

  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script>
(function() {
  var REFRESH_INTERVAL = 30000;
  var currentPage = 1;
  var trendChart = null;
  var scoreChart = null;

  function fetchJSON(url) {
    return fetch(url).then(function(r) { return r.json(); });
  }

  function badgeHTML(level, text) {
    var cls = "badge-info";
    if (level === "ok" || level === "green") cls = "badge-green";
    else if (level === "warning" || level === "yellow") cls = "badge-yellow";
    else if (level === "alert" || level === "red") cls = "badge-red";
    return '<span class="badge ' + cls + '">' + text + '</span>';
  }

  function escapeHtml(s) {
    if (s == null) return "-";
    var d = document.createElement("div");
    d.appendChild(document.createTextNode(String(s)));
    return d.innerHTML;
  }

  function updateHealth(data) {
    var dot = document.getElementById("header-dot");
    var text = document.getElementById("header-status-text");
    dot.className = "traffic-dot " + data.overall;
    var labels = {green: "Healthy", yellow: "Warning", red: "Critical"};
    text.textContent = labels[data.overall] || "Unknown";

    var grid = document.getElementById("health-grid");
    var html = "";
    for (var i = 0; i < data.components.length; i++) {
      var c = data.components[i];
      html += '<div class="health-item">' +
        '<span class="health-dot ' + c.status + '"></span>' +
        '<div class="health-text">' +
        '<div class="name">' + escapeHtml(c.name) + '</div>' +
        '<div class="detail">' + escapeHtml(c.detail) + '</div>' +
        '</div></div>';
    }
    grid.innerHTML = html;
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

  function updateTrend(data) {
    var ctx = document.getElementById("trend-chart");
    if (!data.dates || data.dates.length === 0) return;
    var cfg = {
      type: "line",
      data: {
        labels: data.dates,
        datasets: [
          {
            label: "Spam",
            data: data.spam,
            borderColor: "#dc3545",
            backgroundColor: "rgba(220,53,69,0.1)",
            fill: true, tension: 0.3
          },
          {
            label: "Ham",
            data: data.ham,
            borderColor: "#28a745",
            backgroundColor: "rgba(40,167,69,0.1)",
            fill: true, tension: 0.3
          }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true }
        }
      }
    };
    if (trendChart) {
      trendChart.data = cfg.data;
      trendChart.update();
    } else {
      trendChart = new Chart(ctx, cfg);
    }
  }

  function updateDistribution(data) {
    var ctx = document.getElementById("score-chart");
    if (!data.counts || data.counts.length === 0) return;
    var labels = data.bins.map(function(b) { return b.toFixed(2); });
    var cfg = {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Count",
          data: data.counts,
          backgroundColor: "rgba(74,144,217,0.7)",
          borderColor: "#4a90d9",
          borderWidth: 1
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false },
               title: { display: true, text: "Score" } },
          y: { beginAtZero: true,
               title: { display: true, text: "Count" } }
        }
      }
    };
    if (scoreChart) {
      scoreChart.data = cfg.data;
      scoreChart.update();
    } else {
      scoreChart = new Chart(ctx, cfg);
    }
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

  function updateRecent(data) {
    var tbody = document.getElementById("recent-tbody");
    var html = "";
    for (var i = 0; i < data.items.length; i++) {
      var item = data.items[i];
      var scoreStr = item.score != null ? item.score.toFixed(4) : "-";
      var latStr = item.latency_ms != null ? item.latency_ms.toFixed(1) : "-";
      var labelBadge = "";
      if (item.label === "spam") {
        labelBadge = badgeHTML("alert", "spam");
      } else if (item.label === "ham") {
        labelBadge = badgeHTML("ok", "ham");
      } else {
        labelBadge = badgeHTML("info", escapeHtml(item.label) || "-");
      }
      html += "<tr><td>" + escapeHtml(item.timestamp) + "</td>" +
        "<td>" + labelBadge + "</td>" +
        "<td>" + scoreStr + "</td>" +
        "<td>" + latStr + "</td></tr>";
    }
    tbody.innerHTML = html || '<tr><td colspan="4">No data</td></tr>';

    var pag = document.getElementById("pagination");
    pag.innerHTML =
      '<button id="prev-btn">&laquo; Prev</button>' +
      '<span class="page-info">Page ' + data.page +
      ' / ' + data.total_pages + '</span>' +
      '<button id="next-btn">Next &raquo;</button>';

    document.getElementById("prev-btn").disabled = data.page <= 1;
    document.getElementById("next-btn").disabled = data.page >= data.total_pages;
    document.getElementById("prev-btn").onclick = function() {
      currentPage = Math.max(1, data.page - 1);
      refreshRecent();
    };
    document.getElementById("next-btn").onclick = function() {
      currentPage = Math.min(data.total_pages, data.page + 1);
      refreshRecent();
    };
  }

  function refreshRecent() {
    fetchJSON("/api/stats/recent?page=" + currentPage + "&per_page=20")
      .then(updateRecent);
  }

  function refresh() {
    Promise.all([
      fetchJSON("/api/stats/summary"),
      fetchJSON("/api/stats/distribution"),
      fetchJSON("/api/stats/drift"),
      fetchJSON("/api/stats/backends"),
      fetchJSON("/api/stats/trend"),
      fetchJSON("/api/stats/recent?page=" + currentPage + "&per_page=20"),
      fetchJSON("/api/stats/health")
    ]).then(function(results) {
      updateSummary(results[0]);
      updateDistribution(results[1]);
      updateDrift(results[2]);
      updateBackends(results[3]);
      updateTrend(results[4]);
      updateRecent(results[5]);
      updateHealth(results[6]);
      document.getElementById("loading").style.display = "none";
      document.getElementById("content").style.display = "block";
    }).catch(function(err) {
      console.error("Dashboard refresh failed:", err);
    });
  }

  // FEEDBACK_JS

  if (typeof Chart === "undefined") {
    var s = document.createElement("script");
    s.onload = function() { refresh(); setInterval(refresh, REFRESH_INTERVAL); };
    s.onerror = function() {
      document.getElementById("loading").textContent =
        "Failed to load Chart.js. Charts will not render.";
      refresh();
      setInterval(refresh, REFRESH_INTERVAL);
    };
    s.src = "https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js";
    document.head.appendChild(s);
  } else {
    refresh();
    setInterval(refresh, REFRESH_INTERVAL);
  }
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
