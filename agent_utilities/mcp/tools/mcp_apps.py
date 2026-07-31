"""MCP Apps: task-progress (`ui://graph-os/task-progress.html`) and trace-waterfall
(`ui://graph-os/trace-waterfall.html`).

CONCEPT:AU-ECO.ui.mcp-apps-host

Registers each app's entry-point tool (``graph_task_progress_app`` /
``graph_trace_waterfall_app``) whose result carries ``_meta.ui.resourceUri`` per the
MCP Apps extension (``io.modelcontextprotocol/ui``, already unconditionally
advertised by fastmcp 4 -- see
``fastmcp.server.low_level.LowLevelServer.get_capabilities``), and the ``ui://``
resource itself: a small, dependency-free HTML/JS page.

Of the original "apps worth shipping" list (trace waterfall, workflow viewer,
evaluation scorecard, task progress, ontology diff, approval form), two now ship:
task progress (first, because it needed nothing new on the server -- it polls the
exact ``graph_jobs`` tool the Tasks↔WorkItem bridge (CONCEPT:AU-ECO.mcp.tasks-workitem-bridge)
already exposes) and trace waterfall (D-25-1 -- it polls the new
``graph_traces action='waterfall'`` action added alongside it, which flattens one
trace's Span/Generation subgraph from the always-on KG-native
``harness.tracing.get_kg_trace_sink()`` sink). The remaining three (workflow viewer,
evaluation scorecard, ontology diff, approval form) are still deferred (see the
deferred-work ledger) -- the framework here (this module's ``AppConfig`` usage plus
the host-side bridge in ``agent-webui``) is the reusable part; each additional app is
just another tool + resource pair following this same shape.

The HTML never calls a tool directly: it only ever *asks its host* to make a
call, over a small postMessage JSON-RPC bridge (``mcpapp/ready`` ->
``mcpapp/init`` -> ``mcpapp/tool-call`` -> ``mcpapp/tool-result`` |
``mcpapp/tool-error``), documented at the top of the HTML and mirrored by the
host-side bridge (``agent-webui/src/lib/mcp-apps/bridge.ts``). The host is
what actually authorizes and executes ``tools/call`` -- this module makes NO
claim about what the host trusts; see that module's own docstring for the
"never trust a server-declared annotation without policy verification" gate.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

TASK_PROGRESS_RESOURCE_URI = "ui://graph-os/task-progress.html"

# Self-contained: no external script/style/font/image loads, so the CSP this
# app needs is the empty set (no connect/resource/frame domains) -- the
# strictest possible declaration. The host still decides what it actually
# grants (never trusts this at face value); see the module docstring.
_TASK_PROGRESS_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Task Progress</title>
<style>
  body { font: 13px system-ui, sans-serif; margin: 12px; color: #1a1a1a; }
  .status { display: inline-block; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
  .status-working { background: #dbeafe; color: #1e40af; }
  .status-input_required { background: #fef3c7; color: #92400e; }
  .status-completed { background: #dcfce7; color: #166534; }
  .status-failed, .status-cancelled { background: #fee2e2; color: #991b1b; }
  #prompt { margin-top: 8px; }
  #err { color: #991b1b; margin-top: 8px; }
</style>
</head>
<body>
  <div>Task <code id="jobId">-</code>: <span id="status" class="status">-</span></div>
  <div id="prompt" hidden>
    <p id="promptText"></p>
    <button id="approve">Approve</button>
    <button id="deny">Deny</button>
  </div>
  <div id="err" hidden></div>
<script>
(function () {
  "use strict";
  // Bridge protocol (see mcp_apps.py / agent-webui bridge.ts docstrings):
  //   iframe -> host: {type: "mcpapp/ready"}
  //   host -> iframe: {type: "mcpapp/init", props: {jobId}}
  //   iframe -> host: {type: "mcpapp/tool-call", id, name, arguments}
  //   host -> iframe: {type: "mcpapp/tool-result", id, result}
  //                 | {type: "mcpapp/tool-error", id, error: {message}}
  var pending = Object.create(null);
  var seq = 0;
  var jobId = null;

  function post(message) {
    window.parent.postMessage(message, "*");
  }

  function callTool(name, args) {
    var id = "call-" + (++seq);
    return new Promise(function (resolve, reject) {
      pending[id] = { resolve: resolve, reject: reject };
      post({ type: "mcpapp/tool-call", id: id, name: name, arguments: args });
    });
  }

  window.addEventListener("message", function (event) {
    var data = event.data;
    if (!data || typeof data !== "object") return;
    if (data.type === "mcpapp/init") {
      jobId = data.props && data.props.jobId;
      document.getElementById("jobId").textContent = jobId || "-";
      poll();
      return;
    }
    if (data.type === "mcpapp/tool-result" && pending[data.id]) {
      pending[data.id].resolve(data.result);
      delete pending[data.id];
      return;
    }
    if (data.type === "mcpapp/tool-error" && pending[data.id]) {
      pending[data.id].reject(new Error((data.error && data.error.message) || "tool error"));
      delete pending[data.id];
    }
  });

  function showError(message) {
    var el = document.getElementById("err");
    el.hidden = false;
    el.textContent = message;
  }

  function render(task) {
    var statusEl = document.getElementById("status");
    statusEl.textContent = task.status;
    statusEl.className = "status status-" + task.status;
    var promptEl = document.getElementById("prompt");
    if (task.status === "input_required") {
      promptEl.hidden = false;
      document.getElementById("promptText").textContent =
        task.statusMessage || "Input required.";
    } else {
      promptEl.hidden = true;
    }
  }

  function poll() {
    if (!jobId) return;
    callTool("graph_jobs", { action: "status", job_id: jobId })
      .then(function (result) {
        var parsed = typeof result === "string" ? JSON.parse(result) : result;
        render(parsed);
        if (parsed.status !== "succeeded" && parsed.status !== "failed" &&
            parsed.status !== "cancelled" && parsed.status !== "dead_letter") {
          setTimeout(poll, 2000);
        }
      })
      .catch(function (err) { showError(String(err && err.message || err)); });
  }

  function respond(approved) {
    if (!jobId) return;
    callTool("graph_jobs", {
      action: "input",
      job_id: jobId,
      input_responses: JSON.stringify({ approved: approved }),
    })
      .then(function () { poll(); })
      .catch(function (err) { showError(String(err && err.message || err)); });
  }

  document.getElementById("approve").addEventListener("click", function () { respond(true); });
  document.getElementById("deny").addEventListener("click", function () { respond(false); });

  post({ type: "mcpapp/ready" });
})();
</script>
</body>
</html>
"""


TRACE_WATERFALL_RESOURCE_URI = "ui://graph-os/trace-waterfall.html"

# Self-contained, same CSP posture as task-progress (no external loads).
_TRACE_WATERFALL_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Trace Waterfall</title>
<style>
  body { font: 12px system-ui, sans-serif; margin: 12px; color: #1a1a1a; }
  #summary { margin-bottom: 10px; }
  #summary .stat { display: inline-block; margin-right: 14px; color: #444; }
  .row { display: flex; align-items: center; height: 20px; white-space: nowrap; }
  .label { width: 260px; overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; }
  .track { flex: 1; background: #f1f5f9; height: 12px; border-radius: 2px; position: relative; }
  .bar { height: 12px; border-radius: 2px; min-width: 2px; }
  .bar-tool { background: #60a5fa; }
  .bar-generation { background: #34d399; }
  .bar-retrieval { background: #a78bfa; }
  .bar-general { background: #94a3b8; }
  .bar-error { background: #f87171 !important; }
  .bar-root { background: #1a1a1a; }
  .dur { width: 70px; text-align: right; flex-shrink: 0; color: #555; padding-left: 6px; }
  #err { color: #991b1b; margin-top: 8px; }
  button { margin-bottom: 8px; }
</style>
</head>
<body>
  <button id="refresh">Refresh</button>
  <div id="summary"></div>
  <div id="rows"></div>
  <div id="err" hidden></div>
<script>
(function () {
  "use strict";
  // Bridge protocol (see mcp_apps.py / agent-webui bridge.ts docstrings):
  //   iframe -> host: {type: "mcpapp/ready"}
  //   host -> iframe: {type: "mcpapp/init", props: {traceId}}
  //   iframe -> host: {type: "mcpapp/tool-call", id, name, arguments}
  //   host -> iframe: {type: "mcpapp/tool-result", id, result}
  //                 | {type: "mcpapp/tool-error", id, error: {message}}
  var pending = Object.create(null);
  var seq = 0;
  var traceId = null;

  function post(message) {
    window.parent.postMessage(message, "*");
  }

  function callTool(name, args) {
    var id = "call-" + (++seq);
    return new Promise(function (resolve, reject) {
      pending[id] = { resolve: resolve, reject: reject };
      post({ type: "mcpapp/tool-call", id: id, name: name, arguments: args });
    });
  }

  window.addEventListener("message", function (event) {
    var data = event.data;
    if (!data || typeof data !== "object") return;
    if (data.type === "mcpapp/init") {
      traceId = data.props && data.props.traceId;
      load();
      return;
    }
    if (data.type === "mcpapp/tool-result" && pending[data.id]) {
      pending[data.id].resolve(data.result);
      delete pending[data.id];
      return;
    }
    if (data.type === "mcpapp/tool-error" && pending[data.id]) {
      pending[data.id].reject(new Error((data.error && data.error.message) || "tool error"));
      delete pending[data.id];
    }
  });

  function showError(message) {
    var el = document.getElementById("err");
    el.hidden = !message;
    el.textContent = message || "";
  }

  function fmtMs(ms) {
    if (ms === null || ms === undefined) return "-";
    return ms >= 1000 ? (ms / 1000).toFixed(2) + "s" : Math.round(ms) + "ms";
  }

  function addRow(container, label, pct, kindClass, ms, indent) {
    var row = document.createElement("div");
    row.className = "row";
    var lab = document.createElement("div");
    lab.className = "label";
    lab.style.paddingLeft = (indent * 14) + "px";
    lab.textContent = label;
    var track = document.createElement("div");
    track.className = "track";
    var bar = document.createElement("div");
    bar.className = "bar " + kindClass;
    bar.style.width = Math.max(2, pct) + "%";
    track.appendChild(bar);
    var dur = document.createElement("div");
    dur.className = "dur";
    dur.textContent = fmtMs(ms);
    row.appendChild(lab);
    row.appendChild(track);
    row.appendChild(dur);
    container.appendChild(row);
  }

  // Nested-DURATION waterfall (see engine_surface_tools.py::_trace_waterfall
  // docstring): no absolute start timestamps are recorded, so bars are sized by
  // each node's own latency relative to the trace total and nested by
  // parentId -- this shows where time/cost went, not concurrent wall-clock
  // overlap.
  function render(data) {
    var rowsEl = document.getElementById("rows");
    rowsEl.innerHTML = "";
    var trace = data.trace;
    var totalMs = trace.latencyMs || 1;
    var summary = document.getElementById("summary");
    summary.innerHTML =
      '<span class="stat"><b>' + (trace.name || trace.id) + "</b></span>" +
      '<span class="stat">status: ' + trace.status + "</span>" +
      '<span class="stat">' + fmtMs(trace.latencyMs) + "</span>" +
      '<span class="stat">$' + (trace.costUsd || 0).toFixed(4) + "</span>" +
      '<span class="stat">' + (trace.inputTokens || 0) + " in / " +
        (trace.outputTokens || 0) + " out tok</span>";
    addRow(rowsEl, trace.name || trace.id, 100, "bar-root", trace.latencyMs, 0);

    var byParent = Object.create(null);
    (data.nodes || []).forEach(function (n) {
      var pid = n.parentId || trace.id;
      (byParent[pid] = byParent[pid] || []).push(n);
    });

    function walk(parentId, depth) {
      (byParent[parentId] || []).forEach(function (n) {
        var pct = totalMs > 0 ? (n.latencyMs || 0) / totalMs * 100 : 0;
        var kindClass = "bar-" + (n.kind || "general");
        if (n.error) kindClass += " bar-error";
        var label = n.name || n.id;
        if (n.kind === "generation" && n.model) label += " (" + n.model + ")";
        addRow(rowsEl, label, pct, kindClass, n.latencyMs, depth);
        walk(n.id, depth + 1);
      });
    }
    walk(trace.id, 1);
  }

  function load() {
    if (!traceId) return;
    showError("");
    callTool("graph_traces", { action: "waterfall", trace_id: traceId })
      .then(function (result) {
        var parsed = typeof result === "string" ? JSON.parse(result) : result;
        if (parsed.error) { showError(parsed.error); return; }
        if (parsed.degraded) { showError(parsed.error || "trace sink unavailable"); return; }
        render(parsed.result || parsed);
      })
      .catch(function (err) { showError(String(err && err.message || err)); });
  }

  document.getElementById("refresh").addEventListener("click", load);

  post({ type: "mcpapp/ready" });
})();
</script>
</body>
</html>
"""


def register_mcp_apps_tools(mcp: Any) -> None:
    """Register the task-progress + trace-waterfall MCP Apps' entry-point tools
    and resources."""
    from fastmcp.apps.config import AppConfig, ResourceCSP, ResourcePermissions

    @mcp.tool(
        name="graph_task_progress_app",
        description=(
            "Launch a live task-progress MCP App for a durable job (an "
            "'orch-<id>' handle from graph_jobs dispatch). Polls status "
            "through the host-mediated graph_jobs tool and, if the task "
            "reaches input_required, lets the user approve or deny."
        ),
        tags=["graph-os", "jobs", "mcp-apps"],
        app=AppConfig(
            resource_uri=TASK_PROGRESS_RESOURCE_URI,
            visibility=["model"],
            csp=ResourceCSP(),
            permissions=ResourcePermissions(),
        ),
    )
    async def graph_task_progress_app(
        job_id: str = Field(description="Durable job handle from graph_jobs dispatch."),
    ) -> dict[str, Any]:
        return {"jobId": job_id}

    # Wrapped defensively, matching server_factory.py's own custom_route
    # registration: several unit tests build a server against a minimal
    # FastMCP test double that only implements `.tool()`/`.custom_route()`,
    # not `.resource()`. A real FastMCP instance always has `.resource()`;
    # only the (server-construction-only) test doubles take this path.
    resource = getattr(mcp, "resource", None)
    if callable(resource):

        @resource(
            uri=TASK_PROGRESS_RESOURCE_URI,
            name="Task Progress",
            description="Live task-progress viewer, polling graph_jobs status.",
            mime_type="text/html",
        )
        async def task_progress_resource() -> str:
            return _TASK_PROGRESS_HTML

    @mcp.tool(
        name="graph_trace_waterfall_app",
        description=(
            "Launch a trace-waterfall MCP App (D-25-1) for one trace id. Renders "
            "the trace's Span/Generation subgraph as a nested-duration waterfall "
            "(each node's own latency relative to the trace total, nested by "
            "parent span) through the host-mediated "
            "'graph_traces action=waterfall' tool -- the fastest way to see where "
            "a run's time and cost went."
        ),
        tags=["graph-os", "observability", "traces", "mcp-apps"],
        app=AppConfig(
            resource_uri=TRACE_WATERFALL_RESOURCE_URI,
            visibility=["model"],
            csp=ResourceCSP(),
            permissions=ResourcePermissions(),
        ),
    )
    async def graph_trace_waterfall_app(
        trace_id: str = Field(description="Trace id, e.g. from a RunTrace/ToolCall."),
    ) -> dict[str, Any]:
        return {"traceId": trace_id}

    if callable(resource):

        @resource(
            uri=TRACE_WATERFALL_RESOURCE_URI,
            name="Trace Waterfall",
            description=(
                "Nested-duration waterfall for one trace, polling "
                "graph_traces action=waterfall."
            ),
            mime_type="text/html",
        )
        async def trace_waterfall_resource() -> str:
            return _TRACE_WATERFALL_HTML
