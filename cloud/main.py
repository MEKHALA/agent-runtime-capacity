"""Amazon Bedrock AgentCore Runtime entry point (direct code deployment).

Implements the runtime's HTTP contract on 0.0.0.0:8080:
  GET  /ping         -> {"status": "Healthy"} or {"status": "HealthyBusy"}
  POST /invocations  -> JSON payload {"framework", "steps", "replicates"}, JSON result

Standard library only, so the server adds nothing to the dependency set being
measured. Pings are answered on a separate thread while a benchmark runs; they
are rare and cost microseconds, but they do land in Strands' process-CPU
windows, so the run records how many arrived.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import bench

_busy = threading.Event()
_pings = {"during_runs": 0}


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/ping"):
            if _busy.is_set():
                _pings["during_runs"] += 1
            self._reply(200, {"status": "HealthyBusy" if _busy.is_set() else "Healthy"})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/invocations"):
            self._reply(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(n) or b"{}")
        _busy.set()
        _pings["during_runs"] = 0
        try:
            out = bench.run(**payload)
            out["pings_during_run"] = _pings["during_runs"]
            self._reply(200, out)
        except Exception as e:  # report, do not crash the microVM
            self._reply(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            _busy.clear()

    def log_message(self, *_args):  # keep stdout quiet
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
