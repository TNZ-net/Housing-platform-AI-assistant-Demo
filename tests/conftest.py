import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest

@pytest.fixture
def fake_server():
    seen, reply = [], {"status": 200, "body": {}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers["Content-Length"])
            seen.append({"path": self.path, "body": json.loads(self.rfile.read(size)),
                         "auth": self.headers.get("Authorization")})
            data = json.dumps(reply["body"]).encode()
            self.send_response(reply["status"])
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # keep the test output quiet
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)  # port 0 = pick any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", seen, reply
    server.shutdown()