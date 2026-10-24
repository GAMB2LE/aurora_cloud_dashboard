import unittest
from unittest.mock import patch

import request_context


class _Request:
    path = "/app"
    query_arguments = {b"tab": [b"power"], "live": ["1"]}
    remote_ip = "127.0.0.1"


class _Session:
    id = "session-1"
    request = _Request()
    server_context = type("Server", (), {"sessions": [1, 2]})()


class _Document:
    session_context = _Session()


class RequestContextTests(unittest.TestCase):
    @patch.object(request_context, "_document", return_value=_Document())
    def test_request_values_are_normalized(self, _document):
        self.assertEqual(request_context.session_id(), "session-1")
        self.assertEqual(request_context.request_path(), "/app")
        self.assertEqual(request_context.request_query_args(), {"tab": "power", "live": "1"})
        self.assertEqual(request_context.client_ip(), "127.0.0.1")
        self.assertEqual(request_context.server_session_count(), 2)

    @patch.object(request_context, "_document", return_value=None)
    def test_import_smoke_context_is_empty(self, _document):
        self.assertIsNone(request_context.session_id())
        self.assertEqual(request_context.request_query_args(), {})

    @patch.object(request_context, "request_header")
    @patch.object(request_context, "request_path")
    def test_share_base_uses_page_path_after_websocket_callback(self, path, header):
        header.side_effect = {"X-Forwarded-Proto": "https", "Host": "data.gamb2le.co.uk"}.get
        for request_path, page_path in (
            ("/app/ws", "/app"),
            ("/custom/prefix/app/ws", "/custom/prefix/app"),
            ("/ws", "/"),
            ("/prefix/ws/ws", "/prefix/ws"),
        ):
            with self.subTest(request_path=request_path):
                path.return_value = request_path
                self.assertEqual(request_context.request_base_url(), f"https://data.gamb2le.co.uk{page_path}")

    @patch.object(request_context, "request_header")
    @patch.object(request_context, "request_path")
    def test_share_base_preserves_other_paths_and_host_port(self, path, header):
        header.side_effect = {"X-Forwarded-Proto": "http", "Host": "localhost:5006"}.get
        for request_path in ("/app", "/custom/app/", "/news", "/ws/app", "/app/ws/", "/"):
            with self.subTest(request_path=request_path):
                path.return_value = request_path
                self.assertEqual(request_context.request_base_url(), f"http://localhost:5006{request_path}")

    @patch.object(request_context, "request_header", return_value=None)
    @patch.object(request_context, "request_path", return_value=None)
    def test_share_base_keeps_import_smoke_defaults(self, _path, _header):
        self.assertEqual(request_context.request_base_url(), "http://127.0.0.1:5006/app")


if __name__ == "__main__":
    unittest.main()
