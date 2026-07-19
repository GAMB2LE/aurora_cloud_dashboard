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


if __name__ == "__main__":
    unittest.main()
