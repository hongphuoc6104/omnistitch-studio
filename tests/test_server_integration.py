import http.client
import os
import tempfile
import threading
import unittest
import urllib.parse

from http.server import ThreadingHTTPServer

import server


class TestServerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(server.DATA_DIR, exist_ok=True)
        cls.temp_dir = tempfile.TemporaryDirectory(dir=server.DATA_DIR)
        cls.image_path = os.path.join(cls.temp_dir.name, "stream.bmp")
        cls.payload = b"BM" + (b"x" * (2 * server.FILE_STREAM_CHUNK_BYTES + 17))
        with open(cls.image_path, "wb") as stream:
            stream.write(cls.payload)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.AlignmentToolRequestHandler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(5)
        cls.temp_dir.cleanup()

    def _connection(self):
        return http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)

    def test_image_response_streams_allowed_file_without_cross_origin_cors(self):
        path = "/api/image?" + urllib.parse.urlencode({"path": self.image_path})
        connection = self._connection()
        connection.request("GET", path, headers={"Origin": "https://attacker.example"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIsNone(response.getheader("Access-Control-Allow-Origin"))
        self.assertEqual(int(response.getheader("Content-Length")), len(self.payload))
        self.assertEqual(response.read(), self.payload)
        connection.close()

    def test_same_origin_cors_and_workspace_source_denial(self):
        port = self.httpd.server_address[1]
        connection = self._connection()
        connection.request("GET", "/api/health", headers={"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}"})
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.getheader("Access-Control-Allow-Origin"), f"http://127.0.0.1:{port}")
        connection.close()

        denied = "/api/image?" + urllib.parse.urlencode({"path": os.path.join(server.WORKSPACE_DIR, "README.md")})
        connection = self._connection()
        connection.request("GET", denied)
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 403)
        connection.close()

    def test_oversized_json_request_is_rejected_before_body_read(self):
        connection = self._connection()
        connection.request(
            "POST",
            "/api/scan_folder",
            body=b"x",
            headers={"Content-Length": str(server.MAX_JSON_REQUEST_BYTES + 1)},
        )
        response = connection.getresponse()
        response.read()
        self.assertEqual(response.status, 413)
        connection.close()

    def test_dzi_and_image_variant_fallback(self):
        # 1. Tạo DZI folder legacy không suffix trong OUTPUTS_DIR
        legacy_dzi_dir = os.path.join(server.OUTPUTS_DIR, "test_legacy_slide_dzi")
        os.makedirs(legacy_dzi_dir, exist_ok=True)
        dzi_xml_path = os.path.join(legacy_dzi_dir, "test_legacy_slide.dzi")
        with open(dzi_xml_path, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?><Image TileSize="254" Overlap="1" Format="jpg"></Image>')

        try:
            # Request URL có suffix biến thể _01_goc -> Server phải tự fallback sang legacy DZI
            connection = self._connection()
            connection.request("GET", "/dzi/test_legacy_slide_01_goc_dzi/test_legacy_slide_01_goc.dzi")
            response = connection.getresponse()
            content = response.read()
            self.assertEqual(response.status, 200)
            self.assertIn(b"TileSize", content)
            connection.close()

            # 2. Tạo Image legacy không suffix
            legacy_img_path = os.path.join(server.OUTPUTS_DIR, "test_legacy_slide.tif")
            with open(legacy_img_path, "wb") as f:
                f.write(b"II*\x00\x08\x00\x00\x00")

            # Request URL ảnh có suffix _01_goc -> Server phải tự fallback sang legacy image
            connection = self._connection()
            rel_req = f"data/result/test_legacy_slide_01_goc.tif"
            connection.request("GET", "/api/image?" + urllib.parse.urlencode({"path": rel_req}))
            img_res = connection.getresponse()
            img_bytes = img_res.read()
            self.assertEqual(img_res.status, 200)
            self.assertEqual(img_bytes, b"II*\x00\x08\x00\x00\x00")
            connection.close()
            # 3. Tạo DZI có suffix _01_goc, request URL legacy không suffix -> Server tự fallback
            v_dzi_dir = os.path.join(server.OUTPUTS_DIR, "test_v_slide_01_goc_dzi")
            os.makedirs(v_dzi_dir, exist_ok=True)
            v_dzi_xml = os.path.join(v_dzi_dir, "test_v_slide_01_goc.dzi")
            with open(v_dzi_xml, "w", encoding="utf-8") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?><Image TileSize="254" Overlap="1" Format="jpg"></Image>')

            connection = self._connection()
            connection.request("GET", "/dzi/test_v_slide_dzi/test_v_slide.dzi")
            res_v = connection.getresponse()
            self.assertEqual(res_v.status, 200)
            res_v.read()
            connection.close()
        finally:
            if os.path.exists(dzi_xml_path):
                os.remove(dzi_xml_path)
            if os.path.exists(legacy_dzi_dir):
                os.rmdir(legacy_dzi_dir)
            legacy_img_path = os.path.join(server.OUTPUTS_DIR, "test_legacy_slide.tif")
            if os.path.exists(legacy_img_path):
                os.remove(legacy_img_path)
            v_dzi_xml = os.path.join(server.OUTPUTS_DIR, "test_v_slide_01_goc_dzi", "test_v_slide_01_goc.dzi")
            if os.path.exists(v_dzi_xml):
                os.remove(v_dzi_xml)
            v_dzi_dir = os.path.join(server.OUTPUTS_DIR, "test_v_slide_01_goc_dzi")
            if os.path.exists(v_dzi_dir):
                os.rmdir(v_dzi_dir)


if __name__ == "__main__":
    unittest.main()
