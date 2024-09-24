# cors_server.py
from http.server import SimpleHTTPRequestHandler, HTTPServer

# run curl -L -o decoder_model.onnx https://huggingface.co/openai-community/gpt2/resolve/main/onnx/decoder_model.onnx to download the model locally


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


if __name__ == "__main__":
    httpd = HTTPServer(("localhost", 8000), CORSRequestHandler)
    print("Serving on http://localhost:8000")
    httpd.serve_forever()
