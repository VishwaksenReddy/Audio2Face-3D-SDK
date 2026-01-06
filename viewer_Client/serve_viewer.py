from __future__ import annotations

import argparse
import json
import struct
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ingest.audio_pipeline import process_wav_bytes_to_pcm16_chunks


def pack_processed_audio(sample_rate: int, chunks: list[tuple[int, bytes]]) -> bytes:
    out = bytearray()
    out += b"A2PC"  # Audio2Face Processed Chunks
    out += struct.pack("<III", 1, int(sample_rate), len(chunks))
    for start_sample_index, pcm16le in chunks:
        if (len(pcm16le) % 2) != 0:
            raise ValueError("PCM16 chunk must have an even byte length")
        sample_count = len(pcm16le) // 2
        out += struct.pack("<qI", int(start_sample_index), int(sample_count))
        out += pcm16le
    return bytes(out)


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/process_audio":
            self.send_error(404, "Not found")
            return

        content_len = int(self.headers.get("Content-Length", "0"))
        if content_len <= 0:
            self.send_error(400, "Missing request body")
            return
        if content_len > 128 * 1024 * 1024:
            self.send_error(413, "Audio file too large")
            return

        wav_bytes = self.rfile.read(content_len)
        qs = parse_qs(parsed.query)
        target_sr = int(qs.get("sr", ["16000"])[0])
        chunk_s = float(qs.get("chunk_s", ["1.0"])[0])

        try:
            sample_rate, chunks = process_wav_bytes_to_pcm16_chunks(
                wav_bytes, target_sample_rate=target_sr, chunk_duration_s=chunk_s
            )
            payload = pack_processed_audio(sample_rate, chunks)
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the three.js viewer + local audio processing endpoint.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    def handler(*handler_args, **handler_kwargs):
        return Handler(*handler_args, directory=str(root), **handler_kwargs)

    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving viewer on http://{args.host}:{args.port}/")
    print("Audio processing endpoint: POST /api/process_audio (WAV bytes)")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

