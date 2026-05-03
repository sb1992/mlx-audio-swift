#!/usr/bin/env python3
"""
Text encoder sidecar for MeanAudio.
Runs FLAN-T5-Large + LAION CLAP to produce text features for the MLX Swift pipeline.

Usage as CLI:
    python text_encoder_sidecar.py --text "A dog barking in a park" --output features.npz

Usage as HTTP server (for Larynx app integration):
    python text_encoder_sidecar.py --serve --port 8765

The server accepts POST /encode with JSON {"text": "..."} and returns
binary numpy arrays for text_features (1, 77, 1024) and text_features_c (1, 512).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("meanaudio-sidecar")


class TextEncoderSidecar:
    def __init__(self, clap_ckpt: str = None, device: str = "mps"):
        self.device = device

        log.info("Loading FLAN-T5-Large encoder...")
        from transformers import T5EncoderModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.t5 = T5EncoderModel.from_pretrained("google/flan-t5-large").eval().to(device)

        log.info("Loading LAION CLAP...")
        import laion_clap
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base").eval()
        if clap_ckpt:
            self.clap.load_ckpt(clap_ckpt, verbose=False)
        else:
            default_ckpt = "./weights/music_speech_audioset_epoch_15_esc_89.98.pt"
            if Path(default_ckpt).exists():
                self.clap.load_ckpt(default_ckpt, verbose=False)
            else:
                log.warning(f"CLAP checkpoint not found at {default_ckpt}. Using random weights.")

        log.info("Text encoders ready.")

    @torch.inference_mode()
    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Encode text to T5 features + CLAP embedding.

        Returns:
            text_features: (1, 77, 1024) T5 encoder output
            text_features_c: (1, 512) CLAP text embedding
        """
        # T5 encoding
        tokens = self.tokenizer(
            [text],
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        text_features = self.t5(input_ids=input_ids, attention_mask=attention_mask)[0]

        # CLAP encoding
        text_features_c = self.clap.get_text_embedding([text], use_tensor=True)

        return (
            text_features.cpu().numpy().astype(np.float32),
            text_features_c.cpu().numpy().astype(np.float32),
        )


def run_cli(args):
    sidecar = TextEncoderSidecar(clap_ckpt=args.clap_ckpt, device=args.device)
    text_f, text_fc = sidecar.encode(args.text)
    np.savez(args.output, text_features=text_f, text_features_c=text_fc)
    log.info(f"Saved features to {args.output}")
    log.info(f"  text_features: {text_f.shape}")
    log.info(f"  text_features_c: {text_fc.shape}")


def run_server(args):
    from http.server import HTTPServer, BaseHTTPRequestHandler

    sidecar = TextEncoderSidecar(clap_ckpt=args.clap_ckpt, device=args.device)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/encode":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            text = body.get("text", "")

            text_f, text_fc = sidecar.encode(text)

            import io
            buf = io.BytesIO()
            np.savez(buf, text_features=text_f, text_features_c=text_fc)
            data = buf.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            log.info(format % args)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    log.info(f"Serving on http://127.0.0.1:{args.port}")
    log.info("POST /encode with {\"text\": \"...\"} to get features")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MeanAudio text encoder sidecar")
    parser.add_argument("--device", default="mps", help="Device (mps, cuda, cpu)")
    parser.add_argument("--clap-ckpt", default=None, help="Path to CLAP checkpoint")

    sub = parser.add_subparsers(dest="command")

    cli_parser = sub.add_parser("encode", help="Encode text and save to file")
    cli_parser.add_argument("--text", required=True, help="Text prompt")
    cli_parser.add_argument("--output", default="features.npz", help="Output .npz path")

    server_parser = sub.add_parser("serve", help="Run HTTP server")
    server_parser.add_argument("--port", type=int, default=8765, help="Port")

    args = parser.parse_args()
    if args.command == "encode":
        run_cli(args)
    elif args.command == "serve":
        run_server(args)
    else:
        parser.print_help()
