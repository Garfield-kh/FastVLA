#!/usr/bin/env python
"""Serve a FastVLA model to a simulator process.

    python scripts/serve.py --model checkpoints/fastvla-robotwin --port 8765

The simulator side uses ``fastvla/client.py`` (stdlib + numpy only), so the
benchmark and the model can live in different Python environments.
"""
import argparse
import os
import socket
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastvla.client import recv_msg, send_msg  # noqa: E402
from fastvla.policy import FastVLAPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="released model directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    pol = FastVLAPolicy(a.model, a.device)
    info = dict(model=os.path.abspath(a.model), benchmark=pol.bench, chunk=pol.chunk,
                hist_offsets=pol.hist_offsets, n_view=pol.n_view)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(8)
    print(f"[serve] {info} listening on {a.host}:{a.port}", flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            while True:
                req = recv_msg(conn)
                op = req.get("op")
                if op == "info":
                    send_msg(conn, info)
                elif op == "reset":
                    pol.reset()
                    send_msg(conn, {"ok": True})
                elif op == "act":
                    a_ = pol.act(req["views"], np.asarray(req["state"], np.float32),
                                 req["instruction"], req.get("hist_views"))
                    send_msg(conn, {"action": a_})
                else:
                    raise ValueError(f"unknown op {op!r}")
        except (ConnectionError, EOFError):
            pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
