"""Policy-server protocol: length-prefixed pickle over TCP.

This file only needs the standard library and numpy, so the simulator environment
(RoboTwin / LIBERO) can import it without installing torch or FastVLA.

Requests:  {"op": "info"} | {"op": "reset"} |
           {"op": "act", "views": [...], "state": [...], "instruction": str,
            "hist_views": {offset: [...]} or None}
Replies:   info -> config summary incl. "hist_offsets"; reset -> {"ok": True};
           act  -> {"action": np.ndarray [chunk, action_dim]}
"""
import pickle
import socket
import struct


def send_msg(sk, obj):
    b = pickle.dumps(obj, protocol=4)
    sk.sendall(struct.pack(">I", len(b)) + b)


def recv_msg(sk):
    def _n(n):
        buf = b""
        while len(buf) < n:                 # recv may return fewer bytes than asked
            c = sk.recv(n - len(buf))
            if not c:
                raise ConnectionError("connection closed")
            buf += c
        return buf
    return pickle.loads(_n(struct.unpack(">I", _n(4))[0]))


class PolicyClient:
    def __init__(self, host="127.0.0.1", port=8765):
        self.sk = socket.create_connection((host, port))
        self.info = self.call({"op": "info"})

    def call(self, msg):
        send_msg(self.sk, msg)
        return recv_msg(self.sk)

    def reset(self):
        self.call({"op": "reset"})

    def act(self, views, state, instruction, hist_views=None):
        return self.call({"op": "act", "views": views, "state": state,
                          "instruction": instruction, "hist_views": hist_views})["action"]
