"""Per-camera DINOv2 encoders for the current (and history) frames."""
import torch
from torch import nn

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class MultiViewDino(nn.Module):
    """One DINOv2 per camera; frames of the same camera at different times share it.

    Input ``imgs`` is [B, n_slot, 3, H, W] in [0, 1]; slot s comes from camera
    ``cam_of_slot[s]``. The first ``n_view`` slots must be the current frame of
    cameras 0..n_view-1 in order. Returns (patch tokens incl. CLS [B, n_slot, L, v],
    CLS [B, n_slot, v]).
    """

    def __init__(self, n_view, backbone="facebook/dinov2-base", cam_of_slot=None,
                 pretrained=True):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        self.n_view = n_view
        self.cam_of_slot = tuple(range(n_view) if cam_of_slot is None else cam_of_slot)
        assert tuple(self.cam_of_slot[:n_view]) == tuple(range(n_view))
        self.n_slot = len(self.cam_of_slot)
        if pretrained:
            nets = [AutoModel.from_pretrained(backbone) for _ in range(n_view)]
        else:  # weights come from a FastVLA checkpoint; skip the download of DINOv2 weights
            cfg = AutoConfig.from_pretrained(backbone)
            nets = [AutoModel.from_config(cfg) for _ in range(n_view)]
        self.nets = nn.ModuleList(nets)
        self.dim = self.nets[0].config.hidden_size
        for net in self.nets:
            # mask_token never receives a gradient in a plain forward pass and would
            # make DDP's reducer wait forever.
            if hasattr(net.embeddings, "mask_token"):
                net.embeddings.mask_token.requires_grad_(False)
        self.register_buffer("mean", IMAGENET_MEAN, persistent=False)
        self.register_buffer("std", IMAGENET_STD, persistent=False)

    def forward(self, imgs):
        B, N = imgs.shape[:2]
        assert N == self.n_slot, f"got {N} image slots, model expects {self.n_slot}"
        x = (imgs - self.mean) / self.std
        # Group the slots by camera so each encoder runs one batched forward.
        slots = [[] for _ in range(self.n_view)]
        for s, c in enumerate(self.cam_of_slot):
            slots[c].append(s)
        out = [None] * N
        for c, ss in enumerate(slots):
            g = self.nets[c](pixel_values=x[:, ss].flatten(0, 1)).last_hidden_state
            g = g.view(B, len(ss), *g.shape[1:])
            for j, s in enumerate(ss):
                out[s] = g[:, j]
        h = torch.stack(out, 1)
        return h, h[:, :, 0]
