"""Training-time image augmentation (GPU): aspect-preserving random crop + random erasing.

Only the DINOv2 inputs are augmented; the VLM cache is precomputed and never is.
Frames of the same camera at different times share one crop / erasing so history
frames do not get fake motion; different cameras are augmented independently.
"""
import torch
import torch.nn.functional as F

MILD = dict(crop_scale_min=0.85, erasing_p=0.25, erasing_scale_min=0.01, erasing_scale_max=0.05)


def _crop_resize(x, cfg):
    """x: [B, K, 3, H, W]; one crop per sample, shared by its K frames."""
    B, K, _, H, W = x.shape
    scale = torch.empty(B, device=x.device).uniform_(cfg["crop_scale_min"], 1.0)
    # Height and width are both scaled by sqrt(scale): the crop keeps the aspect ratio.
    ch = (torch.sqrt(scale) * H).long().clamp(8, H)
    cw = (torch.sqrt(scale) * W).long().clamp(8, W)
    out = torch.empty_like(x)
    for i in range(B):
        sh, sw = int(ch[i]), int(cw[i])
        top = int(torch.randint(0, H - sh + 1, (1,)))
        left = int(torch.randint(0, W - sw + 1, (1,)))
        out[i] = F.interpolate(x[i, :, :, top:top + sh, left:left + sw], size=(H, W),
                               mode="bilinear", align_corners=False)
    return out


def _erase(x, cfg):
    B, _, _, H, W = x.shape
    do = torch.rand(B, device=x.device) < cfg["erasing_p"]
    for i in torch.nonzero(do).flatten().tolist():
        a = float(torch.empty(1).uniform_(cfg["erasing_scale_min"], cfg["erasing_scale_max"]))
        eh = max(1, int((a * H * W) ** 0.5))
        ew = max(1, int(a * H * W / eh))
        top = int(torch.randint(0, max(1, H - eh), (1,)))
        left = int(torch.randint(0, max(1, W - ew), (1,)))
        x[i, :, :, top:top + eh, left:left + ew] = torch.rand(1, device=x.device)
    return x


def _aug(x, cfg):
    return _erase(_crop_resize(x, cfg), cfg)


def aug_views(imgs, cam_of_slot, cfg=MILD):
    """imgs: [B, n_slot, 3, H, W] in [0, 1]."""
    B, N = imgs.shape[:2]
    if len(set(cam_of_slot)) == N:            # one slot per camera
        return _aug(imgs.flatten(0, 1)[:, None], cfg)[:, 0].view(B, N, *imgs.shape[2:])
    out = imgs.clone()
    for c in sorted(set(cam_of_slot)):
        idx = [i for i, cc in enumerate(cam_of_slot) if cc == c]
        out[:, idx] = _aug(imgs[:, idx], cfg)
    return out
