"""The frozen VLM side: DM0.5 prefill, RoPE lookup tables, cache assembly.

The same ``prefill`` is used to build the offline training bank and to compute
the first-frame cache online at evaluation time, so the two are identical by
construction (up to bf16 batch-composition noise).
"""
import importlib.util
import os
import warnings

import numpy as np
import torch

IMAGE_SOFT_TOKEN_ID = 262144      # Gemma-3 image soft token in DM0.5's tokenizer
VLM_TOKENS_PER_IMAGE = 256

# Prompt layout of DM0.5 (opendm/data/transforms.py, ChatTokenization). The trailing
# period after the instruction is part of the original template.
BENCH_PROMPT = {
    "robotwin": dict(robot="Aloha RoboTwin2", speed="0.5",
                     images=("Head", "Left wrist", "Right wrist")),
    "libero": dict(robot="Franka", speed="0.5", images=("Head", "Left wrist")),
}

DEFAULT_VLM = {
    "robotwin": ("Dexmal/DM05-robotwin2", "bef09062f8d9744b7691d0a5db4e23834e46071e"),
    "libero": ("Dexmal/DM05-libero", "25a8e0d38a8eaeaae41a44d7b4a2378fd8ce1088"),
}


def build_prompt(bench, instruction):
    p = BENCH_PROMPT[bench]
    return (f"Robot: {p['robot']}\n"
            f"Overall speed: {p['speed']}\n"
            f"Task: {instruction}.\n")


def load_vlm(repo_id, revision=None, device="cuda", dtype=torch.bfloat16):
    """Loads DM0.5 (requires the ``opendm`` package) and its processor."""
    from opendm.model.dm05.dm05_arch import DM05ForConditionalGeneration
    from transformers import AutoProcessor
    kw = {}
    if importlib.util.find_spec("flash_attn") is None:
        # The DM0.5 config pins flash_attention_2 for the vision tower, but flash_attn is an
        # optional opendm dependency; without it the vision tower uses SDPA (same maths).
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(repo_id, revision=revision)
        cfg.vlm_config.vision_config._attn_implementation = "sdpa"
        kw["config"] = cfg
        if torch.device(device).type == "cuda":
            warnings.warn("flash_attn is not installed; the DM0.5 vision tower uses SDPA")
    m = DM05ForConditionalGeneration.from_pretrained(
        repo_id, revision=revision, torch_dtype=dtype, **kw)
    if torch.device(device).type == "npu":
        # The released DM0.5 config pins flex_attention for the language tower (it compiles
        # through Inductor, which needs Triton) and flash_attention_2 for the vision tower
        # (CUDA-only). Neither exists on Ascend; SDPA is the same maths. The action expert
        # keeps the "eager" backend it sets in its own __init__.
        m.set_attention_implementation(llm_attn_implementation="sdpa",
                                       vision_attn_implementation="sdpa",
                                       action_attn_implementation="eager")
    return m.to(device).eval(), AutoProcessor.from_pretrained(repo_id, revision=revision)


@torch.no_grad()
def prefill(dm05, processor, bench, instructions, images_per_ep, device="cuda"):
    """First-frame prefill of N episodes.

    images_per_ep: for every episode, one raw uint8 HWC image per camera, in the
    order of ``BENCH_PROMPT[bench]["images"]``. DM0.5's own processor resizes them.
    Returns (keys, values, meta): keys/values are tuples over layers of
    [N, kv_heads, L, head_dim]; meta has attention_mask, input_ids and
    effective_prefix_len (number of non-pad prefix tokens).
    """
    from PIL import Image
    labels = BENCH_PROMPT[bench]["images"]
    msgs = []
    for instr, ims in zip(instructions, images_per_ep, strict=True):
        uc = [{"type": "text", "text": build_prompt(bench, instr)}]
        for label, im in zip(labels, ims, strict=True):
            text = f"{label} image: "
            if uc[-1]["type"] == "text":
                uc[-1]["text"] += text
            else:
                uc.append({"type": "text", "text": text})
            uc.append({"type": "image", "image": Image.fromarray(np.asarray(im, dtype=np.uint8))})
        msgs.append([{"role": "user", "content": uc}])

    inputs = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
    kv, prefix_len = dm05._compute_prefix_cache(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        pixel_values=inputs.get("pixel_values"),
        token_type_ids=inputs.get("token_type_ids"))
    keys, values = dm05._extract_prefix_cache_tensors(kv)
    pad_id = dm05.model.vlm.model.language_model.padding_idx
    valid = inputs["input_ids"][:, :prefix_len] != pad_id
    meta = dict(prefix_len=int(prefix_len), effective_prefix_len=valid.sum(dim=1),
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return keys, values, meta


# ----------------------------------------------------------------------------- RoPE
class RopeTables:
    """Per-layer (cos, sin) of the DM0.5 action expert, exported once from the model.

    Reading the cache needs the VLM RoPE at the prefix and query positions; keeping
    the tables avoids loading the 4B VLM during training. Always export them from the
    model (``dump_rope_tables``) instead of re-deriving the frequencies: Gemma-3
    alternates two bases across layers.
    """

    def __init__(self, layer_types, tables, head_dim, max_pos):
        self.layer_types = list(layer_types)
        self.tables = tables
        self.head_dim, self.max_pos = head_dim, max_pos

    @classmethod
    def load(cls, path, device=None):
        d = torch.load(path, map_location="cpu", weights_only=False)
        t = {k: (v[0].to(device) if device else v[0], v[1].to(device) if device else v[1])
             for k, v in d["tables"].items()}
        return cls(d["layer_types"], t, d["head_dim"], d["max_pos"])

    def to(self, device):
        self.tables = {k: (a.to(device), b.to(device)) for k, (a, b) in self.tables.items()}
        return self

    def per_layer(self, position_ids, dtype=torch.float32):
        """position_ids [B, N] -> [(cos, sin)] * n_layers, each [B, N, head_dim]."""
        assert 0 <= int(position_ids.min()) and int(position_ids.max()) < self.max_pos
        return [(self.tables[t][0][position_ids].to(dtype),
                 self.tables[t][1][position_ids].to(dtype)) for t in self.layer_types]


@torch.no_grad()
def dump_rope_tables(repo_id, out_path, revision=None, device="cuda", max_pos=2048):
    """Exports the action expert's RoPE tables (model loaded in fp32)."""
    dm05, _ = load_vlm(repo_id, revision, device, dtype=torch.float32)
    ae = dm05.model.action_expert
    hd = ae.layers[0].self_attn.head_dim
    layer_types = [l.attention_type for l in ae.layers]
    pos = torch.arange(max_pos, device=device).unsqueeze(0)
    dummy = torch.zeros(1, 1, max_pos, hd, device=device, dtype=torch.float32)
    tables = {}
    for t in sorted(set(layer_types)):
        cos, sin = ae.rotary_emb(dummy, pos, layer_type=t)
        tables[t] = (cos[0].float().cpu().contiguous(), sin[0].float().cpu().contiguous())
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(dict(layer_types=layer_types, tables=tables, head_dim=int(hd),
                    max_pos=int(max_pos), source=repo_id), out_path)


# --------------------------------------------------------------- cache assembly
def build_prefix_position_ids(attn_mask):
    """Absolute positions of the prefix tokens, as in DM0.5's prefill (pads take none)."""
    return (attn_mask.long().cumsum(dim=-1) - 1).clamp_min(0)


def build_query_position_ids(eff_len, chunk, n_arms, device):
    """Queries continue right after the prefix; both arms at step t share position eff+t."""
    t = torch.arange(chunk, device=device)
    if n_arms > 1:
        t = t.repeat_interleave(n_arms)
    return eff_len.to(device).unsqueeze(1) + t.unsqueeze(0)


def cache_kwargs(keys, values, attn_mask, eff_len, rope, chunk, n_arms, input_ids=None,
                 n_image_tokens=None, dtype=torch.float32):
    """Everything the model's forward needs about the cache.

    keys/values: tuples over layers of [B, kv_heads, L, head_dim];
    attn_mask/input_ids: [B, L]; eff_len: [B]. With ``input_ids`` the segment ids
    (text=0 / image=1) are added and the image-token count is checked.
    """
    device = attn_mask.device
    assert keys[0].shape[2] == attn_mask.shape[1], "cache length != attention-mask length"
    out = dict(keys=keys, values=values, kv_pad=attn_mask.bool(),
               prefix_rope=rope.per_layer(build_prefix_position_ids(attn_mask), dtype),
               query_rope=rope.per_layer(build_query_position_ids(eff_len, chunk, n_arms,
                                                                  device), dtype))
    if input_ids is not None:
        seg = (input_ids[:, :attn_mask.shape[1]].to(device) == IMAGE_SOFT_TOKEN_ID).long()
        if n_image_tokens is not None:
            n = (seg * attn_mask.long()).sum(1)
            assert int(n.min()) == int(n.max()) == n_image_tokens, (
                f"expected {n_image_tokens} image tokens per episode, got "
                f"[{int(n.min())}, {int(n.max())}]")
        out["kv_seg"] = seg
    return out
