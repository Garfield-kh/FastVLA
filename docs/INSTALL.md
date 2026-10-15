# Installation

The model and the simulators need different Python environments; they talk over a local TCP
socket.

**Model environment** (training, bank building, policy server):

```bash
conda create -n fastvla python=3.11 -y && conda activate fastvla
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/dexmal/opendm.git && cd opendm && git checkout e41e501 && pip install -e . && cd ..
pip install -r requirements.txt
```

Optionally `pip install flash-attn --no-build-isolation` for DM0.5's vision tower; without it
the vision tower falls back to SDPA (same maths; the released checkpoints were checked both ways).

`opendm` provides the DM0.5 model class (`transformers==5.3.0`). The DM0.5 weights are
downloaded from the Hugging Face Hub on first use (`Dexmal/DM05-robotwin2`,
`Dexmal/DM05-libero`; revisions are pinned in `configs/*.yaml` and in each checkpoint's
`config.json`).

**Simulator environments**: install [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)
and/or [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (and the
[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) fork for LIBERO-Plus) following their
instructions. Nothing from this repository has to be installed there; the evaluation clients
only import `fastvla/client.py`.

## Ascend NPU

The policy server also runs on Ascend NPUs: the model environment builds on `torch_npu` instead of the
CUDA wheels. Use torch 2.10.0, **not** 2.11.0 — the aarch64 wheel of 2.11.0 is a CUDA build and
raises `libcublasLt.so.*[0-9] not found` on import.

```bash
conda create -n fastvla python=3.11 -y && conda activate fastvla
pip install torch==2.10.0 torch_npu==2.10.0 torchvision==0.25.0
git clone https://github.com/dexmal/opendm.git && cd opendm && git checkout e41e501 \
  && pip install -e . --no-deps && cd ..        # dm05_arch only needs torch + transformers
pip install transformers==5.3.0 -r requirements.txt
```

Then pass `--device npu:0` to `scripts/serve.py`. The simulator side is unchanged; on a host with
no GPU graphics stack the benchmarks have to render through OSMesa (`MUJOCO_GL=osmesa`,
`PYOPENGL_PLATFORM=osmesa`, set before `robosuite` is imported).

On Ascend three attention kernels are swapped for equivalent ones:

| | CUDA | Ascend |
|---|---|---|
| language tower | `flex_attention` | `sdpa` — FlexAttention compiles through Inductor, which needs Triton |
| vision tower | `flash_attention_2` | `sdpa` — `flash_attn` is CUDA-only |
| task-memory cross-attention mask | broadcast over the query dim | materialised to `[B, 1, Sq, Skv]` — the fused kernel rejects the broadcast |

The installation check below passes on Ascend (`--device npu:0`, `max|diff|` = 3.9e-3).

To check an installation against the released checkpoints, see [Checking an installation](../README.md#checking-an-installation).
