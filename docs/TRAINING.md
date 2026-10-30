# Training Fast-VLA

Run everything in the model environment described in the [README](../README.md#installation).

## 1. Data

```bash
# RoboTwin 2.0: the aloha-agilex demos from https://huggingface.co/datasets/TianxingChen/RoboTwin2.0
# (50 x demo_clean and 500 x demo_randomized per task, unzipped)
python scripts/prepare_data.py robotwin --clean-root <clean> --rand-root <randomized> --out data/robotwin
# LIBERO: the official hdf5 demos (run where `libero` and `h5py` are installed)
python scripts/prepare_data.py libero --hdf5-dir <LIBERO>/libero/datasets --out data/libero
```

## 2. First-frame cache bank

The VLM is frozen, so its first-frame cache is computed once for every training episode.

```bash
# RoboTwin: 22 shards x 1,250 episodes, fp8, ~1.6 TiB in total. One GPU per shard.
for s in $(seq 0 21); do
  CUDA_VISIBLE_DEVICES=$((s % 8)) python scripts/build_bank.py --config configs/robotwin.yaml --shard $s &
done; wait
python scripts/build_bank.py --config configs/robotwin.yaml --check
# LIBERO: 2 shards x 1,000 episodes, bf16, ~150 GiB
for s in 0 1; do python scripts/build_bank.py --config configs/libero.yaml --shard $s; done
python scripts/build_bank.py --config configs/libero.yaml --check
```

Each training episode is paired with one fixed instruction: for LIBERO the task instruction,
for RoboTwin the paraphrase listed in `assets/robotwin_instructions.json`, which is the one
encoded in the bank of the released model. During training the bank is memory-mapped; keeping
it in the page cache (i.e. enough RAM) matters for speed.

## 3. Train

```bash
# RoboTwin (released checkpoint: 2 devices x batch 128, cosine schedule)
torchrun --nproc_per_node=2 scripts/train.py --config configs/robotwin.yaml --out runs/robotwin
# LIBERO (released checkpoint: 1 device, batch 32, 30k steps)
python scripts/train.py --config configs/libero.yaml --out runs/libero
```

Checkpoints are saved every `save_every` steps in the release format and can be served
directly. Config keys can be overridden with `--set key=value`.

With `torch_npu` installed, `build_bank.py` and `train.py` run on Ascend NPUs without changes
(see [INSTALL.md](INSTALL.md#ascend-npu)).

## Notes on data

- **State and action on RoboTwin** are the same 14-D joint-target vector (as in DM0.5 and the
  RoboTwin baselines), z-score normalised. LIBERO uses the 8-D end-effector state and 7-D
  actions, quantile-normalised to [-1, 1].
- **Images.** RoboTwin JPEG frames are decoded with `cv2.imdecode` and used as-is (they were
  encoded from RGB arrays with `cv2.imencode`; decoding them with PIL swaps red and blue).
  LIBERO frames are used in the orientation stored in the demonstrations; the evaluation
  client passes simulator frames in the same orientation.
- **Determinism.** The data stream is a function of (seed, step), so resuming and multi-GPU
  runs see the same batches; GPU kernels are not deterministic, so two runs differ slightly.
