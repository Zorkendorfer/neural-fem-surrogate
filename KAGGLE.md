# Training & benchmarking FieldNet on Kaggle (free GPU)

Kaggle gives free NVIDIA GPUs (P100 / T4×2, ~30 GPU-hours/week). Two reasons to
use it here:

1. **FNO training is much faster on CUDA.** Locally the FNO runs on CPU because
   `torch.fft` is unsupported on Apple MPS. CUDA supports it — expect ~10–30×
   faster epochs.
2. **It makes the speedup headline real.** Running `scripts/bench.py` on CUDA
   times the FNO *on GPU*, which is the honest path to the ≥1000× / ~10,000×
   speedup figure (locally the FNO can only be timed on CPU).

It will **not** change accuracy — same model, same maths. Use Kaggle for speed
and for the GPU benchmark numbers, not to chase a better error.

The repo is already GPU-ready: `get_device()` auto-detects CUDA and every
script accepts `--device cuda`.

---

## Step 1 — Upload the dataset as a Kaggle Dataset

`data/dataset.npz` (~373 MB) is gitignored, so cloning the repo will not bring
it. Upload it once:

1. Go to <https://www.kaggle.com/datasets> → **New Dataset**.
2. Drag in your local `data/dataset.npz`.
3. Title it e.g. **`fieldnet-dataset`** → **Create**.

(Alternative: regenerate it on Kaggle — see the last section. Uploading is
faster and reproduces your exact local data.)

## Step 2 — Create the notebook

1. <https://www.kaggle.com/code> → **New Notebook**.
2. Right-hand panel → **Settings**:
   - **Accelerator** → **`GPU T4 x2`**. *Not `P100`* — its GPU architecture
     (`sm_60`) is too old for Kaggle's current PyTorch build.
   - **Internet** → **On** (required for `git clone` / `pip`).
3. **Input** → **Add Input** → search for your `fieldnet-dataset` → add it.
   It mounts at `/kaggle/input/fieldnet-dataset/dataset.npz`.

## Step 3 — Run these cells

**Cell 1 — clone the repo and install the package**

```python
!git clone https://github.com/Zorkendorfer/neural-fem-surrogate.git
%cd neural-fem-surrogate
!pip install -q -e .
```

**Cell 2 — confirm the GPU and locate the dataset**

```python
import torch, glob
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "Set Accelerator -> GPU in Settings"

# auto-discover the uploaded dataset (Kaggle's mount folder name varies)
print("datasets under /kaggle/input:", glob.glob("/kaggle/input/*"))
npz = glob.glob("/kaggle/input/**/*.npz", recursive=True)
print("npz files found:", npz)
assert npz, "No .npz found -- click 'Add Input' and attach your dataset"
DATASET = npz[0]
print("DATASET =", DATASET)
```

**Cell 3 — train the FNO on the GPU**

```python
!python scripts/train.py --model fno --dataset {DATASET} --device cuda
```

**Cell 4 — train the DeepONet on the GPU** (600 epochs — it needs the longer schedule)

```python
!python scripts/train.py --model deeponet --dataset {DATASET} --device cuda --epochs 600
```

**Optional — use both T4 GPUs (replaces Cells 3 + 4).** The two models are
independent jobs, so pin one to each GPU and train them concurrently; total
time becomes `max(FNO, DeepONet)` instead of the sum. Use *two* cells so the
launch does not block the kernel.

*Launch cell* — starts both trainings and returns immediately. Their output is
redirected to log files, so this cell itself prints almost nothing:

```python
import os, subprocess
def launch(gpu, args, log):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(["python", "scripts/train.py", *args],
                            env=env, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT)

p0 = launch(0, ["--model", "fno", "--dataset", DATASET, "--device", "cuda"],
            "fno.log")
p1 = launch(1, ["--model", "deeponet", "--dataset", DATASET, "--device", "cuda",
                "--epochs", "600"], "deeponet.log")
print("launched both -- run the monitor cell to watch progress")
```

*Monitor cell* — re-run it whenever you want a progress check:

```python
for name, p, log in [("FNO", p0, "fno.log"), ("DeepONet", p1, "deeponet.log")]:
    state = "running" if p.poll() is None else f"finished (exit {p.returncode})"
    print(f"=== {name}: {state} ===")
    !tail -n 6 {log}
```

Wait until both report `finished (exit 0)` before running the benchmark.
Training a *single* model across both GPUs would need `DataParallel`/DDP — not
worth it here (the models are small and train fast on one GPU).

**Cell 5 — run the benchmark on the GPU**

```python
!python scripts/bench.py --dataset {DATASET} --device cuda --batch-size 64
```

This times the FNO *and* DeepONet on CUDA — both single-forward latency and
batched throughput — so `benchmark/results.md` now carries the real GPU
speedup. (Raise `--batch-size 128` for a larger throughput number if VRAM
allows.)

**Cell 6 — (optional) run the test suite**

```python
!pip install -q pytest && python -m pytest -q
```

**Cell 7 — package the outputs for download**

```python
!cd /kaggle/working/neural-fem-surrogate && \
 zip -r /kaggle/working/fieldnet_outputs.zip checkpoints benchmark
print("download /kaggle/working/fieldnet_outputs.zip from the Output panel")
```

## Step 4 — Download and commit locally

1. In the notebook, open the right-hand **Output** panel → download
   `fieldnet_outputs.zip` (or **Save Version** to persist it).
2. Unzip it into your local repo root, overwriting `checkpoints/` and
   `benchmark/`.
3. Commit:
   - `benchmark/` (figures + tables) **is** tracked → it goes into the repo and
     feeds the README.
   - `checkpoints/` **is** gitignored (model weights are large) → keep it
     locally for the Phase 6 demo; it is not committed.

---

## Notes & troubleshooting

- **Session limits:** ~9 h interactive / ~12 h on commit — far more than needed.
- **`CUDA available: False`:** Accelerator wasn't set to GPU, or the session
  needs a restart after changing Settings.
- **`CUDA error: no kernel image is available for execution on the device`:**
  you were given a P100, which is too old for Kaggle's PyTorch build. Switch
  the Accelerator to `GPU T4 x2` and re-run from Cell 1.
- **`git clone` fails:** Internet toggle is off.
- **Doubled path (`.../neural-fem-surrogate/neural-fem-surrogate`):** Cell 1 was
  run twice. Re-run it only once per session (after a restart).
- **Live demo (Phase 6):** Hugging Face Spaces' free tier is CPU-only, so the
  *deployed* demo runs on CPU regardless (DeepONet ~11 ms, FNO ~52 ms — both
  under the 150 ms budget). Kaggle is for training + benchmark numbers only.

## Alternative — regenerate the dataset on Kaggle

If you skip Step 1, regenerate it instead of Cell 2's dataset path:

```python
!python scripts/generate.py --workers 4         # ~15-30 min on Kaggle's CPUs
DATASET = "data/dataset.npz"
```

Reproducible (fixed seed) — it yields the identical dataset, but spends session
time on CPU-bound FEM solves.
