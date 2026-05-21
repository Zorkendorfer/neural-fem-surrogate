# SPEC.md — FieldNet: A Neural-Operator Surrogate for Structural FEA

> Working repo name: `neural-fem-surrogate`. Display name: **FieldNet**. Rename freely.
> This file is the source of truth for the build. Claude Code: read this fully before writing code, and re-read the relevant phase before starting it.

---

## 0. Why this project exists (do not lose sight of this)

This is a **portfolio weapon engineered to clear the hiring bar for a senior/staff Scientific ML role** (AI-for-science, physical-AI, simulation, robotics). It is not a research paper and not a toy. Every decision serves one of three hiring signals:

1. **Modern SciML competence** — neural *operators* (FNO + DeepONet), not PINNs.
2. **Engineering rigor** — a benchmark against ground-truth FEM proving accuracy *and* a large wall-clock speedup.
3. **Production deployment** — a live, interactive, deployed demo. This is the single thing most ML applicants can't show. **It is non-negotiable.**

**Definition of success:** a stranger lands on the repo README, sees a GIF of a live demo where dragging a load updates a full stress field in real time, sees a benchmark plot showing ~10,000× speedup over FEM at <3% field error, and within 60 seconds thinks "this person can build and ship physics ML." Then they click a link and use the live demo themselves.

---

## 1. Non-negotiables

- The finished repo MUST contain a **deployed, publicly reachable interactive demo** (URL in README).
- The finished repo MUST contain a **benchmark** comparing the surrogate to the FEM solver on (a) accuracy, (b) inference latency / speedup, (c) generalization to geometries unseen in training.
- Implement the **FNO spectral convolution from scratch** (torch.fft). Importing a black-box FNO defeats the hiring signal. The `neuraloperator` library may be used only as a correctness cross-check.
- Everything reproducible from a single command + a config file. Fixed seeds. No notebooks as the source of truth (notebooks allowed only for exploration in `/notebooks`).

---

## 2. The problem (physics)

**Parametric 2D linear elasticity: a rectangular plate with a circular hole under far-field tension.** Classic, well-understood (Kirsch solution exists for the infinite-plate limit → free analytical sanity check), visually compelling (stress concentration at the hole), and rich enough to demonstrate operator generalization.

**Learn the operator** `G: θ → field`, where:

- **Inputs `θ`** (parameter space, sampled via Sobol/Latin Hypercube):
  - hole radius `r` ∈ [0.05, 0.30] (plate half-width = 1.0)
  - far-field load magnitude `σ∞` ∈ [1, 10] MPa
  - load angle `α` ∈ [0, 90]° (uniaxial direction)
  - (optional v2) Young's modulus `E`, Poisson ratio `ν`
- **Output fields** on the plate domain:
  - displacement `(u_x, u_y)`
  - **von Mises stress** `σ_vm` ← the engineering money metric (stress concentration factor at the hole edge)

**Generalization test that matters:** hold out an entire region of geometry space (e.g. all `r > 0.25`) from training. The headline claim is "generalizes to geometries it never saw," so this OOD split must exist from day one.

---

## 3. Tech stack & constraints

- Python 3.11, **PyTorch** (CUDA if available, must also run CPU).
- **FEM data generator: `scikit-fem`** (pure-Python, pip-installable, fully reproducible). It is the sole ground-truth source. Alternative: FEniCSx (heavier; only if scikit-fem proves limiting).
- `numpy`, `scipy`, `einops`, `matplotlib`. Config via `pydantic` + YAML.
- Experiment tracking: **Weights & Biases** (fall back to TensorBoard if no account).
- Serving: **FastAPI** + **Docker** + minimal HTML/JS frontend (canvas heatmap + sliders). Gradio allowed only as a v0 throwaway to prove the loop, not as the final demo.
- Deploy target: **Hugging Face Spaces (Docker SDK)** or Fly.io / Render. Must be free or near-free.

---

## 4. System design (one paragraph)

Offline: a parametric FEM solver generates thousands of `(θ, field)` pairs on a fixed reference grid. Two neural operators learn `θ → field`: an **FNO** (operates on a uniform grid; the hole is encoded as a signed-distance-function input channel + a domain mask) and a **DeepONet** (branch net consumes `θ`, trunk net consumes query coordinates; handles the irregular domain natively). Both are trained, benchmarked head-to-head against FEM and each other, and the winner is wrapped in a FastAPI inference service behind an interactive web demo.

---

## 5. Repo structure

```
neural-fem-surrogate/
├── SPEC.md                  # this file
├── README.md                # the portfolio centerpiece (Phase 7)
├── pyproject.toml
├── configs/
│   ├── data.yaml            # parameter ranges, grid res, n_samples, splits
│   ├── fno.yaml
│   └── deeponet.yaml
├── src/fieldnet/
│   ├── fem/                 # scikit-fem data generator + Kirsch sanity check
│   ├── data/                # sampling, normalization, SDF/mask, Dataset/Loader
│   ├── models/
│   │   ├── fno.py           # spectral conv FROM SCRATCH
│   │   └── deeponet.py
│   ├── train.py
│   ├── eval/                # metrics, benchmark, plots
│   └── serve/               # FastAPI app + static frontend + Dockerfile
├── scripts/                 # one-command entrypoints (generate, train, bench, serve)
├── tests/
└── notebooks/               # exploration only
```

---

## 6. Build plan (phased — each phase ships something verifiable)

### Phase 0 — Scaffold
- Goal: runnable skeleton, reproducibility wiring.
- Tasks: repo structure, `pyproject.toml`, config loading (pydantic+YAML), global seed util, logging, `scripts/` entrypoints as stubs, CI that runs `pytest`.
- **Acceptance:** `pip install -e .` works; `pytest` runs (even if trivial); configs load.

### Phase 1 — FEM data generator
- Goal: produce ground-truth `(θ, field)` dataset.
- Tasks: scikit-fem mesh of plate-with-hole parametrized by `r`; linear-elastic solve under far-field tension at angle `α`; extract `u_x,u_y,σ_vm`; interpolate to a fixed `128×128` reference grid; compute SDF + mask channels; Sobol-sample `θ` (start `n=2000`, scale to `5000`); write to disk (`.npz`/`zarr`). Implement an analytical **Kirsch-solution check** for the large-plate, small-hole case.
- **Acceptance:** dataset of ≥2000 samples on disk; Kirsch check passes (<5% error vs analytic at the hole edge for the valid regime); a script renders 3 random samples as field plots.

### Phase 2 — Data pipeline
- Goal: clean train-time data path with the OOD split baked in.
- Tasks: per-channel normalization (store stats); PyTorch `Dataset`/`DataLoader`; **splits**: train / val / test + a separate **OOD-geometry holdout** (`r>0.25`); FNO view (grid tensors + SDF channel) and DeepONet view (param vector + coordinate samples).
- **Acceptance:** loaders yield correctly shaped, normalized batches for both model views; OOD set shares zero geometry overlap with train (assert in a test).

### Phase 3 — Models
- Goal: both operators implemented cleanly and configurable.
- Tasks: **FNO** with spectral convolution implemented from scratch via `torch.fft.rfft2/irfft2` (configurable modes, width, layers); **DeepONet** (branch MLP over `θ`, trunk MLP over coords, dot-product head). Shape/forward unit tests.
- **Acceptance:** both models forward-pass on a real batch and overfit a single sample to ~0 loss (sanity test in `tests/`).

### Phase 4 — Training
- Goal: trained checkpoints for both models.
- Tasks: training loop, **relative L2 field loss** (primary), optional **physics-consistency term** (penalize the elasticity equilibrium residual / traction-free hole boundary — a current, impressive add; gate behind a config flag), W&B logging, checkpointing, LR schedule, early stopping.
- **Acceptance:** both models train to **val relative L2 < 5%** on `σ_vm`; training is resumable from checkpoint; W&B run logged.

### Phase 5 — Evaluation & THE benchmark (the money phase)
- Goal: the evidence that gets him hired.
- Tasks/metrics:
  - Accuracy: relative L2 on `σ_vm` and displacement (in-distribution + OOD).
  - **Engineering metric:** error in peak stress / stress-concentration factor at the hole edge.
  - **Speedup:** wall-clock of one surrogate forward pass vs one full FEM solve (report mean ± std over 100 runs, CPU and GPU).
  - **Generalization:** in-dist vs OOD error gap, with field plots side-by-side (FEM truth | surrogate | error map).
  - FNO vs DeepONet head-to-head table.
- **Acceptance:** a `benchmark/` output containing a results table + the three headline figures (accuracy bars, speedup bar/log-scale, truth-vs-pred-vs-error triptych). Targets: ≥1000× speedup; OOD `σ_vm` rel L2 < 8%.

### Phase 6 — Deployment (non-negotiable)
- Goal: a live interactive demo on the public internet.
- Tasks: FastAPI service loading the best checkpoint, `/predict` endpoint (`θ` → field arrays as JSON or PNG); minimal frontend: sliders for `r`, `σ∞`, `α`, a canvas rendering the `σ_vm` heatmap that updates on input (debounced); Dockerfile; deploy to HF Spaces / Fly.io; latency budget < 150 ms per inference.
- **Acceptance:** a public URL works from a phone; dragging a slider updates the stress field in near-real-time; README links to it.

### Phase 7 — README & narrative (this *is* the portfolio)
- Goal: a recruiter/engineer gets the value in 60 seconds.
- Tasks: top-of-README **GIF of the live demo**; one-line pitch; the speedup + accuracy numbers above the fold; a short "how it works" with one architecture diagram; the FNO-from-scratch implementation called out as the key flex; reproducibility instructions; live demo link; a 2-paragraph blog-style writeup of one genuine technical insight (e.g. why FNO beat/lost to DeepONet here, or what the physics-residual term changed).
- **Acceptance:** README leads with a working demo GIF + headline numbers; full reproduce-from-scratch instructions verified by a clean clone.

---

## 7. Scope guards (do NOT do these in v1)
- No 3D. No nonlinear material. No transient/dynamics. No mesh adaptivity.
- No PINN. No exotic architectures (Geo-FNO, GNO, transformers) until v1 ships.
- Don't gold-plate the FEM solver — it's a data factory, not the product.
- Don't skip deployment to "do more modeling." Deployment > another model.

## 8. Stretch goals (only after v1 is deployed)
- Physics-informed operator loss as the headline (PDE residual, traction-free boundary).
- Reskin to an **employer-targeted domain** for tailored applications: additive-manufacturing melt-pool fields, or structural-dynamics frequency response — same machinery, hotter framing.
- Geometry generalization via Geo-FNO; multi-hole / arbitrary geometry via DeepONet + SDF.
- Active learning: use surrogate uncertainty to choose the next FEM solves.

## 9. Notes for Claude Code
- Work phase by phase; do not start a phase until the previous phase's acceptance criteria pass.
- Write the test for each phase's acceptance criterion, then make it pass.
- After each phase, print a short status: what shipped, how to verify it manually, what's next.
- Prefer small, reviewable commits per task. Keep functions pure and testable; keep config out of code.
- If a default here is impractical when you hit it, flag the tradeoff and propose an alternative rather than silently diverging.
