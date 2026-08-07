# Design Specification — Glys Engine RUL Estimation

| | |
|---|---|
| **Date** | 2026-08-06 |
| **Status** | Approved |
| **Course** | Industrial Computing (DIBSE 26), part 2 — Deep Learning project |
| **Upstream brief** | `mckoh/industrial_computing_dibse_26/documents/project_exercise.md` |
| **Language policy** | Code, docstrings, commit messages, `README.md` → **English**. Graded report (`BERICHT.md`) → **German**. |

---

## 1. Context

The Glys commission a predictive-maintenance system for spaceship engines. Given a
thermal-camera image of an engine, estimate its **remaining useful life (RUL)** in
operating hours.

Grading rubric from the brief:

| Weight | Requirement |
|---|---|
| 25 % | Feature extraction |
| 10 % | Reproducible environment |
| 15 % | Suitable network architecture |
| 15 % | A second, alternative architecture (explicitly *may* be non-convolutional) |
| 15 % | Training settings and training run |
| 20 % | Analysis of estimation accuracy |

The brief permits either a regression or a classification formulation.

---

## 2. Dataset audit — verified findings

These findings were established empirically before design and are the foundation for
every decision below. They must be **re-derived at runtime** by the pipeline, never
hard-coded.

### 2.1 The 11 files are 6 unique images

| Files | md5 (prefix) |
|---|---|
| `003h.jpeg` = `005h.jpeg` | `87558f…` |
| `024h.jpeg` = `026h.jpeg` | `4dac2b…` |
| `047h.jpeg` = `051h.jpeg` | `8b2bf8…` |
| `073h.jpeg` = `076h.jpeg` | `bc54a4…` |
| `078h.jpeg` = `082h.jpeg` | `1b3259…` |
| `100h.jpeg` | `c2ee1e…` |

Five pairs are **byte-identical** yet carry different labels. Any split that separates
files rather than hashes leaks the test sample into training.

### 2.2 Irreducible error floor

Identical pixels with differing labels bound achievable accuracy. Predicting each
group's mean:

- **MAE floor = 15 / 11 = 1.364 h**
- **RMSE floor = √(24.5 / 11) = 1.492 h**

Any reported error below these values is memorisation or leakage, not skill. Both
floors are drawn on every accuracy plot.

### 2.3 Image structure

Synthetic 1920×1080 vector renders on white: an intake **cone**, a main **body**, and a
**pylon**. Geometry is *identical* across all images; only colour varies.
Connected-component labelling yields exactly 3 components in all 6 images with zero
noise blobs, ordered left-to-right as cone / body / pylon.

### 2.4 The temperature ladder

Region colours mapped through the calibrated scale:

Measured through the calibrated pipeline (`colorscale.ColorScale` + `segment.regions`):

| RUL | cone | body | pylon | Σ °C |
|---|---|---|---|---|
| 3 / 5 h | 827 | 1194 | 1194 | 3215 |
| 24 / 26 h | 657 | 827 | 1194 | 2679 |
| 47 / 51 h | 657 | 827 | 827 | 2312 |
| 73 / 76 h | 0 | 657 | 827 | 1485 |
| 78 / 82 h | 0 | 0 | 657 | 658 |
| 100 h | 0 | 0 | 0 | 1 |

The dataset uses **only four distinct temperatures — {0.33, 657.33, 827.36, 1193.81} °C**.
The thermal state is a 3-symbol word over a 4-letter alphabet; 6 of 64 possible states are
observed. Heat drains front-to-back, pylon last.

Values are quantised by the LUT (3685 entries over 1200 °C ≈ 0.33 °C per step), so the
cold end reads 0.33 °C rather than exactly zero. These are the pipeline's own numbers —
the report must quote `reports/results.json`, never a figure derived by hand.

`Σ °C` is **strictly decreasing across the six distinct thermal states**, and merely
non-increasing across the eleven files — byte-identical pairs necessarily share a value.
That distinction is the reason splits must be grouped by content hash: at sample level the
relationship is not injective, and treating the eleven files as eleven independent
observations overstates the evidence by nearly a factor of two.

Three scalars therefore carry 100 % of the available information.

### 2.5 Three traps this creates

1. **Alpha channel.** `temp.png` is RGBA. A naive `.convert("RGB")` renders transparent
   pixels **black**, silently corrupting the colour scale and every temperature derived
   from it. All image loading must composite over white first.
2. **Luminance is not monotone in temperature.** It peaks at **189 @ 825 °C** and falls
   to **124 @ 1200 °C**; hue disambiguates the hot end. Consequently grayscale
   conversion destroys the signal, and "brighter = hotter" is *wrong* above 825 °C.
3. **Region area is a leakage-adjacent artefact.** Areas vary < 2 % across images
   (cone 143 463–144 864, body 412 828–415 532, pylon 58 547–59 728), but the variation
   *correlates with colour* because the near-white threshold treats anti-aliased edges
   of dark and bright regions differently. Area is a JPEG artefact, not physics, and is
   dropped from the feature set with this justification recorded.

### 2.6 Why the CNN is expected to lose

Geometry is fixed; only the colour of three static masks changes. There is no spatial
pattern for a convolutional prior to exploit. The expectation that a ~250-parameter MLP
outperforms a CNN is a *prediction of this design*, to be confirmed or refuted
empirically — and it is exactly the alternative-architecture idea the brief floats.

---

## 3. Goals and non-goals

**Goals**

- Reach the best honest accuracy the data supports, and report it against the provable
  floor rather than in place of it.

  **Measured correction to an earlier assumption.** This design originally predicted that
  isotonic regression on Σ°C would land essentially *on* the 1.364 h floor. That holds
  in-sample (measured 1.364 h) but fails under grouped cross-validation, where the same
  model scores **11.68 h** — 8.6× worse. The gap is structural, not a modelling defect:
  with six distinct states and leave-one-group-out, each fold trains on five states and
  must predict a sixth. Interior states are interpolation (MAE 8.31 h); the two endpoints
  are extrapolation beyond anything seen, and isotonic clips to its training range, so a
  held-out 3 h reads 25 h and a held-out 100 h reads 80 h (MAE 20.67 h across those folds).

  The floor therefore bounds what is achievable *given the labels*; the endpoint geometry
  bounds what is achievable *given six states*. Both belong in the report, and the honest
  number is the cross-validated one.
- Make every methodological choice traceable to a dataset property established in §2.
- One-command reproduction, verified continuously rather than asserted.
- Document the full optimisation campaign including dead ends.

**Non-goals**

- Synthesising additional training data. Only the 11 supplied files are used
  (augmentation is transformation of those files, not fabrication).
- Supporting arbitrary thermal-imagery formats. Swappability is limited to the format
  described in §6.
- Progress tracking, branch/PR ceremony, or experiment-tracking servers.

---

## 4. Repository layout

```
.
├── README.md                    English: setup, contract, results, metric glossary
├── BERICHT.md                   German: the graded report, ordered by rubric
├── pyproject.toml · uv.lock     exact pins, hash-locked
├── Dockerfile · compose.yaml
├── Makefile                     convenience wrappers over compose
├── .github/workflows/ci.yml
├── data/raw/triebwersbilder/    the 11 files + temp.png, vendored
│   └── MANIFEST.json            source URL + md5 per file, verified at runtime
├── src/glys_rul/
│   ├── config.py                paths, seeds, scale range, hyperparameters
│   ├── io.py                    alpha-safe loading, manifest verification
│   ├── colorscale.py            bar detection → LUT, rgb↔°C, invertibility assert
│   ├── segment.py               foreground mask → ordered region masks
│   ├── features.py              image → feature vector + feature table
│   ├── audit.py                 dedup, grouping, error floors, contract report
│   ├── dataset.py               loading, grouped splits, in-fold augmentation
│   ├── models.py                CNN, MLP, monotone MLP, transfer arm
│   ├── train.py                 nested grouped CV, seed sweep
│   ├── baselines.py             the baseline ladder
│   ├── conformal.py             jackknife+ intervals
│   ├── explain.py               occlusion saliency, permutation importance
│   ├── counterfactual.py        maintenance thresholds
│   ├── evaluate.py              metrics, figures, results.json
│   ├── search.py                Optuna campaign → experiments.csv
│   ├── export.py                weights → JSON for the web demo
│   └── cli.py                   subcommands: validate · reproduce · search
├── docs/design/                 this document
├── web/                         GitHub Pages demo (static, no build step)
├── tests/
└── reports/                     committed: results.json, experiments.csv, figures/
```

There are no notebooks. `BERICHT.md` is the single narrative and all logic lives in
`src/`, driven through `cli.py`, so nothing is defined twice and every figure in the
report is regenerated by the pipeline rather than pasted.

---

## 5. Component design

Each module has one purpose, a narrow interface, and is testable in isolation.

### `io.py`
`load_rgb(path) -> np.ndarray[H,W,3] float` — opens any mode, composites RGBA/LA/P over
opaque white, returns float RGB. The single entry point for reading pixels; nothing else
in the codebase calls PIL directly.
`verify_manifest(dir) -> None` — raises if a vendored file's md5 does not match.

### `colorscale.py`
`ColorScale.from_image(path, vmin, vmax)` — detects the gradient bar's bounding box,
averages its interior rows into an N×3 LUT, and **asserts invertibility**: every bar
sample's nearest neighbour must lie within 50 °C of its true value. Measured on the
supplied scale: 3685 LUT entries, 0 ambiguous samples, max round-trip error 12.7 °C.
`to_celsius(rgb) -> float` · `to_map(image) -> np.ndarray[H,W] float` (°C per pixel).

Nearest-neighbour lookup in RGB is sufficient given the invertibility assertion; the
assertion is what makes that safe rather than assumed.

### `segment.py`
`regions(image) -> dict[str, np.ndarray]` — foreground = pixels with mean channel value
below the near-white threshold (default 245), `scipy.ndimage.label`, components under
5 000 px discarded as noise, survivors
sorted by bounding-box x-start and named `[cone, body, pylon]`. Raises if the component
count differs from the configured expectation. Masks are **eroded 4 px** before sampling
to exclude JPEG ringing at edges.

### `features.py`
Per region: **median °C** (median, not mean — robust to residual edge artefacts).
Global: Σ °C, max °C, hot-fraction > 800 °C, cone→body→pylon gradient, and a 16-bin
histogram over engine pixels. Area and thermal-energy features are computed but flagged
degenerate per §2.5 and excluded from modelling, with the exclusion recorded in the
feature report rather than silently applied.

### `audit.py`
Produces the contract report: file count, md5 groups, label range and parse results,
components found per image, colour-scale round-trip error, **derived** MAE/RMSE floors,
and zero-variance feature detection. Runs first in every pipeline invocation and fails
loudly on contract violation.

### `dataset.py`
Groups by md5. Split strategy adapts: LOGO while groups ≤ 10, `GroupKFold(5)` above.
Augmentation is applied **inside the training fold only**.

**Legal augmentation:** ±8 % translation, ±10 % scale, ±5° rotation, additive Gaussian
sensor noise in °C space, small global calibration bias.

**Forbidden augmentation:** brightness, contrast, hue, gamma, grayscale. Colour *is* the
label; photometric jitter silently relabels the sample. Horizontal flip is off by
default (it permutes cone↔pylon semantics) and retained only as a robustness probe.

---

## 6. Data contract and swappability

Swapping datasets is supported for the supplied format only. No descriptor files.

**Contract**

- A directory of images named `<hours>h.{jpeg,jpg,png}` — filename encodes RUL in hours.
- A colour-scale reference image; range configured in `config.py` (default 0–1200 °C).
- Engine renders decomposing into the configured number of connected components.

**Mechanism** — `compose.yaml` mounts the repository's own data by default:

```yaml
volumes:
  # Default: the dataset vendored in this repo.
  # To run on your own data, change the left-hand path to your folder.
  - ./data/raw/triebwersbilder:/data:ro
```

**Consequence.** Because data is swappable, nothing derived from it may be a constant.
Error floors, group counts, fold counts, bin edges and feature variances are all computed
at runtime. `make validate` runs the audit alone and prints the contract report.

---

## 7. Reproducibility

### 7.1 Execution

```bash
docker compose up                        # full reproduction, ~6 min (measured 5m37s)
docker compose --profile search up       # optimisation campaign, ~40 min
docker compose --profile lab up          # Jupyter
uv sync && make reproduce                # non-Docker fallback, same lockfile
```

The fallback is deliberate: if the grader cannot or will not run Docker, the 10 %
criterion must not be forfeited.

### 7.2 Determinism stack

| Control | Eliminates |
|---|---|
| `platform: linux/amd64` | ISA differences (Apple Silicon via emulation) |
| base image pinned by **digest**, deps `--require-hashes` | dependency drift |
| `TF_ENABLE_ONEDNN_OPTS=0` | oneDNN dispatching different kernels on AVX2 vs AVX-512 hosts |
| `CUDA_VISIBLE_DEVICES=""` | nondeterministic cuDNN kernels |
| `intra_op = inter_op = 1` | float reduction order varying with thread count |
| `keras.utils.set_random_seed()` + `enable_op_determinism()` | init / shuffle / dropout ordering |
| `PYTHONHASHSEED=0` | set and dict iteration order |
| `user: "${UID:-1000}:${GID:-1000}"` | root-owned output files on the host |

CPU-only costs nothing here: ~25 k parameters over 11 samples runs in seconds.

**Claim discipline.** The determinism claim is *verified*, not assumed, and stated at
the precision it was tested: identical on the platforms exercised (local + CI amd64).

### 7.3 Continuous proof

Outputs are split so that reproduction is machine-checkable:

- `reports/results.json` — metrics only, deterministic, committed as the golden copy.
- `reports/run_meta.json` — timestamps, library versions, git SHA. Not diffed.

CI on every push runs the container and then:

```bash
git diff --exit-code reports/results.json
```

A green badge is proof of reproducibility for a grader who never clones the repo. CI also
uploads `reports/` as a downloadable artifact.

---

## 8. Modelling

**Formulation.** Regression is primary: `Dense(1, activation="linear")` with `mse` loss,
exactly as the brief describes, target scaled to RUL/100. The same predictions are
additionally binned into 10 h classes to produce a confusion matrix, satisfying the
classification hint without training a classifier on ~1 sample per class.

### Architectures

| | **A — CNN** | **B — Feature MLP** |
|---|---|---|
| input | 128×128 calibrated temperature map (1 channel, °C/1200) | 3 region medians (+ optional derived scalars) |
| body | 3 × [Conv(16→32→64) · BN · ReLU · MaxPool] → GAP → Dense(32) | Dense(16) → Dense(8) |
| head | `Dense(1, linear)` | `Dense(1, linear)` |
| params | ~25 k | ~250 |

Capacity is deliberately small and justified by an explicit parameter-count versus
sample-count argument. Two further arms come from the search campaign: a
**monotone-constrained MLP** (non-negative weights on a negated temperature input,
guaranteeing RUL decreases monotonically with heat) and a **frozen MobileNetV2** transfer
arm connecting to Megatutorial 3.

### Training protocol

- **Nested grouped CV.** Outer LOGO over the 6 hash groups; hyperparameter search runs in
  an inner loop over the remaining 5. The outer fold never informs tuning.
- **Reported honestly.** Both the nested (honest) and best-tuned (optimistic) scores are
  published side by side; the gap between them is itself a finding about selection bias
  at n = 11.
- **5 seeds × 6 folds = 30 runs** per configuration. At n = 11 a single-seed number is
  noise; results are reported as mean ± std.
- Adam, full-batch, fixed epoch budget chosen a priori. **No early stopping** — with one
  held-out group there is no honest validation signal, and this is stated rather than
  worked around.

---

## 9. Optimisation campaign

Runs off the critical path under `--profile search`; every trial is appended to
`reports/experiments.csv` and committed.

| Axis | Values |
|---|---|
| Architecture | Optuna ~100 trials each: depth, width, lr, epochs, L2, dropout |
| Input representation | raw RGB · temperature map · 16-bin histogram · 3 scalars; 32/64/128/224 px |
| Loss | MSE · Huber · log-target |
| Augmentation | off · geometric · geometric + sensor noise, three strengths |
| Model family | CNN · MLP · monotone MLP · frozen MobileNetV2 · isotonic |
| Post-hoc | seed/fold ensembling · test-time augmentation |

The ledger records **what failed and why**. A table containing only successes reads as
cherry-picking; the dead ends are the evidence that the search was real.

---

## 10. Evaluation

**Baseline ladder** — each rung answers a specific objection:

| # | Model | Question it settles |
|---|---|---|
| 0 | predict the mean | what does *no* skill look like? |
| 1 | 1-NN on Σ °C | is this just table lookup? |
| 2 | linear regression on Σ °C | does one parameter suffice? |
| 3 | isotonic regression on Σ °C | does exploiting monotonicity close the gap? |
| 4 | Arch B (MLP) | does a learned model beat the closed form? |
| 5 | Arch A (CNN) | does convolution buy anything? |

**Metrics** — MAE, RMSE, R², and a **skill score** `1 − MAE_model / MAE_mean` so a reader
sees instantly whether a model beat the trivial baseline. Both error floors are drawn on
every plot.

**Diagnostics** — predicted-vs-actual with y = x; residuals vs Σ °C; 10 h-bin confusion
matrix; learning curve over k = 2…5 training groups; permutation importance;
**shuffled-label control** (train on permuted labels: training loss still → 0 while CV
error collapses to baseline, *measuring* memorisation capacity rather than asserting it).

**Honest limitations section** — the model has seen six states, interpolates nothing
between them, is undefined beyond 100 h, and depends entirely on colour calibration
holding.

---

## 11. Add-ons

### 11.1 Interactive demo (GitHub Pages)
`web/` — a static page with three temperature sliders that render an engine in the
dataset's exact visual style on a canvas and predict RUL live. The ~250-parameter MLP is
exported to JSON and evaluated in ~20 lines of hand-written JavaScript: no backend, no
TensorFlow.js dependency, no Docker required of the grader. A CNN lane via
`tfjs-converter` is a stretch goal, not a dependency.

### 11.2 Conformal prediction intervals
Jackknife+ over the grouped folds yields `RUL ∈ [a, b]` at a target coverage instead of a
bare point estimate. Correct practice at n = 11, beyond course scope, ~50 lines, and it
renders as the demo's uncertainty band.

### 11.3 Counterfactual maintenance thresholds
"To gain 20 more operating hours, pylon temperature must fall below X °C." Falls out of
the monotone model almost free and answers the question the Glys actually asked — turning
a regressor into decision support.

### 11.4 Occlusion saliency
Occlusion-based attribution showing whether the CNN attends to the pylon on high-RUL
samples or merely memorises. One figure, materially strengthening §10.

---

## 12. Deliverables

| Artifact | Language | Notes |
|---|---|---|
| `README.md` | English | Setup, data contract, results table, and a closing section defining every score: what it means, how it is computed, and why the floor is the ceiling. Links to `BERICHT.md` at the top. |
| `BERICHT.md` | German | The graded write-up, ordered by rubric item, embedding the committed figures. Sole narrative document. |
| `reports/` | — | `results.json`, `experiments.csv`, all figures — committed, so the repo is readable without execution. |
| `web/` | English | Deployed to GitHub Pages. |

The README carries the rubric as a checklist linking to the module and report section
satisfying each item.

---

## 13. Git strategy

Standalone **public** repository `Cl3mi/master_deep-learning`, initialised in place at
`sem_2/industrial_computing/project/`; the surrounding coursework monorepo ignores that
path via its local `.git/info/exclude`, so neither history is polluted. Public
visibility satisfies the brief without adding members and enables free GitHub Pages
hosting for the demo. The remote is created and pushed once the work is coherent, not
incrementally.

Conventional Commits, straight to `main`, no branches or PRs. Every commit leaves the
pipeline green. `.gitignore` present from commit 1 — no cleanup commits in the history.
Final commit tagged `v1.0-submission` so later work cannot muddy what was graded.

Planned sequence:

```
docs: add design specification                     ← commit 1
chore: initialize project structure and tooling
chore(docker): reproducible CPU-only container and compose entrypoint
feat(data): vendor Glys dataset with provenance manifest
feat(audit): detect duplicate images, derive irreducible error floors
feat(colorscale): calibrate temperature LUT with invertibility check
feat(segment): extract engine regions via connected components
feat(features): build physical feature table
feat(dataset): grouped splits and label-preserving augmentation
feat(models): CNN and feature-MLP architectures
feat(train): nested grouped cross-validation with seed sweep
feat(eval): baseline ladder, metrics and figures
feat(conformal): jackknife+ prediction intervals
feat(explain): occlusion saliency and permutation importance
feat(counterfactual): maintenance threshold derivation
feat(search): Optuna optimisation campaign
feat(web): interactive engine demo for GitHub Pages
test: assert deterministic end-to-end reproduction
ci: run containerized pipeline and diff golden results
docs: findings, results and rubric mapping
chore(release): tag submission
```

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| Colour-scale inversion ambiguous on a swapped dataset | `ColorScale` asserts invertibility at load; fails loudly |
| Swapped dataset yields ≠ 3 components | `segment.regions` raises with a diagnostic; `make validate` catches it first |
| Cross-microarchitecture float divergence | oneDNN off, single-threaded, platform pinned; claim stated only at tested precision |
| TensorFlow wheel availability on arm64 | `platform: linux/amd64` forces emulation; workload is seconds |
| Search campaign exceeds its time budget | Off the critical path behind a profile; trial count is configurable |
| CNN fails to beat baselines | This is a *predicted* result (§2.6), reported as a finding with evidence |

---

## 15. Acceptance criteria

1. `docker compose up` on a clean checkout reproduces `reports/results.json`
   byte-identically; CI enforces this on every push.
2. Repointing the compose mount at a conforming folder runs end-to-end, and a
   non-conforming folder fails with an actionable contract error — never silently
   produces results from mismatched input.
3. Error floors are computed from the data at runtime, never hard-coded.
4. No train/test split separates two files sharing an md5.
5. Both architectures train, and the baseline ladder is reported with floors on every plot.
6. `experiments.csv` contains the full campaign including failed configurations.
7. The best model's honest nested-CV MAE is reported against the 1.364 h floor, with the
   optimistic tuned score shown alongside.
8. README closes with a metric glossary explaining each score and its derivation.
9. `BERICHT.md` covers all six rubric items in German.
10. The demo page loads from GitHub Pages and predicts without a backend.
