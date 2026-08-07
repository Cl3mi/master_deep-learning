# Glys Engine — Remaining Useful Life Estimation

Estimating how many operating hours a spaceship engine has left, from one thermal image.

**[Interactive demo](https://cl3mi.github.io/master_deep-learning/)** ·
**[Bericht (Deutsch)](BERICHT.md)** ·
**[Design spec](docs/design/2026-08-06-glys-rul-design.md)**

## The finding that shapes everything

The dataset has **11 files but only 6 unique images** — five byte-identical pairs carrying
different labels (`003h`=`005h`, `024h`=`026h`, `047h`=`051h`, `073h`=`076h`, `078h`=`082h`).

| Consequence | Why |
|---|---|
| Splits group by **content hash**, never filename | Holding out `005h` leaves its pixel-identical twin `003h` in training — undetectable leakage |
| Error is **bounded below**: MAE 1.364 h, RMSE 1.492 h | Identical pixels with different labels are indistinguishable to any model; the best possible prediction per pair is one constant |
| Only **three numbers** matter | Geometry is identical across all images; only cone/body/pylon colour varies |

Concretely: nearest-neighbour regression scores **1.36 h in-sample** and **22.82 h** under
grouped CV. Same model, same data — a factor of 17, purely from being denied the lookup.

## Quickstart

```bash
docker compose up                         # full reproduction, ~17 min, writes reports/
uv sync && uv run glys-rul reproduce      # without Docker
make help                                 # validate · search · test · verify
```

**Your own data:** edit the mount path in [`compose.yaml`](compose.yaml). Contract: images
named `<hours>h.jpeg`, a colour-scale reference `temp.png`, engines forming three connected
regions on a light background. `make validate` checks it before any training. Nothing
data-derived is hard-coded — floors, group counts and fold counts are computed at runtime.

## Results

Grouped leave-one-group-out CV: six folds, each holding out one content group entirely.

| Model | MAE [h] | Skill | Input |
|---|---:|---:|---|
| **feature_mlp** (5 seeds) | **10.77 ± 1.23** | **0.676** | Σ°C |
| isotonic | 11.68 | 0.649 | Σ°C |
| linear | 11.92 | 0.642 | Σ°C |
| monotone_mlp (5 seeds) | 20.38 ± 0.34 | 0.388 | cone/body/pylon |
| nearest_neighbour | 22.82 | 0.314 | Σ°C |
| mean | 33.28 | 0.000 | — |
| cnn | 41.07 | −0.234 | 64×64 temperature map |
| cnn_unmasked_background | 47.01 | −0.413 | as above, background unmasked |

*Floor: MAE 1.364 h · RMSE 1.492 h.* Neural rows are the mean ± sd over five seeds
(`results.json → seed_sweep`); the `models` block in that file holds the single seed-0 run
(feature_mlp 9.20 h, monotone_mlp 20.65 h) used for the figures and out-of-fold predictions.

**Read the spread before the headline.** At eleven samples a single run is initialisation
noise: over ten seeds this model ranges 7.99–14.80 h (mean 11.01, sd 1.96). Isotonic
regression is *deterministic* at 11.68 h, so the neural model's ~0.9 h margin **sits inside
one standard deviation**. Better in expectation, not cleanly separable from seed noise.

![Baseline ladder](reports/figures/ladder.png)
![Predicted versus actual](reports/figures/predicted_vs_actual.png)

### Why 10.8 h and not 1.36 h

The floor bounds what the *labels* permit. A second bound comes from having only six
states: each fold trains on five and predicts a sixth — interpolation inside, extrapolation
at the edges. **That edge behaviour is exactly why the network wins:**

| | endpoint MAE | interior MAE | held-out 3 h | held-out 100 h |
|---|---:|---:|---:|---:|
| feature_mlp | **5.65** | 10.53 | **11.14** | **97.33** |
| isotonic | 20.67 | **8.31** | 25.00 | 80.00 |

Isotonic is *better in the interior*. It clips to its training range and structurally
cannot predict outside it; the network extrapolates. The whole advantage is at the edges.

### Supporting evidence

| Check | Result |
|---|---|
| Learning curve (2→5 groups) | 55.0 → 27.0 → 17.9 → 10.2 h — steep, no plateau: **data-limited, not model-limited** |
| Shuffled-label control | 34.46 h vs 33.28 h no-skill — no residual signal to memorise, so the result is genuine |
| Permutation importance | cone 13.20 h · pylon 9.24 h · body −0.11 h |
| Occlusion (CNN) | sensitivity diffuse across the whole frame including background — it never attends to the engine |
| Conformal interval | ±21.95 h at 90 % coverage (jackknife+, finite-sample corrected) |

## Optimisation campaign

42 trials, 42 successful, logged in full — failures included — to
[`reports/experiments.csv`](reports/experiments.csv).

| Family | best | median | | Input | best | median |
|---|---:|---:|---|---|---:|---:|
| feature_mlp | **8.05** | 12.19 | | Σ°C | **8.05** | **12.71** |
| monotone_mlp | 12.56 | 15.89 | | three regions | 14.55 | 22.10 |
| cnn | 24.87 | 30.33 | | | | |

**All ten leading configurations use the summed feature.** Three separate temperatures
carry strictly more information and perform strictly worse — at eleven samples the sum is a
physics-motivated reduction that regularises better than anything the optimiser learns.

The reported figure stays **10.77 h**, not the campaign's 8.05 h: that configuration was
selected by the same CV that scores it. The gap measures how much tuning flatters a model
at this sample size.

## Rubric coverage

| Requirement | Weight | Where |
|---|---|---|
| Feature-Extraction | 25 % | [`colorscale`](src/glys_rul/colorscale.py) · [`segment`](src/glys_rul/segment.py) · [`features`](src/glys_rul/features.py) · [`audit`](src/glys_rul/audit.py) · BERICHT §1 |
| Reproduzierbare Umgebung | 10 % | [`Dockerfile`](Dockerfile) · [`compose.yaml`](compose.yaml) · [`ci.yml`](.github/workflows/ci.yml) · BERICHT §2 |
| Netzarchitektur | 15 % | [`models.py::build_cnn`](src/glys_rul/models.py) · BERICHT §3 |
| Alternative Architektur | 15 % | [`build_mlp`, `build_monotone_mlp`](src/glys_rul/models.py) · BERICHT §4 |
| Trainingseinstellungen | 15 % | [`train`](src/glys_rul/train.py) · [`dataset`](src/glys_rul/dataset.py) · BERICHT §5 |
| Schätzgenauigkeit | 20 % | [`evaluate`](src/glys_rul/evaluate.py) · [`figures/`](reports/figures) · BERICHT §6 |

## Reproducibility

`results.json` holds metrics only; volatile provenance sits in `run_meta.json`. CI builds
the pinned container, runs the pipeline and compares — so the claim is machine-checked.

| | reproducibility |
|---|---|
| Data audit, floors, features, labels, curve, control, all four baselines | **exact** |
| `feature_mlp`, `monotone_mlp` | within 1e-4 (observed ~4e-7) |
| `cnn`, `cnn_unmasked_background` | within 1e-2 (observed ~1.5e-3) |

Everything in numpy/scikit-learn is exact anywhere. TensorFlow-trained models shift in the
low-order digits between CPU models because kernels dispatch on SIMD capability — mildly
for dense matmuls, more for convolution. [`compare_results.py`](scripts/compare_results.py)
holds each part as tightly as it genuinely permits; a 5 % drift still fails the build.

Stack: digest-pinned base image · locked dependencies · `platform: linux/amd64` ·
`TF_ENABLE_ONEDNN_OPTS=0` · CPU-only · single-threaded · all generators seeded.

## Layout

```
src/glys_rul/   pipeline modules, one responsibility each
tests/          182 tests; slow ones touch real data or train models
data/raw/       the 11 supplied images + colour scale, md5-manifested
reports/        committed results, feature table, campaign ledger, figures
web/            interactive demo (static, no build step)
```

## Understanding the scores

**MAE** — mean absolute error in hours; directly interpretable. **RMSE** — squares errors
first, so large misses dominate; reported because the networks train on MSE. **R²** —
fraction of label variance explained; negative means worse than predicting the average.
**Skill** — `1 − MAE_model / MAE_mean`; 0 means no better than guessing, and it exists
because a raw MAE is meaningless without knowing the label spread.

**The floors.** Five byte-identical pairs carry different labels, so no function of the
pixels can separate them. The best any model can do is one constant per pair — the median
minimises MAE, the mean minimises RMSE:

```
MAE floor  = 15   / 11      = 1.364 h
RMSE floor = √(24.5 / 11)   = 1.492 h
```

Computed at runtime in [`audit.py::error_floors`](src/glys_rul/audit.py), never hard-coded,
so a swapped dataset gets its own bound. Every accuracy figure draws these lines. Reaching
them means the model is optimal; **beating them means something leaked** — and a test fails
the build if any model reports a score below the floor.
