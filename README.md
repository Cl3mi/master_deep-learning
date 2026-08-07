# Glys Engine — Remaining Useful Life Estimation

Predicting how many operating hours a spaceship engine has left, from a single
thermal-camera image.

**[→ Interactive demo](https://cl3mi.github.io/master_deep-learning/)** ·
**[→ Bericht (Deutsch)](BERICHT.md)** · **[→ Design specification](docs/design/2026-08-06-glys-rul-design.md)**

---

## The finding that shapes everything

The supplied dataset contains **11 files but only 6 unique images**. Five pairs are
byte-identical yet carry different labels:

| Files | Labels | md5 |
|---|---|---|
| `003h.jpeg` = `005h.jpeg` | 3 h, 5 h | `87558f…` |
| `024h.jpeg` = `026h.jpeg` | 24 h, 26 h | `4dac2b…` |
| `047h.jpeg` = `051h.jpeg` | 47 h, 51 h | `8b2bf8…` |
| `073h.jpeg` = `076h.jpeg` | 73 h, 76 h | `bc54a4…` |
| `078h.jpeg` = `082h.jpeg` | 78 h, 82 h | `1b3259…` |
| `100h.jpeg` | 100 h | `c2ee1e…` |

Three consequences drive every decision in this repository:

1. **Splits must separate content hashes, not filenames.** Hold out `005h.jpeg` and its
   pixel-identical twin `003h.jpeg` sits in the training set. That is undetectable
   leakage, and it is why every split here is grouped by md5.
2. **Error is bounded below.** Identical pixels carrying different labels cannot be told
   apart by any function of the image, so the best possible prediction for a pair is one
   constant. That bounds accuracy at **MAE 1.364 h** and **RMSE 1.492 h**. A model
   reporting less has leaked or memorised.
3. **Only three numbers matter.** Geometry is identical across the dataset; only the
   colour of the cone, body and pylon varies.

How much this matters, concretely: nearest-neighbour regression scores **1.36 h**
in-sample and **22.82 h** under grouped cross-validation. The same model, the same data —
a factor of seventeen, entirely from being allowed to look up the answer.

## Quickstart

```bash
docker compose up          # full reproduction, ~6 min, writes reports/
```

Or without Docker:

```bash
uv sync && uv run glys-rul reproduce
```

Other entry points:

```bash
docker compose --profile validate run --rm validate   # check the data contract only
docker compose --profile search   run --rm search     # optimisation campaign
make help                                             # all targets
```

### Using your own data

Edit the mount path in [`compose.yaml`](compose.yaml). The format contract is:

- a folder of images named `<hours>h.jpeg` — the number is the remaining useful life
- a colour-scale reference image `temp.png` spanning the temperature range
- engines that decompose into three connected regions on a light background

`make validate` checks the contract and prints a report before any training happens.
Nothing derived from the data is hard-coded — error floors, group counts and fold counts
are all computed at runtime, so a different dataset gets its own honest bounds.

## Results

All figures are **grouped leave-one-group-out cross-validation**: six folds, each holding
out one content group entirely, so no model is ever scored on an image it has seen.

| Model | MAE [h] | RMSE [h] | R² | Skill |
|---|---:|---:|---:|---:|
| **feature_mlp** | **9.20** | **10.99** | **0.879** | **0.724** |
| isotonic | 11.68 | 13.07 | 0.829 | 0.649 |
| linear | 11.92 | 12.65 | 0.840 | 0.642 |
| monotone_mlp | 20.65 | 24.54 | 0.397 | 0.379 |
| nearest_neighbour | 22.82 | 23.04 | 0.469 | 0.314 |
| mean | 33.28 | 37.82 | −0.432 | 0.000 |
| cnn | 41.78 | 53.50 | −1.866 | −0.256 |

*Irreducible floor: MAE 1.364 h · RMSE 1.492 h.*

![Baseline ladder](reports/figures/ladder.png)
![Predicted versus actual](reports/figures/predicted_vs_actual.png)

Two supporting results:

**Learning curve** — cross-validated MAE against the number of training groups:
55.0 → 27.0 → 17.9 → 10.2 h for 2 → 5 groups. Steeply descending with no plateau, which
says the remaining error is a *data* limitation rather than a modelling one.

**Shuffled-label control** — with labels randomly permuted the model scores 34.46 h
against a no-skill baseline of 33.28 h, i.e. very slightly *worse* than guessing the
average. There is no residual signal left to memorise, so the 9.20 h result is genuine.

## Rubric coverage

| Requirement | Weight | Where |
|---|---|---|
| Feature-Extraction | 25 % | [`colorscale.py`](src/glys_rul/colorscale.py) · [`segment.py`](src/glys_rul/segment.py) · [`features.py`](src/glys_rul/features.py) · [`audit.py`](src/glys_rul/audit.py) · [`reports/features.csv`](reports/features.csv) |
| Reproduzierbare Umgebung | 10 % | [`Dockerfile`](Dockerfile) · [`compose.yaml`](compose.yaml) · [`uv.lock`](uv.lock) · [`ci.yml`](.github/workflows/ci.yml) |
| Netzarchitektur | 15 % | [`models.py::build_cnn`](src/glys_rul/models.py) |
| Alternative Architektur | 15 % | [`models.py::build_mlp`](src/glys_rul/models.py), [`build_monotone_mlp`](src/glys_rul/models.py) |
| Trainingseinstellungen | 15 % | [`train.py`](src/glys_rul/train.py) · [`dataset.py`](src/glys_rul/dataset.py) |
| Schätzgenauigkeit | 20 % | [`evaluate.py`](src/glys_rul/evaluate.py) · [`reports/figures/`](reports/figures) |

## Reproducibility

`results.json` holds metrics only; volatile provenance lives in `run_meta.json`. That
split is what makes the guarantee checkable: CI builds the pinned container, runs the
pipeline and fails if `results.json` differs by a single byte.

Verified during development: a container run produced results **byte-identical** to a run
on the host — different distribution, different libc, independently built wheels.

The determinism stack: base image pinned by digest · dependencies from a lockfile ·
`platform: linux/amd64` · `TF_ENABLE_ONEDNN_OPTS=0` (oneDNN otherwise dispatches different
kernels on AVX2 versus AVX-512 hosts) · CPU-only · single-threaded · all generators seeded.

## Repository layout

```
src/glys_rul/     pipeline modules, one responsibility each
tests/            176 tests; the slow ones touch real data or train models
data/raw/         the 11 supplied images + colour scale, with an md5 manifest
reports/          committed results, feature table and figures
web/              the interactive demo (static, no build step)
docs/design/      the design specification
```

---

## Understanding the scores

**MAE — mean absolute error, in hours.** The average distance between predicted and actual
remaining life. Directly interpretable: an MAE of 9.2 h means predictions are typically
about nine hours out. Computed as `mean(|predicted − actual|)` over the out-of-fold
predictions.

**RMSE — root mean squared error, in hours.** The same idea, but errors are squared before
averaging, so large mistakes count disproportionately. Reported because the neural models
are trained on mean squared error, making RMSE the quantity they actually optimise. RMSE
is always ≥ MAE; a large gap between them means a few big misses rather than uniform
inaccuracy.

**R² — coefficient of determination.** The fraction of label variance the model explains.
1.0 is perfect, 0 means no better than predicting the average, and negative means worse
than that. Both the `mean` baseline (−0.432) and the CNN (−1.866) are negative here, which
is a real statement: on held-out groups they are worse than a constant.

**Skill score.** `1 − MAE_model / MAE_baseline`, where the baseline is the
cross-validated mean predictor. This exists because a raw MAE is meaningless without
knowing the label spread — 9.2 h could be excellent or useless depending on whether labels
span 20 hours or 2000. 1.0 is perfect, 0 means the model adds nothing over guessing,
negative means it actively hurts.

**The floors — the most important numbers here.** Because five image pairs are
byte-identical while carrying different labels, no function of the pixels can distinguish
them. The best any model can do is predict one constant per pair: the median minimises
MAE, the mean minimises RMSE. Summed over the dataset this gives

```
MAE floor  = 15   / 11        = 1.364 h
RMSE floor = √(24.5 / 11)     = 1.492 h
```

computed at runtime in [`audit.py::error_floors`](src/glys_rul/audit.py) — never
hard-coded, so a swapped dataset gets its own bound. Every accuracy figure draws these
lines. Reaching them means the model is optimal; **beating them means something leaked**,
and a test in the suite fails the build if any model reports a score below the floor.

**Why the best model is 9.20 h and not 1.36 h.** The floor bounds what is achievable given
the *labels*. A second, separate bound comes from having only *six distinct thermal
states*: under leave-one-group-out, each fold trains on five states and must predict a
sixth. For interior states that is interpolation and works well; for the two endpoints it
is extrapolation beyond anything observed. Holding out the 3 h group, the model has seen
nothing hotter than 24 h and predicts 25 h; holding out 100 h, it has seen nothing cooler
than 82 h and predicts 80 h. Those two folds carry most of the remaining error — 20.7 h
against 8.3 h for the interior. The learning curve above says the same thing from the
other direction: every additional state roughly halves the error, with no sign of
flattening.
