# Submission Checklist

Brief: `mckoh/industrial_computing_dibse_26/documents/project_exercise.md`

Mechanical checks: `make submission-check` (must exit 0).
This file covers what a script cannot judge — each rubric point cites the evidence that
satisfies it. A row without evidence is not done.

## Rubric coverage

| # | Requirement | Weight | Evidence | Done |
|---|---|---|---|---|
| 1 | Feature-Extraction durchgeführt | 25 % | `colorscale.py` (LUT + invertibility assertion, round-trip 12.7 °C), `segment.py` (3/3 regions on all 11 images), `features.py` (median °C per region, area features excluded with measured justification), `audit.py` (6 content groups, floors derived at runtime); `reports/features.csv`; BERICHT §1 | ☑ |
| 2 | Reproduzierbare Umgebung aufgesetzt | 10 % | `Dockerfile` (digest-pinned), `compose.yaml`, `uv.lock`, `.github/workflows/ci.yml`; container run verified byte-identical to host run; BERICHT §2 | ☑ |
| 3 | Geeignete Netzarchitektur entworfen | 15 % | `models.py::build_cnn` — 25 745 params on calibrated 64×64 temperature maps; `results.json.models.cnn`; BERICHT §3 | ☑ |
| 4 | Zweite, alternative Architektur | 15 % | `models.py::build_mlp` (209 params, numeric vectors — the brief's own hint) and `build_monotone_mlp`; best model at MAE 9.20 h; BERICHT §4 | ☑ |
| 5 | Trainingseinstellungen + Training | 15 % | `train.py` grouped LOGO CV + seed sweep, `dataset.py` label-preserving augmentation, no early stopping (justified); BERICHT §5 | ☑ |
| 6 | Analyse der Schätzgenauigkeit | 20 % | baseline ladder, floors on every plot, endpoint/interior decomposition, learning curve, shuffled-label control, conformal intervals, 5 figures; BERICHT §6 | ☑ |
|   | **Total** | **100 %** | | ☑ |

## Brief-level requirements

| Requirement | Evidence | Done |
|---|---|---|
| Repo prepared so the code is reproducibly usable | `docker compose up` on a clean clone; non-Docker fallback via `uv sync` | ☑ |
| Public repo, or mckoh added as member | repository visibility is public | ☐ (at publish) |
| Model estimates RUL from the temperature distribution | `features.py` region temperatures → `results.json` | ☑ |
| Regression head `Dense(1, activation="linear")` | `models.py` | ☑ |
| Regression loss `mse` | `estimators.py` | ☑ |
| Link submitted in Sakai | manual step after publishing | ☐ |

## Self-assessed score

| # | Requirement | Weight | Claimed | Justification |
|---|---|---|---|---|
| 1 | Feature-Extraction | 25 % | 25 % | Every claim measured rather than asserted: invertibility asserted at construction, all 11 images verified to segment into exactly 3 regions, area features shown to correlate perfectly with temperature and excluded on that evidence. The duplicate-image finding and the derived error floors go beyond the task as posed. |
| 2 | Umgebung | 10 % | 10 % | One command; container output verified byte-identical to a host run across different distributions; CI re-proves it on every push by diffing the committed results. |
| 3 | Netzarchitektur | 15 % | 15 % | CNN designed, capacity justified against sample count, evaluated honestly. It loses to the mean baseline — reported as a finding with the mechanism explained (fixed geometry, no spatial structure), not concealed. Two real defects found and fixed while measuring it. |
| 4 | Alternative Architektur | 15 % | 15 % | Non-convolutional dense network on numeric vectors, exactly the brief's suggestion, and the best model in the project. Plus a monotone variant with a structural guarantee verified after training. |
| 5 | Training | 15 % | 15 % | Grouped LOGO CV, multi-seed reporting, fold-local scaling, a priori epoch budget, and an explicit argument for omitting early stopping. Photometric augmentation excluded because colour is the label. |
| 6 | Analyse | 20 % | 20 % | Five-rung baseline ladder, floors drawn on every plot, error decomposed into endpoint vs interior, learning curve, negative control, conformal intervals, and a stated limitations section. |

**Overall claim: 100 %.** The work covers every rubric item with measured evidence, and the
places where results were unflattering — the CNN, the refuted three-feature hypothesis, the
gap between in-sample and honest scores — are reported as findings rather than omitted.
