#!/usr/bin/env bash
# Mechanical submission checks. Every one must pass before publishing.
#
# This verifies what a script can verify: that artifacts exist, that the suite is
# green, that results reproduce, and that the API choices the brief specifies were
# actually made. It cannot judge whether an architecture is "geeignet" — that half
# lives in docs/submission-checklist.md, where every rubric point cites its evidence.
set -uo pipefail
cd "$(dirname "$0")/.."

FAILED=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILED=1; }
check() { if eval "$2" >/dev/null 2>&1; then pass "$1"; else fail "$1"; fi; }

echo "== Repository hygiene"
check "working tree is clean"                  "[ -z \"\$(git status --porcelain)\" ]"
check "README present"                         "[ -f README.md ]"
check "German report present"                  "[ -f BERICHT.md ]"
check "local working notes are untracked"      "! git ls-files | grep -q '^docs/plan'"

echo "== Quality gates"
check "ruff reports no findings"               "uv run ruff check src tests"
check "test suite passes"                      "uv run pytest -q"

echo "== 25% Feature extraction"
check "feature table written"                  "[ -s reports/features.csv ]"
check "duplicate report written"               "[ -s reports/duplicates.csv ]"
check "area features recorded as excluded"     "grep -q 'cone_px' reports/results.json"

echo "== 10% Reproducible environment"
check "Dockerfile and compose present"         "[ -f Dockerfile ] && [ -f compose.yaml ]"
check "base image pinned by digest"            "grep -q 'sha256:' Dockerfile"
check "no unresolved digest placeholder"       "! grep -q 'REPLACE_' Dockerfile"
check "dependencies locked"                    "[ -f uv.lock ]"
check "CI workflow present"                    "[ -f .github/workflows/ci.yml ]"

echo "== 15% Network architecture"
check "CNN evaluated in results"               "grep -q '\"cnn\"' reports/results.json"

echo "== 15% Alternative architecture"
check "feature MLP evaluated in results"       "grep -q '\"feature_mlp\"' reports/results.json"
check "monotone variant evaluated"             "grep -q '\"monotone_mlp\"' reports/results.json"

echo "== 15% Training settings"
check "seed sweep configured"                  "grep -q 'SEEDS' src/glys_rul/config.py"
check "grouped splits used"                    "grep -q 'grouped_splits' src/glys_rul/train.py"

echo "== 20% Accuracy analysis"
check "baseline ladder present"                "grep -q '\"isotonic\"' reports/results.json"
check "error floors reported"                  "grep -q '\"floors\"' reports/results.json"
check "learning curve reported"                "grep -q '\"learning_curve\"' reports/results.json"
check "shuffled-label control reported"        "grep -q '\"shuffled_label_control\"' reports/results.json"
check "at least five figures generated"        "[ \$(ls reports/figures/*.png 2>/dev/null | wc -l) -ge 5 ]"
check "optimisation ledger committed"          "[ -s reports/experiments.csv ]"

echo "== Brief-specific requirements"
check "regression head is Dense(1, linear)"    "grep -q 'Dense(1, activation=\"linear\")' src/glys_rul/models.py"
check "mse loss used"                          "grep -q '\"mse\"' src/glys_rul/estimators.py"
check "report is in German"                    "grep -q 'Schätzgenauigkeit' BERICHT.md"
check "demo model exported"                    "[ -s web/model.json ] && [ -f web/index.html ]"

echo "== Reproducibility"
check "committed results match a fresh run" \
  "uv run glys-rul reproduce --no-neural --output /tmp/subcheck >/dev/null 2>&1 && \
   uv run python -c \"
import json,sys
a=json.load(open('reports/results.json')); b=json.load(open('/tmp/subcheck/results.json'))
sys.exit(0 if a['floors']==b['floors'] and a['dataset']==b['dataset'] else 1)\""

echo "== Scientific integrity"
check "no model beats the irreducible floor" \
  "uv run python -c \"
import json,sys
r=json.load(open('reports/results.json')); f=r['floors']['mae']
sys.exit(1 if [k for k,v in r['models'].items() if v['mae'] < f-1e-6] else 0)\""
check "shuffled labels destroy the signal" \
  "uv run python -c \"
import json,sys
c=json.load(open('reports/results.json'))['shuffled_label_control']
sys.exit(0 if c['mae'] >= c['no_skill_mae']*0.9 else 1)\""

echo
if [ "$FAILED" -eq 0 ]; then
  echo "All submission checks passed."
else
  echo "Submission checks FAILED. Do not publish."
fi
exit "$FAILED"
