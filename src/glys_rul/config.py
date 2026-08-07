"""Central configuration. Values only, no logic.

Nothing derived from the dataset belongs here: error floors, group counts and
fold counts are computed at runtime so that swapping the dataset stays safe.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- paths -------------------------------------------------------------
DATA_DIR = Path("/data") if Path("/data").is_dir() else REPO_ROOT / "data/raw/triebwersbilder"
SCALE_IMAGE = DATA_DIR / "temp.png"
MANIFEST = DATA_DIR / "MANIFEST.json"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
WEB_DIR = REPO_ROOT / "web"

# --- data contract -----------------------------------------------------
IMAGE_GLOB = "*.jpeg"
LABEL_PATTERN = r"^(?P<rul>\d+)h$"
SCALE_VMIN = 0.0
SCALE_VMAX = 1200.0
WHITE_THRESHOLD = 245.0      # mean channel value below which a pixel is foreground
MIN_COMPONENT_PX = 5000      # smaller connected components are noise
ERODE_ITERATIONS = 4         # strip JPEG ringing before sampling colour
EXPECTED_REGIONS = 3
REGION_NAMES = ("cone", "body", "pylon")
MAX_ROUNDTRIP_ERROR_C = 50.0  # colour scale must invert at least this precisely

# --- modelling ---------------------------------------------------------
SEEDS = (0, 1, 2, 3, 4)
TARGET_SCALE = 100.0         # RUL is divided by this before training
IMAGE_SIZE = 64           # 128 px costs 4x the time for no measured accuracy gain
CNN_EPOCHS = 200
CNN_AUGMENT_ROUNDS = 3    # training set is originals plus this many augmented copies
MLP_EPOCHS = 500
LEARNING_RATE = 1e-3
MAX_LOGO_GROUPS = 10         # above this, fall back to GroupKFold
GROUP_KFOLD_SPLITS = 5
HOT_THRESHOLD_C = 800.0
BIN_WIDTH_H = 10             # for the classification view
CONFORMAL_ALPHA = 0.1        # 90 % target coverage
