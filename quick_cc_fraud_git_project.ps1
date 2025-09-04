# ------------------------------------
# Open PowerShell and go to the folder
# ------------------------------------
cd "C:\Users\omoaletsane\OneDrive - Botswana Savings Bank\Desktop\quick_cc_fraud"

# --------------------------
# Make sure Git is installed
# --------------------------
git --version

# ------------------------------------------------------------------------
# Create a .gitignore so you don’t push the big dataset or local artifacts
# ------------------------------------------------------------------------
@"
# Data (do NOT push the Kaggle CSV)
creditcard.csv
*.csv

# Outputs / figures
outputs/
outputs_fast/
*.png
*.pdf
*.svg

# Python cache & envs
__pycache__/
*.pyc
.venv/
venv/

# Editors / OS
.ipynb_checkpoints/
.DS_Store
Thumbs.db
"@ | Out-File -Encoding utf8 .gitignore

# -----------------------------------------
# Initialize the repo and set your identity
# -----------------------------------------
git init -b main
git config user.name  "Oduetse Moaletsane"
git config user.email "omoaletsane@gmail.com"

# --------------------------
# Stage only the right files
# --------------------------
git add README.md requirements.txt quick_cc_fraud.py .gitignore
git status

# ------
# Commit
# ------
git commit -m "Initial commit: fast baseline fraud model (no dataset)"

# --------------------------------------------------------------------------------
# Create a new repo on GitHub
# Go to GitHub → New repository → name it quick_cc_fraud and make it Public.
# Do not add a README/.gitignore/license on GitHub (we already have them locally).
# --------------------------------------------------------------------------------
# ------------------------
# Add the remote and push.
# ------------------------
git remote add origin https://github.com/omoaletsane/quick_cc_fraud.git
git push -u origin main

# ----------------------------------
# UNIT AND INTEGRATION TESTS
# ----------------------------------
# Create the tests/ folder and files
# From your repo root
# ----------------------------------
mkdir tests -Force | Out-Null

# tests\conftest.py
@'
import pandas as pd
import numpy as np

COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

def make_tiny_creditcard_csv(path, n_rows=500, fraud_ratio=0.02, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_rows * fraud_ratio))
    n_good = n_rows - n_fraud

    def block(n, cls):
        df = pd.DataFrame({
            "Time": rng.normal(50000, 20000, size=n),
            "Amount": rng.gamma(2.0, 30.0, size=n),
            **{f"V{i}": rng.normal(0, 1, size=n) for i in range(1, 29)},
            "Class": cls,
        })
        return df

    tiny = pd.concat([block(n_good, 0), block(n_fraud, 1)], axis=0).sample(frac=1, random_state=seed)
    tiny[COLUMNS].to_csv(path, index=False)
    return path
'@ | Out-File -Encoding utf8 tests\conftest.py

# tests\test_unit_quick.py
@'
from pathlib import Path
import pandas as pd
from conftest import make_tiny_creditcard_csv, COLUMNS

def test_tiny_csv_structure(tmp_path):
    csv_path = tmp_path / "tiny.csv"
    make_tiny_creditcard_csv(csv_path, n_rows=50, fraud_ratio=0.1)

    df = pd.read_csv(csv_path)
    # Has all required columns
    for c in COLUMNS:
        assert c in df.columns, f"Missing column {c}"
    # Has both classes present
    assert set(df["Class"].unique()) == {0, 1}
    # Non-empty
    assert len(df) > 0
'@ | Out-File -Encoding utf8 tests\test_unit_quick.py

# tests\test_integration_quick.py
@'
import os
import sys
import subprocess
from pathlib import Path
from conftest import make_tiny_creditcard_csv

def test_end_to_end_ultra_fast(tmp_path):
    # Arrange: create tiny dataset
    csv_path = tmp_path / "creditcard.csv"
    make_tiny_creditcard_csv(csv_path, n_rows=600, fraud_ratio=0.03)

    # Env knobs for speed
    env = os.environ.copy()
    env["CREDITCARD_CSV"] = str(csv_path)
    env["ULTRA_FAST"] = "1"         # no GridSearch, single fit per model
    env["SAMPLE_NONFRAUD"] = "300"  # small majority keep (quick)

    # Script path
    script = str(Path.cwd() / "quick_cc_fraud.py")
    assert Path(script).exists(), f"{script} not found"

    # Run the script
    proc = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True, timeout=300)
    print("STDOUT:\\n", proc.stdout)
    print("STDERR:\\n", proc.stderr)
    assert proc.returncode == 0, "Script failed to run"

    # Check outputs
    out_dir = Path("outputs_fast")
    assert out_dir.exists(), "outputs_fast folder missing"

    expected = [
        out_dir / "results_fast.csv",
        out_dir / "cm_logreg.png",
        out_dir / "cm_decisiontree.png",
        out_dir / "run_info.txt",
    ]
    for p in expected:
        assert p.exists(), f"Missing expected output: {p}"
'@ | Out-File -Encoding utf8 tests\test_integration_quick.py

# -------------------------
# Install test dependencies
# -------------------------
pip install -r requirements.txt

# ---------------------------------
# or just ensure pytest is present:
# ---------------------------------
pip install pytest

# ------------
# Run the test
# ------------
pytest -q

# ------------------------------
# Stage, Commit & push the tests
# ------------------------------
git add tests
git status
Commit it
git commit -m "Add tests folder with unit and integration tests"
git push








