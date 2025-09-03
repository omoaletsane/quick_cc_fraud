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
