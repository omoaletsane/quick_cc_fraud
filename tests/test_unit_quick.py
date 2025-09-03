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
