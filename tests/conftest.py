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
