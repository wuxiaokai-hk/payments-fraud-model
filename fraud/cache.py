"""Build features once and cache them (the rolling windows take a few minutes on the full 1.75M rows)."""
import os

import pandas as pd

from .data import load
from .features import build, terminal_features


def features(root, cache_dir, delay=7):
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"features_delay{delay}.pkl")
    if os.path.exists(p):
        return pd.read_pickle(p)
    df = load(root)
    keep = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_FRAUD",
            "TX_FRAUD_SCENARIO"]
    base = os.path.join(cache_dir, "features_delay7.pkl")
    if delay != 7 and os.path.exists(base):          # reuse card features, rebuild terminal ones
        X = pd.read_pickle(base).drop(columns=keep)
        X = X.drop(columns=[c for c in X if c.startswith("term_")]).join(terminal_features(df, delay))
    else:
        X = build(df, delay)
    out = df[keep].join(X.fillna(0.0))
    out.to_pickle(p)
    return out
