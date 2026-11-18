"""Load the simulated card transactions (183 daily files, April to September 2018).

Columns: TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID (the merchant terminal), TX_AMOUNT,
TX_TIME_SECONDS, TX_TIME_DAYS, TX_FRAUD (label), TX_FRAUD_SCENARIO (1 large amount, 2 compromised terminal,
3 compromised card). The scenario column is kept for analysis only and never used as a feature.
"""
import glob
import os

import pandas as pd


def load(root, start=None, end=None):
    files = sorted(glob.glob(os.path.join(root, "data", "*.pkl")))
    if start:
        files = [f for f in files if os.path.basename(f)[:10] >= start]
    if end:
        files = [f for f in files if os.path.basename(f)[:10] <= end]
    df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    df["CUSTOMER_ID"] = df.CUSTOMER_ID.astype(int)
    df["TERMINAL_ID"] = df.TERMINAL_ID.astype(int)
    df["TX_TIME_DAYS"] = df.TX_TIME_DAYS.astype(int)
    return df.sort_values("TX_DATETIME").reset_index(drop=True)
