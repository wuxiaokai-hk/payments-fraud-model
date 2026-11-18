"""Features a payments company could compute at authorization time.

Card (customer) behaviour: number and mean amount of the card's transactions over the last 1, 7 and 30
days, including the current one.
Terminal (merchant) risk: number of transactions and fraud rate on the terminal over 1, 7 and 30-day windows
that END `delay` days before now. Fraud labels arrive late (chargebacks take days to weeks), so a feature may
only use labels that would already be known. Setting delay=0 reproduces the classic leakage mistake.
"""
import numpy as np
import pandas as pd

WINDOWS = (1, 7, 30)


def card_features(df):
    out = []
    for _, g in df.groupby("CUSTOMER_ID", sort=False):
        s = g.set_index("TX_DATETIME").TX_AMOUNT
        cols = {}
        for w in WINDOWS:
            r = s.rolling(f"{w}D")
            cols[f"card_n_{w}d"] = r.count().values
            cols[f"card_mean_amt_{w}d"] = r.mean().values
        out.append(pd.DataFrame(cols, index=g.index))
    return pd.concat(out).reindex(df.index)


def terminal_features(df, delay=7):
    out = []
    for _, g in df.groupby("TERMINAL_ID", sort=False):
        s = g.set_index("TX_DATETIME")
        n_delay = s.TX_FRAUD.rolling(f"{delay}D").count() if delay else 0
        f_delay = s.TX_FRAUD.rolling(f"{delay}D").sum() if delay else 0
        cols = {}
        for w in WINDOWS:
            n = s.TX_FRAUD.rolling(f"{delay + w}D").count() - n_delay
            k = s.TX_FRAUD.rolling(f"{delay + w}D").sum() - f_delay
            if not delay:            # leakage version: includes the current transaction's own label
                n, k = s.TX_FRAUD.rolling(f"{w}D").count(), s.TX_FRAUD.rolling(f"{w}D").sum()
            cols[f"term_n_{w}d"] = np.asarray(n, float)
            cols[f"term_risk_{w}d"] = np.where(np.asarray(n) > 0, np.asarray(k) / np.maximum(np.asarray(n), 1), 0.0)
        out.append(pd.DataFrame(cols, index=g.index))
    return pd.concat(out).reindex(df.index)


def build(df, delay=7):
    X = pd.DataFrame(index=df.index)
    X["amount"] = df.TX_AMOUNT
    X["log_amount"] = np.log1p(df.TX_AMOUNT)
    X["weekend"] = (df.TX_DATETIME.dt.weekday >= 5).astype(int)
    X["night"] = (df.TX_DATETIME.dt.hour <= 6).astype(int)
    c = card_features(df)
    X = X.join(c)
    for w in WINDOWS:
        X[f"amt_vs_card_mean_{w}d"] = df.TX_AMOUNT / c[f"card_mean_amt_{w}d"]
    X = X.join(terminal_features(df, delay))
    return X
