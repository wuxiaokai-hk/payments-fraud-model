"""Time-respecting evaluation of a card-fraud model, as a payments company would run it.

Each fold: train on 4 weeks of transactions, skip 1 week (labels of the most recent week are not known yet:
chargebacks arrive late), then score the next week. Folds roll forward one week at a time.

Metrics per fold:
- ROC AUC and average precision (AP; the area under the precision-recall curve, the right summary when
  fraud is under 1% of transactions);
- card precision@100: each day, the share of fraudulent cards among the 100 cards with the highest risk, the
  question an investigation team with a fixed daily capacity actually faces;
- cost: missed fraud (amount + chargeback fee) plus review cost of every alert, at the threshold chosen on the
  training window to minimise that cost.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_FRAUD",
        "TX_FRAUD_SCENARIO"]


def feature_cols(df):
    return [c for c in df.columns if c not in META]


def models(seed=0):
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                                            l2_regularization=1.0, random_state=seed),
    }


def folds(df, start="2018-05-01", train_weeks=4, delay_weeks=1, test_weeks=1, n=None):
    t0 = pd.Timestamp(start)
    i = 0
    while True:
        tr0 = t0 + pd.Timedelta(weeks=i)
        tr1 = tr0 + pd.Timedelta(weeks=train_weeks)
        te0 = tr1 + pd.Timedelta(weeks=delay_weeks)
        te1 = te0 + pd.Timedelta(weeks=test_weeks)
        if te1 > df.TX_DATETIME.max() + pd.Timedelta(seconds=1) or (n and i >= n):
            return
        tr = df[(df.TX_DATETIME >= tr0) & (df.TX_DATETIME < tr1)]
        te = df[(df.TX_DATETIME >= te0) & (df.TX_DATETIME < te1)]
        yield i, tr, te
        i += 1


def card_precision_at_k(te, score, k=100):
    d = te[["TX_DATETIME", "CUSTOMER_ID", "TX_FRAUD"]].assign(score=score, day=te.TX_DATETIME.dt.date)
    per_card = d.groupby(["day", "CUSTOMER_ID"]).agg(score=("score", "max"), fraud=("TX_FRAUD", "max"))
    vals = [g.nlargest(k, "score").fraud.mean() for _, g in per_card.groupby(level="day")]
    return float(np.mean(vals))


def cost(y, amount, alert, cfg):
    """Missed fraud costs its amount plus the chargeback fee; every alert costs a review."""
    missed = (y == 1) & ~alert
    return float((amount[missed] + cfg["chargeback_fee"]).sum() + alert.sum() * cfg["review_cost"])


def best_threshold(y, amount, p, cfg):
    grid = np.unique(np.quantile(p, np.linspace(0.90, 0.9999, 300)))
    costs = [cost(y, amount, p >= t, cfg) for t in grid]
    return float(grid[int(np.argmin(costs))])


def run(df, cfg, n_folds=None):
    X = feature_cols(df)
    rows, preds = [], []
    for i, tr, te in folds(df, n=n_folds):
        for name, m in models().items():
            m.fit(tr[X], tr.TX_FRAUD)
            p_tr = m.predict_proba(tr[X])[:, 1]
            p = m.predict_proba(te[X])[:, 1]
            thr = best_threshold(tr.TX_FRAUD.values, tr.TX_AMOUNT.values, p_tr, cfg)
            alert = p >= thr
            y, amt = te.TX_FRAUD.values, te.TX_AMOUNT.values
            c_model = cost(y, amt, alert, cfg)
            c_none = cost(y, amt, np.zeros(len(y), bool), cfg)
            rows.append({"fold": i, "test_start": te.TX_DATETIME.min().date(), "model": name,
                         "auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
                         "cp100": card_precision_at_k(te, p), "threshold": thr, "alerts": int(alert.sum()),
                         "frauds": int(y.sum()), "caught": int((alert & (y == 1)).sum()),
                         "cost": c_model, "cost_no_model": c_none, "saving_pct": 100 * (1 - c_model / c_none)})
            preds.append(te[["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TX_AMOUNT", "TX_FRAUD",
                             "TX_FRAUD_SCENARIO"]].assign(model=name, fold=i, p=p, alert=alert))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
