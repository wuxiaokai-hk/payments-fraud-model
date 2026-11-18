import numpy as np
import pandas as pd

from fraud.evaluate import cost, best_threshold, card_precision_at_k, folds
from fraud.features import terminal_features, card_features


def _tx():
    t = pd.date_range("2018-04-01", periods=40, freq="D")
    return pd.DataFrame({"TX_DATETIME": t, "CUSTOMER_ID": 1, "TERMINAL_ID": 7, "TX_AMOUNT": 10.0,
                         "TX_FRAUD": [0] * 20 + [1] * 20})


def test_terminal_risk_respects_label_delay():
    """With a 7-day delay, the first fraud (day 20) must not be visible before day 27."""
    df = _tx()
    x = terminal_features(df, delay=7)
    assert (x.term_risk_30d.iloc[:27] == 0).all()
    assert x.term_risk_30d.iloc[27] > 0


def test_leaky_version_sees_current_label():
    df = _tx()
    x = terminal_features(df, delay=0)
    assert x.term_risk_1d.iloc[20] == 1.0      # the current transaction's own label leaks in


def test_card_features_count_window():
    df = _tx()
    x = card_features(df)
    assert x.card_n_7d.iloc[10] == 7 and x.card_n_1d.iloc[10] == 1


def test_cost_and_threshold():
    y = np.array([1, 0, 0, 1])
    amt = np.array([100.0, 5, 5, 50])
    cfg = {"chargeback_fee": 15, "review_cost": 3}
    assert cost(y, amt, np.array([False] * 4), cfg) == 180.0
    assert cost(y, amt, np.array([True, False, False, True]), cfg) == 6.0
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.01).astype(int)
    amt = rng.uniform(10, 200, 5000)
    p = np.where(y == 1, rng.uniform(0.5, 1, 5000), rng.uniform(0, 0.6, 5000))
    t = best_threshold(y, amt, p, cfg)
    assert cost(y, amt, p >= t, cfg) < cost(y, amt, np.zeros(5000, bool), cfg) / 2


def test_card_precision_at_k():
    te = pd.DataFrame({"TX_DATETIME": pd.to_datetime(["2018-05-01"] * 4), "CUSTOMER_ID": [1, 2, 3, 4],
                       "TX_FRAUD": [1, 0, 1, 0]})
    assert card_precision_at_k(te, np.array([0.9, 0.8, 0.1, 0.0]), k=2) == 0.5


def test_folds_are_out_of_time_with_a_gap():
    t = pd.date_range("2018-04-01", "2018-09-30", freq="6h")
    df = pd.DataFrame({"TX_DATETIME": t})
    for _, tr, te in folds(df):
        assert te.TX_DATETIME.min() - tr.TX_DATETIME.max() >= pd.Timedelta(days=7)
