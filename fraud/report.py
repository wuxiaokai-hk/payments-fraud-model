"""Build every result of the project and write report/report.html (charts embedded) and report/results.json."""
import base64
import html
import json
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss

from .cache import features
from .evaluate import run, folds, models, feature_cols, cost, best_threshold

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK2, GRID, SURF = "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "font.size": 9, "lines.linewidth": 2, "legend.frameon": False,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
SCEN = {1: "Large amount (> 220)", 2: "Compromised terminal", 3: "Compromised card"}


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)
    return name


def psi(a, b, bins=10):
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return 0.0
    pa = np.histogram(np.clip(a, q[0], q[-1]), q)[0] / len(a) + 1e-6
    pb = np.histogram(np.clip(b, q[0], q[-1]), q)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def main(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    out = cfg["report_dir"]
    os.makedirs(out, exist_ok=True)
    R = {}

    df = features(cfg["data_root"], cfg["cache_dir"], delay=cfg["label_delay_days"])
    res, preds = run(df, cfg["costs"])
    res.to_csv(os.path.join(out, "folds.csv"), index=False)
    summ = res.groupby("model")[["auc", "ap", "cp100", "saving_pct", "alerts", "caught", "frauds"]].mean()
    R["summary"] = summ.round(4).reset_index().to_dict("records")
    R["n_folds"] = int(res.fold.nunique())
    R["n_transactions"] = len(df)
    R["fraud_rate"] = float(df.TX_FRAUD.mean())

    # 1. AP by week (drift over time)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, (m, g) in enumerate(res.groupby("model")):
        ax.plot(pd.to_datetime(g.test_start), g.ap, marker="o", ms=4, color=SERIES[i], label=m)
    ax.set_ylabel("average precision")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title("Out-of-time accuracy, week by week")
    ax.legend(loc="lower left")
    ch_week = _save(fig, out, "ap_by_week.png")

    # 2. The leakage trap: terminal risk computed with labels not yet known (delay 0)
    leak = features(cfg["data_root"], cfg["cache_dir"], delay=0)
    res0, _ = run(leak, cfg["costs"])
    lk = pd.DataFrame({"realistic (7-day label delay)": res.groupby("model").ap.mean(),
                       "leaky (labels known at once)": res0.groupby("model").ap.mean()})
    R["leakage"] = lk.round(4).reset_index().to_dict("records")

    # 3. Detection by fraud scenario, at the cost-optimal threshold
    best = summ.ap.idxmax()
    pb = preds[(preds.model == best) & (preds.TX_FRAUD == 1)]
    scen = pb.groupby("TX_FRAUD_SCENARIO").agg(frauds=("alert", "size"), caught=("alert", "sum"),
                                               amount=("TX_AMOUNT", "sum"))
    scen["recall"] = scen.caught / scen.frauds
    amt_caught = pb[pb.alert].groupby("TX_FRAUD_SCENARIO").TX_AMOUNT.sum()
    scen["amount_recall"] = amt_caught.reindex(scen.index, fill_value=0) / scen.amount
    scen.index = scen.index.map(SCEN)
    R["scenarios"] = scen.round(4).reset_index().rename(columns={"TX_FRAUD_SCENARIO": "scenario"}).to_dict("records")
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.barh(scen.index, scen.recall * 100, color=SERIES[0], height=0.5)
    for i, v in enumerate(scen.recall * 100):
        ax.annotate(f"{v:.0f}%", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK2)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.set_xlabel("% of fraudulent transactions caught")
    ax.set_title(f"What the model catches, by fraud pattern ({best})")
    ch_scen = _save(fig, out, "recall_by_scenario.png")

    # 4. Calibration and cost on the last fold
    X = feature_cols(df)
    last = list(folds(df))[-1]
    _, tr, te = last
    gb = models()[best]
    gb.fit(tr[X], tr.TX_FRAUD)
    p_raw = gb.predict_proba(te[X])[:, 1]
    cal = CalibratedClassifierCV(models()[best], method="isotonic", cv=3).fit(tr[X], tr.TX_FRAUD)
    p_cal = cal.predict_proba(te[X])[:, 1]
    R["calibration"] = {"brier_raw": brier_score_loss(te.TX_FRAUD, p_raw),
                        "brier_isotonic": brier_score_loss(te.TX_FRAUD, p_cal),
                        "mean_predicted_raw": float(p_raw.mean()), "mean_predicted_isotonic": float(p_cal.mean()),
                        "actual_rate": float(te.TX_FRAUD.mean())}
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for i, (p, lab) in enumerate([(p_raw, "raw"), (p_cal, "isotonic")]):
        fy, fx = calibration_curve(te.TX_FRAUD, p, n_bins=10, strategy="quantile")
        ax.plot(fx, fy, marker="o", ms=4, color=SERIES[i], label=lab)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_title("Calibration (last week)")
    ax.legend(loc="upper left")
    ch_cal = _save(fig, out, "calibration.png")

    # 5. Cost sensitivity: savings as review cost changes, threshold re-optimised each time
    rows = []
    for rc in cfg["review_cost_grid"]:
        c2 = dict(cfg["costs"], review_cost=rc)
        p_tr = gb.predict_proba(tr[X])[:, 1]
        t = best_threshold(tr.TX_FRAUD.values, tr.TX_AMOUNT.values, p_tr, c2)
        a = p_raw >= t
        rows.append({"review_cost": rc, "alerts_per_day": a.sum() / 7, "recall": float(a[te.TX_FRAUD == 1].mean()),
                     "saving_pct": 100 * (1 - cost(te.TX_FRAUD.values, te.TX_AMOUNT.values, a, c2) /
                                          cost(te.TX_FRAUD.values, te.TX_AMOUNT.values, np.zeros(len(a), bool), c2))})
    R["cost_sensitivity"] = pd.DataFrame(rows).round(3).to_dict("records")

    # 6. What drives the model (permutation importance on AP, last week, sample)
    s = te.sample(min(150000, len(te)), random_state=0)
    pi_ = permutation_importance(gb, s[X], s.TX_FRAUD, scoring="average_precision", n_repeats=3, random_state=0,
                                 n_jobs=1)
    imp = pd.Series(pi_.importances_mean, index=X).sort_values(ascending=False).head(10)
    R["importance"] = imp.round(4).to_dict()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.barh(imp.index[::-1], imp.values[::-1], color=SERIES[0], height=0.6)
    ax.set_xlabel("drop in average precision when the feature is shuffled")
    ax.set_title("What drives the model (last week)")
    ch_imp = _save(fig, out, "importance.png")

    # 7. Drift: PSI between the first training window and the last test week
    _, tr0, _ = next(folds(df))
    drift = pd.Series({c: psi(tr0[c].values, te[c].values) for c in X}).sort_values(ascending=False)
    R["psi_top"] = drift.head(5).round(4).to_dict()
    R["ap_trend_per_week"] = {m: float(np.polyfit(g.fold, g.ap, 1)[0]) for m, g in res.groupby("model")}

    json.dump(R, open(os.path.join(out, "results.json"), "w"), indent=1, default=str)
    write_html(out, R, cfg, [ch_week, ch_scen, ch_cal, ch_imp])
    print(json.dumps(R, indent=1, default=str)[:3000])


def _img(out, name):
    raw = open(os.path.join(out, name), "rb").read()
    w = int.from_bytes(raw[16:20], "big") * 96 // 150
    return f'<img width="{w}" style="max-width:100%;height:auto" src="data:image/png;base64,{base64.b64encode(raw).decode()}">'


def _tab(records):
    if not records:
        return ""
    cols = list(records[0])
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = "".join("<tr>" + "".join(
        f"<td>{v:,.4g}</td>" if isinstance(v, float) else f"<td>{html.escape(str(v))}</td>" for v in r.values())
        + "</tr>" for r in records)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html(out, R, cfg, charts):
    css = ("body{font:15px/1.55 -apple-system,Segoe UI,Inter,sans-serif;background:#fcfcfb;color:#0b0b0b;margin:0}"
           "main{max-width:940px;margin:0 auto;padding:28px 22px 70px}h1{font-size:26px}h2{font-size:19px;"
           "margin-top:38px;border-top:1px solid #e4e3df;padding-top:12px}table{border-collapse:collapse;"
           "width:100%;font-size:13px;margin:8px 0 14px}th,td{padding:6px 8px;border-bottom:1px solid #e4e3df;"
           "text-align:left}th{color:#52514e;font-size:12px}.s{color:#52514e;font-size:13px}")
    h = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Payments fraud model</title><style>{css}"
         "</style></head><body><main><h1>Card-fraud model: out-of-time results</h1>",
         f"<p class=s>{R['n_transactions']:,} simulated card transactions (fraud rate {R['fraud_rate']:.2%}), "
         f"{R['n_folds']} weekly folds: train 4 weeks, skip 1 week (labels not yet known), test 1 week. "
         f"Costs: chargeback fee ${cfg['costs']['chargeback_fee']}, review ${cfg['costs']['review_cost']} per alert.</p>",
         "<h2>Accuracy and savings (mean over folds)</h2>", _tab(R["summary"]), _img(out, charts[0]),
         "<h2>The leakage trap</h2><p>Average precision when terminal risk uses labels that would not be known "
         "yet, against the realistic 7-day delay:</p>", _tab(R["leakage"]),
         "<h2>What it catches</h2>", _tab(R["scenarios"]), _img(out, charts[1]),
         "<h2>Calibration</h2>", _tab([R["calibration"]]), _img(out, charts[2]),
         "<h2>Cost sensitivity (last week)</h2>", _tab(R["cost_sensitivity"]),
         "<h2>Drivers</h2>", _img(out, charts[3]),
         "<h2>Drift</h2><p class=s>Top PSI between the first training window and the last test week:</p>",
         _tab([{"feature": k, "psi": v} for k, v in R["psi_top"].items()]),
         "<p class=s>Data: Fraud Detection Handbook simulated transactions (Le Borgne et al., 2022).</p>",
         "</main></body></html>"]
    open(os.path.join(out, "report.html"), "w", encoding="utf-8").write("\n".join(h))


if __name__ == "__main__":
    main()
