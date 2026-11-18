# payments-fraud-model: a card-fraud model evaluated the way a payments company runs one

This project scores card transactions for fraud risk and decides which ones to block or review. It is
evaluated on 1.75 million transactions under the conditions a payments company actually faces:

- **Labels arrive late.** A fraud is confirmed only when the chargeback comes in, about a week later. So the
  model trains on data that ends a week before the period it scores, and merchant (terminal) risk features use
  only labels that would already be known.
- **Decisions cost money.** Missed fraud costs its amount plus a $15 chargeback fee, and each alert costs a $3
  review. The threshold is chosen to minimise that cost, not to maximise accuracy.
- **The future is out of time.** There are 16 weekly folds, each training on 4 weeks, skipping 1 and testing on
  the next. There is no random split.

**[Open the report](report/report.html)** (download it and open it in a browser).

## Results (mean of 16 out-of-time weeks)

| Model | ROC AUC | Average precision | Card precision@100 per day | Cost saved vs no model |
|---|---|---|---|---|
| Gradient boosting | 0.902 | **0.750** | 0.571 | **84.9%** |
| Logistic regression | 0.899 | 0.719 | 0.554 | 83.4% |

What else the evaluation shows:
- **The leakage trap.** If merchant risk uses fraud labels as soon as the transaction happens (labels a real
  system would not have yet), average precision jumps from **0.75 to 0.997**. Many published fraud models report
  numbers like that; this pipeline makes the delay explicit and tests it.
- **What it catches.** The model catches 99% of large-amount fraud and 89% of compromised-card fraud (97% by
  amount), but only **69% of compromised-terminal fraud**. A compromised merchant terminal can only be flagged
  once its chargebacks arrive, so the first week of each attack gets through. That is the gap to close, for
  example with merchant-level anomaly signals that do not wait for labels.
- **Cost trade-off.** As review cost rises from $1 to $30 per alert, the cost-optimal alert volume falls from
  about 350 to 60 per day, and savings fall from 85% to 63%. Operations capacity and fraud loss are set
  together.
- **Calibration.** The raw model's mean predicted probability (0.84%) is already close to the actual fraud rate
  (0.88%), and isotonic recalibration does not improve the Brier score.
- **Drift.** For gradient boosting, weekly average precision moves between about 0.72 and 0.77, with only a
  slight downward trend (about 0.001 per week). Feature PSI between the first training month and the last test
  week stays below 0.02.
- **Drivers.** Ranked by permutation importance on average precision: the transaction amount relative to the
  card's 30-day average, the terminal's fraud rate over the last 7 and 30 *known* days, and the card's usual
  spend.

## Run it

```bash
pip install -r requirements.txt
python tools/get_data.py      # clones the simulated transactions (about 130 MB)
python -m fraud.report        # features (cached after the first run), 16 folds x 2 models, leakage test, report
python -m pytest -q
```

Costs, label delay and the review-cost grid are in [`config.yaml`](config.yaml).

The tests check the time logic directly:
- With a 7-day label delay, a terminal's first fraud stays invisible until day 7.
- The leaky version sees the current label.
- Card windows count correctly.
- Training and test folds are at least 7 days apart.
- The cost-optimal threshold beats no model.
- Card precision@k matches a hand-computed example.

## Layout

```
fraud/data.py       loads the 183 daily files
fraud/features.py   card behaviour (1/7/30-day counts and mean amounts), amount vs card average,
                    terminal fraud rate over windows that end `delay` days ago (delay=0 = the leaky version)
fraud/evaluate.py   rolling out-of-time folds, models, AP, card precision@k, cost, cost-optimal threshold
fraud/report.py     every result above, charts, report/report.html and report/results.json
```

## Data

The data are the simulated card transactions of the *Reproducible Machine Learning for Credit Card Fraud
Detection* handbook (Le Borgne, Siblini, Lebichot and Bontempi, 2022; ULB Machine Learning Group and
Worldline). There are about 5,000 cards (4,990 with transactions), 10,000 terminals and 1.75 million transactions from April to September 2018,
and 0.84% are fraudulent, under three patterns: large amounts, compromised terminals and compromised cards. The
pattern label is used only to break down the results, never as a feature. The data are downloaded at run time
and not stored here.
