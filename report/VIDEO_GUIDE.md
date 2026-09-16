# Video presentation guide (7–10 minutes)

This is a script *skeleton* plus the technical explanations you need in order to
answer follow-up questions. **Do not read it aloud verbatim.** Read the
"why it works this way" boxes until you can explain each idea in your own words —
that is what the viva and the marking rubric are actually testing.

The rubric rewards: implementation-specific explanation, technical
understanding, one important technical/modelling decision discussed in depth,
and one limitation or failure case. It penalises generic description.

---

## Timing plan

| Time | Section | What is on screen |
|---|---|---|
| 0:00–0:45 | Problem and research question | Title slide |
| 0:45–2:15 | Data handling and memory management | `results/memory_benchmark.csv`, `src/ingest.py` |
| 2:15–3:45 | Exploratory findings that drove the design | `eda_01`, `eda_02`, `eda_05` figures |
| 3:45–4:45 | The three models and why those three | Model comparison table |
| 4:45–6:00 | Experiments and results | `results/metrics_square_*.csv`, `tuning_sequences.png` |
| 6:00–7:15 | **Deep dive: one technical decision** | Live code walkthrough |
| 7:15–8:30 | **Failure case** | `failure_window_sq*.png`, `error_by_hour_sq*.png` |
| 8:30–9:15 | Conclusion, limitations, future work | Summary slide |

Screen-record with the repository open. Showing real code and real CSV output is
what separates "Exemplary" from "Developing" on the video criterion.

---

## 0:00–0:45 — Problem and research question

Say roughly:

> Mobile operators need to know how much traffic a cell will carry in the next
> few minutes, because that drives radio resource allocation, load balancing and
> whether capacity needs to be provisioned. I am forecasting Internet traffic
> one step ahead — the next 10 minutes — for individual areas of Milan, and
> asking how three structurally different sequential models compare, and whether
> the answer changes depending on the traffic characteristics of the area.

Then state the concrete setup: 10,000 areas, 10-minute resolution, 62 days,
test week fixed at 16–22 December 2013.

---

## 0:45–2:15 — Data handling and memory management

**Lead with the constraint, not the solution.** The constraint is what makes
your decisions defensible.

> The raw dataset is 20.6 GB of tab-separated text — about 340 million rows.
> My laptop has 8 GB of RAM and 2 cores. So the pipeline had to be designed so
> that memory use is independent of dataset size.

Then walk through the five decisions. Show `src/ingest.py` on screen.

1. **Aggregate away country code during parsing.** Each raw row is
   `(square, interval, country_code)`. The forecasting task is per area, so
   country code is summed out as each chunk is read — 340 M rows collapse to
   `62 × 144 × 10,000 = 89.28` M cells, about 3.8×.
2. **Parse only 3 of 8 columns** with `usecols`. The SMS and call columns are
   never allocated at all.
3. **Chunked streaming**: 1 million rows at a time, folded into an accumulator.
   Peak memory is a function of chunk size, not file size.
4. **Store dense `float32`, read back memory-mapped.** The grid is complete —
   about 1,439,850 of 1,440,000 cells are non-zero every single day — so a dense
   matrix wastes nothing and needs no index. 341 MB total. Reading one area's
   whole series touches 35 KB.
5. **Download → process → delete.** Peak disk is ~720 MB, not 20.6 GB.

Quote the measured result: peak RSS for one day falls from about **2.5 GB**
(naive `read_csv`) to about **150 MB**; disk from 20.6 GB to 341 MB.

> **Why it works this way — be ready to explain**
>
> *Why `float32` and not `float64`?* `float32` keeps ~7 significant decimal
> digits. The activity values are a normalised index derived from CDR counts,
> and their measurement noise is far larger than that, so the precision is not
> the limiting factor — but halving the store from 682 MB to 341 MB is what lets
> it be memory-mapped comfortably on an 8 GB machine.
>
> *Then why accumulate in `float64`?* Because summing thousands of small values
> into a `float32` accumulator loses precision progressively — each addition
> rounds. I accumulate a day in `float64` and cast once at the end. The stored
> value is accurate to `float32`; it is not the result of `float32` arithmetic.
> This is a real distinction and a good thing to be asked about.
>
> *Why dense rather than sparse?* Because the data is not sparse. I checked:
> essentially every (area, interval) cell has traffic. A sparse format would
> store indices alongside values and end up **larger**, not smaller.
>
> *Why memory-mapping?* `numpy.memmap` lets the OS page in only the bytes you
> touch. Reading column 5161 of the store reads that column, not the file. This
> is why the EDA and all training run in a few hundred MB.

**Be honest about the trade-off**, because the rubric rewards awareness of
limitations: the optimisation is one-directional. I threw away the SMS and call
columns. If I later wanted a multivariate model using voice traffic as an
exogenous input, I would have to re-download 20.6 GB. I judged that acceptable
because the task specifies Internet traffic.

Optionally mention the download engineering: Harvard Dataverse throttles each
connection to ~0.17 MB/s, while the link sustains >5 MB/s, so the downloader
issues 8 parallel HTTP range requests per file — about 4 MB/s, a 23× speedup
that turned a 34-hour download into roughly 2 hours.

---

## 2:15–3:45 — Exploratory findings that drove the design

Show only findings that changed a decision. Do not narrate every figure.

**Finding 1 — traffic is extremely unevenly distributed in space.**
Show `eda_01_spatial_distribution.png`. Report the skewness, the Gini
coefficient, and the share of total traffic held by the top 1% of areas
(read the real numbers off `results/eda_spatial_summary.csv`).

> *Consequence:* results from one area do not generalise to the city. That is
> exactly why the assignment asks for three areas, and why I apply one tuned
> configuration to all three rather than tuning per area.

**Finding 2 — the three top areas and squares 4159 and 4556 differ in
magnitude but share a shape.** Show `eda_02_two_week_series.png` and then
`eda_03_two_week_normalised.png`.

> *Consequence:* if the *shape* is shared and only the *scale* differs, then a
> per-area normalisation should let the same architecture transfer across areas.
> That is why the scaler is fitted per area, on training data only.

**Finding 3 — the autocorrelation structure tells you the lookback.**
Show `eda_05_autocorrelation.png`. Point at the ACF peak at lag 144 (one day)
and the periodogram peaks at 24 h and 12 h.

> *Consequence, and this is the key link between EDA and modelling:* a model
> that cannot reach 144 steps back cannot use "same time yesterday". That is why
> the lookback is 144, and it is why the TCN needs at least 6 dilation levels —
> its receptive field is a hard architectural limit, not a soft preference.

**Finding 4 — the series is non-stationary in a specific, exploitable way.**
Show the ADF/KPSS table and `eda_06_stl_decomposition.png`.

> *Consequence:* the raw series fails stationarity, but after differencing at
> lag 144 both tests agree it is stationary. That is precisely the transform
> SARIMA needs, and it is why my SARIMA is `(p,d,q)(0,1,0)[144]` rather than an
> arbitrary order. The STL variance shares tell you how much of the signal is
> the daily cycle versus the irregular remainder — the remainder share is an
> upper bound on how well *any* model can do.

---

## 3:45–4:45 — The three models and why those three

State the selection criterion first: three *different families*, not three
variants of one architecture.

| Model | Family | Aggregates history by | Selected because |
|---|---|---|---|
| SARIMA(p,d,q)(0,1,0)[144] | linear statistical | explicit seasonal differencing + ARMA | The EDA showed seasonal differencing makes the series stationary, so a linear model is genuinely appropriate — and it is the honest reference point for whether deep learning is needed at all. |
| LSTM | recurrent neural | gated sequential hidden state | Most-reported architecture for cellular traffic prediction; gating handles long-range dependence without vanishing gradients. |
| TCN | convolutional neural | stacked dilated causal convolutions | Structurally opposite to the LSTM — parallel over time rather than sequential, fixed receptive field rather than unbounded state. Makes the comparison informative. |

Then explain the baselines, because this is a point in your favour:

> I also report persistence and seasonal-naive, not as competitors but as
> calibration. At 10-minute resolution consecutive values are very similar, so
> persistence is a genuinely strong baseline. Reporting MASE against
> seasonal-naive means a reader can immediately see whether a model earns its
> complexity. Several published comparisons omit this, which makes their gains
> look larger than they are.

---

## 4:45–6:00 — Experiments and results

Show `figures/tuning_sequences.png` and one `results/tuning_*.csv`.

> Tuning was sequential and documented, not a blind grid. Each experiment
> changed one thing, and the reason for the next change came from the previous
> result. The CSV records the configuration, the score, and the rationale for
> every run.

Give one concrete example of the reasoning chain, e.g.:

> Experiment 1 gave the TCN a receptive field of 61 steps — deliberately less
> than a day. It could not reach the lag-144 dependency, and the validation
> error showed it. Experiment 2 added two dilation levels, receptive field 253,
> and the error dropped. That confirmed the ACF finding was actually load-bearing
> rather than decorative.

Then show the per-area metric tables and the forecast plots. Report what
actually happened, including if a baseline wins — an honest negative result
scores better than an overclaimed positive one.

Cover training and inference cost explicitly: parameter counts, seconds to
train, milliseconds per forecast step, and the hardware they were measured on.

---

## 6:00–7:15 — Deep dive: the one technical decision

Pick **one** and go deep. Recommended: **how the seasonal ARIMA was
implemented**, because it shows engineering judgement and understanding of the
model rather than library usage.

Have `src/models/sarima.py` open.

> The daily seasonal period at 10-minute resolution is 144. If you write
> `SARIMAX(order=(p,d,q), seasonal_order=(P,D,Q,144))` in statsmodels with a
> non-zero `P` or `Q`, the state-space representation carries on the order of
> 144 state variables. Every Kalman filter pass over ~8,000 observations then
> becomes minutes of computation, and the optimiser needs many passes, and my
> grid search needs many fits. On 2 cores that is not a feasible experiment.
>
> So I applied the seasonal difference explicitly instead. I take
> `y_t = log(1 + x_t)`, form `z_t = y_t − y_{t−144}`, fit an `ARIMA(p,d,q)` to
> `z`, and invert with `x̂_{t+1} = exp(ẑ_{t+1} + y_{t+1−144}) − 1`.
>
> That is mathematically `SARIMA(p,d,q)(0,1,0)[144]` — the same model — but the
> state dimension now depends only on `max(p, q+1)`. It made the order search
> affordable, and it made the seasonal step visible in the code rather than
> hidden inside a matrix.

Be ready for the obvious challenges:

* *"You dropped the seasonal AR and MA terms — isn't that a weaker model?"*
  Yes, `P = Q = 0` is a restriction. I accepted it because the EDA showed that
  seasonal differencing alone already makes both ADF and KPSS agree the series
  is stationary, so there was limited residual seasonal structure for a seasonal
  ARMA term to capture, and the computational cost was prohibitive. I state this
  as a limitation rather than pretending it is a full SARIMA.
* *"Why `log1p` before differencing?"* Because ARIMA assumes constant error
  variance, and this series is heteroskedastic — the noise scales with the level.
  Working on the log scale stabilises it. `log1p` rather than `log` because
  activity can be zero.
* *"How do you produce one-step-ahead forecasts over the test week without
  leakage?"* Parameters are estimated **once**, on the training partition only.
  I then apply those fixed parameters to the longer series and read off
  predictions with `dynamic=False`, so the forecast for `t+1` is conditioned on
  the true observations up to `t` but no test-period data ever influences the
  parameters. That matches an operational deployment: refit rarely, observe
  continuously.

An alternative deep dive, if you prefer: **the choice of Huber loss over MSE.**
The series has isolated order-of-magnitude spikes; under MSE those few samples
dominate the gradient, and the model hedges by over-predicting normal traffic.
Huber is quadratic near zero and linear in the tail, so the fit stays honest for
the bulk of the data. You can point at `RMSE/MAE` in the results tables as the
diagnostic for how peak-dominated each model's error is.

---

## 7:15–8:30 — Failure case

Show `failure_window_sq<id>.png` and `error_by_hour_sq<id>.png`.

Explain the method first, because *how you chose the example* matters:

> I did not hand-pick a bad-looking window. The script selects the 6-hour span
> with the highest mean absolute error averaged across the three models, so the
> example is chosen by evidence.

Then interpret. Read the real numbers from
`results/failure_analysis_sq<id>.csv` and `results/error_by_hour_sq<id>.csv`,
and structure the explanation around *why*, not just *where*:

* **Errors concentrate at steep transitions**, not at high levels as such. All
  models are effectively smoothers; a sharp ramp is where a smoother lags.
* **Absolute error tracks the traffic level, so the error-by-hour plot needs the
  relative column too** — the largest MAE occurs at busy hours simply because
  the values are larger there. Relative error often peaks in the small hours.
* **Bias sign is informative.** Look at `window_bias`: a model that
  systematically under-predicts a peak is failing differently from one that
  overshoots after it, and the residual panel makes it visible.
* **If the failure coincides with a calendar effect**, say so. The test week is
  16–22 December, immediately before Christmas, and the training data contains
  no comparable pre-holiday period. Any model relying purely on recent history
  has no way to anticipate a behaviour change it has never seen. This is a
  genuine limitation of the experimental design, not a bug.

---

## 8:30–9:15 — Conclusion, limitations, future work

State findings as claims with evidence attached. Then be specific about
limitations — vague ones read as filler:

* Only one city, one two-month window, one season. December includes a holiday
  build-up that the training period does not represent.
* Purely univariate and single-cell: no spatial information is used, although
  the EDA shows neighbouring cells are strongly related.
* One-step-ahead only. Multi-step forecasting is a harder and more
  operationally relevant problem, and these rankings may not carry over.
* Tuning was a guided sequential search on one area under a wall-clock budget,
  not an exhaustive search. A larger budget could change the ranking.
* Hardware-bound: 2 CPU cores capped model size, so this is not evidence about
  how these architectures behave at scale.

Future work worth one sentence each: spatial models (ConvLSTM / graph networks)
to exploit inter-cell correlation; multi-step and probabilistic forecasting
(prediction intervals matter more than point forecasts for capacity planning);
multivariate inputs using the discarded SMS and voice channels; per-area model
selection driven by the traffic-profile clustering the EDA suggests.

---

## Practical recording notes

* Screen-record at 1080p; make the font large enough to read.
* Rehearse once against a timer. Overrunning is a common, avoidable penalty.
* State your name at the start; it is an individual submission.
* Show *your* real outputs. Do not show a figure you cannot explain.
* If a result is unflattering, present it as a finding. Examiners reward the
  honest interpretation and penalise the overclaim.
* Have `src/ingest.py`, `src/models/sarima.py`, `src/models/neural.py` and
  `results/` open in tabs before you start, so navigation does not eat time.
