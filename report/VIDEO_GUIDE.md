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

> The raw dataset is 19.4 GB of tab-separated text — 319,896,289 rows.
> My laptop has 8 GB of RAM and 2 cores. So the pipeline had to be designed so
> that memory use is independent of dataset size.

Then walk through the five decisions. Show `src/ingest.py` on screen.

1. **Aggregate away country code during parsing.** Each raw row is
   `(square, interval, country_code)`. The forecasting task is per area, so
   country code is summed out as each chunk is read — 319.9 M rows collapse to
   `62 × 144 × 10,000 = 89.28` M cells, a factor of 3.6.
2. **Parse only 3 of 8 columns** with `usecols`. The SMS and call columns are
   never allocated at all.
3. **Chunked streaming**: 1 million rows at a time, folded into an accumulator.
   Peak memory is a function of chunk size, not file size.
4. **Store dense `float32`, read back memory-mapped.** The grid is complete —
   about 1,439,850 of 1,440,000 cells are non-zero every single day — so a dense
   matrix wastes nothing and needs no index. 341 MB total. Reading one area's
   whole series touches 35 KB.
5. **Download → process → delete.** Peak disk is ~720 MB, not 20.6 GB.

Quote the measured result, and quote it precisely — the numbers are in
`results/memory_benchmark.csv`: peak RSS for one day falls from **408 MB**
(naive `read_csv`) to **149 MB**, a factor of 2.7; the in-memory payload falls
from **295.6 MB to 11.0 MB**, a factor of 27; disk from 19.4 GB to 341 MB.

> **Make the 2.7× versus 27× distinction yourself — it is the strongest point in
> this section.** Peak RSS only falls 2.7× because about 75 MB of it is the Python
> interpreter and imported libraries, which no optimisation removes. The payload
> figure is the one that matters, because it is the only one that *scales*: the
> naive path would need roughly `62 × 296 MB ≈ 18 GB` to hold all 62 days, while
> the streamed path stays at 11 MB no matter how many days you process. Across the
> full ingestion, peak RSS averaged 166 MB and never exceeded 241 MB.
>
> Also be ready for: *"why measure each variant in a separate subprocess?"*
> Because CPython does not generally return freed heap pages to the OS, so if you
> measured all four in one process, the later variants would inherit a heap the
> earlier ones had already grown and their peaks would look artificially similar.

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

> *Consequence:* a model that cannot reach 144 steps back cannot use "same time
> yesterday". That is the hypothesis the lookback and the TCN receptive field were
> designed to test — the TCN's receptive field is a hard architectural limit set by
> depth and kernel width, not a soft preference.
>
> **Important: do not claim this hypothesis was confirmed. It was not.** My
> controlled experiment found no effect of receptive field (see the results
> section below). Present this as the prediction the EDA motivated, then report
> that the experiment refuted it and explain why. That sequence is much stronger
> than pretending the EDA was straightforwardly vindicated, and it is the truth.

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

**Use the TCN receptive-field story as your reasoning-chain example.** It is the
best thing in the project because it shows you caught your own mistake.

> My first TCN search used a wall-clock budget per configuration, and it appeared
> to say the *smallest* receptive field was best — 61 steps, which provably cannot
> reach the lag-144 daily peak. That contradicted my EDA, so I looked at the cost
> column, and the result fell apart: only that one configuration ever trained to
> convergence. Every deeper configuration hit the time budget after 5 to 15 epochs.
> A TCN's cost per epoch grows with depth, so the budget was systematically
> under-training exactly the models I was trying to test. The experiment had
> confounded receptive field with training effort, so it could not answer the
> question I asked of it.
>
> So I designed a second experiment that removed the confound: same channel width,
> same batch size and learning rate, same lookback, the same number of epochs for
> every configuration, and early stopping disabled so no run could stop before
> another. Only depth varied.

Then give the outcome honestly:

> Validation MAE came out at 103.2, 109.6, 104.6 and 102.5 for receptive fields of
> 61, 125, 253 and 509 steps. That ordering is non-monotonic — the 125-step model
> is worse than both its neighbours. Since depth was the only variable and epochs
> were matched, nothing in the design can explain that, so run-to-run variation
> must be about 5 MAE units. The gap between the shortest and longest receptive
> field is 0.7 units, seven times smaller. So the honest conclusion is a negative
> one: over this range, receptive field has **no** measurable effect.
>
> And there is a good reason. Lag-1 autocorrelation is 0.987 while lag-144 is
> 0.878. At a one-step horizon the last few observations already carry almost all
> the signal — yesterday's value adds little that this morning's values do not
> already imply. The daily cycle is essential for a linear model that has to encode
> it explicitly, which is why seasonal differencing helps SARIMA, and nearly
> irrelevant to a network conditioned on recent history.

Then say what you selected and why, because this is a defensible judgement call:

> Since the four were statistically indistinguishable, I chose the shallowest —
> 5,601 parameters instead of 10,305 — rather than the nominal best. Picking the
> lowest validation MAE would have meant buying 84% more parameters on the strength
> of a 0.7-unit difference. I applied the same parsimony rule to SARIMA, and I
> implemented it in the tuning script rather than choosing by hand, so it is
> reproducible.

### The headline results — lead with the baseline argument

This is the most important 60 seconds of your video. Show
`results/metrics_square_*.csv` and `results/relative_to_persistence.csv`.

> Every model beats the seasonal-naive baseline by a wide margin, so every MASE is
> far below 1. If I stopped there I would conclude all three models succeeded. But
> seasonal naive is a *weak* baseline for this series — lag-1 autocorrelation is
> 0.987 against 0.878 at lag 144 — so the benchmark that actually binds is
> persistence: just repeating the last observation.
>
> Rescored against persistence: **SARIMA is 7.6% to 16.4% worse on all three
> areas** — it never beats it. The **TCN improves on persistence by 14.8%, 15.4%
> and 15.5%** — a spread of 0.007 across three areas that differ fivefold in
> traffic level and actually invert in weekend behaviour, and it was tuned on one
> area only and transferred unchanged. The **LSTM** has a similar mean, 0.905
> against 0.848, but a spread of 0.179 — **25 times larger**. It is the best model
> on one area and merely ties persistence on another.
>
> So the two neural models look comparable on mean accuracy and are not comparable
> at all on reliability. That distinction is the answer to the second half of my
> research question.

Cost, from `results/timing.csv` — and be candid about the caveat:

> SARIMA fits in about one second with 3 parameters. The TCN trains in 644 seconds
> on average with 5,601 parameters; the LSTM in 1,005 seconds with 17,217. So the
> TCN is more accurate *and* cheaper than the LSTM — that is architectural, because
> its convolutions run in parallel across the window while the LSTM has to step
> through 144 positions sequentially, which does not parallelise across two cores.
> Inference is under a millisecond per step for everything, so it is not a
> constraint.
>
> I have to flag that these training times are contaminated. The machine was also
> running my IDE and free RAM dropped to about 466 MB, and the same LSTM
> configuration cost 10.5 seconds per epoch during tuning and 88 seconds per epoch
> during the final run. So I trust the order-of-magnitude comparison and the
> TCN-versus-LSTM ordering, not the absolute numbers.

One more honesty point worth volunteering before you are asked:

> One final run — the LSTM on square 5161 — hit its wall-clock budget and stopped
> at epoch 22 of 40. Its best epoch, 18, matches what I found in tuning, so it had
> most likely converged, but I cannot assert it. And it happens to be the one area
> where the LSTM fails to beat persistence, so I flag it rather than let it pass.

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

Then interpret, using the real numbers. There are two findings and they build on
each other.

**Finding 1 — the models are only different at the peak.** Show
`error_by_hour_sq5161.png`.

> Between midnight and 6 a.m. all three models sit between roughly 15 and 45 MAE
> and are indistinguishable. Between 1 and 4 p.m., when traffic peaks, SARIMA rises
> to about 258 and the LSTM to about 256, while the TCN stays near 163. So the
> TCN's entire 15% advantage is earned at the daily peak — it is not a uniformly
> better model. And because the peak dominates mean absolute error, that is enough.
> This also explains why every model's RMSE-over-MAE ratio is above 1.4: the error
> is concentrated in a few large misses, not spread evenly.

**Finding 2 — the worst window is a calendar effect.** Show
`failure_window_sq5161.png`. Explain the selection method first, then the cause.

> The worst 6-hour window in every one of the three areas is the midday-to-evening
> peak, at 1.8 to 2.8 times the weekly mean error. For square 5161 it is Saturday
> 21 December, 13:10 to 19:00. Within the test week, weekday peaks run 3,057 to
> 3,877, then Saturday jumps to 5,238 and Sunday to 5,496 — this area's weekend
> traffic is 1.38 times its weekday level. Since error scales with traffic level,
> the weekly maximum is where the worst window has to fall.
>
> **Be careful here, and say the negative explicitly — it is a stronger answer.**
> This is *not* a case of extrapolating beyond anything the models have seen. My
> training period contains a daily peak of 8,044 on Saturday 2 November, and
> Saturdays at 6,348 and 6,153 in late November and early December, against a
> training median daily peak of 3,815. So 21 December at 5,238 is a *lower*
> Saturday than several in training. The pre-Christmas surge explanation is
> tempting and my own STL trend actually *falls* after 21 December, so I checked it
> and dropped it.
>
> What really breaks is the **day-of-week transition**: Friday peaks at 3,530 and
> Saturday at 5,238, a 48% jump between consecutive days.

**The three models fail in three different ways — this is the best detail you
have.** Read `window_bias` from `results/failure_analysis_sq5161.csv`:

> SARIMA under-predicts systematically, bias **−207**: it spends the afternoon
> below the observed series. That is structural, and it is the cleanest example in
> my whole project of a model's assumptions showing up in its errors. Its only
> seasonal term is a difference at lag 144 — one day — so its forecast is anchored
> to the same time yesterday, and yesterday was a Friday peaking 48% lower. That
> model has **no representation of day-of-week at all**, so it must under-predict
> every Friday-to-Saturday transition in this area and over-predict every
> Sunday-to-Monday one. The weekly cycle I found in the periodogram is exactly the
> structure it omits.
>
> The LSTM over-predicts, bias **+112**: it tracks the ascent but stays high as
> traffic falls after 5 p.m., overshooting the descent by up to about 500 units.
> Having compressed a 144-step window into a fixed-size hidden state, it reproduces
> a typical peak shape and is late to follow an unusually steep decline.
>
> The TCN is nearly unbiased, **+15**, with the lowest window error. Its errors
> there are variance rather than a systematic misreading of the day.

Then tie it back, because this is the payoff:

> That ordering is consistent with my receptive-field finding. The models that lean
> hardest on periodic structure — SARIMA explicitly, the LSTM through a 144-step
> window it has to compress — are the ones misled when the period breaks. The TCN I
> selected cannot even see a full day, so it has less periodic prior to be wrong
> about and leans on the immediately preceding observations, which on an anomalous
> day are the more reliable evidence.

**State the limitation this exposes**, and volunteer it rather than waiting:

> Every model here is univariate, and none of them gets a **day-of-week** input.
> My own EDA documents that effect — a weekly peak in the periodogram, and
> weekend-to-weekday ratios ranging from 0.43 to 1.38 across areas — so the models
> can only infer it indirectly, and SARIMA cannot infer it at all. One categorical
> feature is the information needed for the single largest failure in my
> evaluation. That is a limitation of my experimental design, not of the
> architectures, and it is the first thing I would add.

---

## 8:30–9:15 — Conclusion, limitations, future work

Close with three claims, each with its evidence attached:

> One: the choice of baseline decides what the results mean. All models beat
> seasonal naive, but SARIMA loses to persistence on all three areas.
> Two: the TCN is the most reliable model — a 15% gain on persistence with a spread
> of 0.007 across three structurally different areas, at a third of the LSTM's
> parameters and two-thirds of its training time.
> Three: the strong daily seasonality is real but largely redundant at a one-step
> horizon, which is why receptive field had no measurable effect.

Then be specific about limitations. Vague ones read as filler; these are real and
each one is tied to something in your results:

* **Single seed per configuration.** My own receptive-field study implies about 5
  MAE units of run-to-run variation, which is comparable to some differences I
  discuss. The cross-area *consistency* result is safe — a spread of 0.007 across
  three areas is not luck — but the 1.6-unit LSTM-versus-TCN gap on square 5059 is
  not resolved by my evidence.
* **Wall-clock budgets make the pipeline non-reproducible in the strict sense**,
  even with the seed fixed, because a differently loaded machine stops at a
  different epoch.
* **Timing measurements are contaminated by machine load** (10.5 versus 88 seconds
  per epoch for the same configuration).
* **Three areas, all high-traffic and geographically adjacent** in the historic
  centre, out of 10,000. I have no evidence about a sparse suburban cell, where
  zero intervals are common and persistence may be even harder to beat.
* **SARIMA was restricted to `P = Q = 0`** for computational reasons, so I did not
  fully explore that model class — read its poor showing with that in mind.
* **Univariate only** — and the largest failure is a calendar effect.
* **Symmetric error metrics.** Over- and under-provisioning cost an operator
  different amounts, so MAE would not rank these models the way a cost function
  would. SARIMA's systematic under-prediction at peaks would be punished much
  harder under an asymmetric cost.

Future work, ordered by return on effort — one sentence each: **calendar features**
first, because they target the biggest observed failure directly; **repeat across
seeds** to turn the LSTM-versus-TCN comparison from suggestive into supported; a
**horizon sweep** at 1, 6, 36 and 144 steps to test the prediction that receptive
field and seasonality should matter progressively more as the horizon grows;
**areas sampled across the traffic distribution** to test whether the TCN's
stability holds outside the busy centre; **spatial models** to exploit inter-cell
correlation; and an **asymmetric training cost** to align the objective with the
operational use case.

---

## The hardest questions you are likely to get

Rehearse these until the answers are yours. They are the places where the work is
genuinely open to challenge.

**"Your SARIMA lost to a one-line baseline. Did you implement it wrong?"**
No, and I can show why. The order search tested eight configurations bounded by the
PACF, which is 0.99 at lag 1, 0.26 at lag 2 and 0.04 at lag 3, so low orders are
the right region. Validation MAE flattened at 136–137 across six of the eight
configurations, well inside the search space rather than at its edge, so I was not
cut off by the boundary. The reason it loses is structural: seasonal differencing at
lag 144 is what makes the series stationary and therefore linearly modellable, but
it also discards the short-range information that dominates at a ten-minute horizon.
Stationarity is not the same thing as predictability. I did restrict it to
`P = Q = 0`, which I state as a limitation.

**"You chose the TCN configuration that did *not* have the best validation score.
Isn't that cherry-picking?"**
It is the opposite — cherry-picking would be taking the 0.7-unit win. My own
experiment shows run-to-run variation of about 5 units, because the 125-step model
came out worse than both its neighbours when depth was the only variable and epochs
were matched. A 0.7-unit difference inside 5 units of noise is not a result. So I
took the model with 5,601 parameters over the one with 10,305. I applied the same
rule to SARIMA, and it is coded in `scripts/04_tune.py`, not applied by hand.

**"Your MASE isn't the standard MASE."**
Correct, and I say so in the report. Hyndman and Koehler scale by the *in-sample
one-step naive* error; I scale by the *seasonal-naive* error on the same evaluation
week. I did that because "same time yesterday" is the meaningful competitor for a
strongly daily series. The consequence is that my MASE is a more lenient bar than
the published version — the canonical denominator would have been persistence,
which is the harder baseline — and my values are not directly comparable with MASE
figures in other papers. That is exactly why I also report every result as a ratio
to persistence, and it is that table, not MASE, that my conclusions rest on.

**"Why should I believe the TCN is better when you ran each model once?"**
For the cross-area consistency claim, because a spread of 0.007 across three areas
that differ fivefold in level and invert in weekend behaviour is not plausibly
three lucky draws. For a single pairwise comparison — the TCN versus the LSTM on
square 5059, a gap of 1.6 MAE — you should not believe it, and I say so in the
limitations. Repeating across five seeds is the second item on my future-work list
for exactly that reason.

**"Isn't 0.987 lag-1 autocorrelation just telling you the problem is trivial?"**
It tells me the problem has a very strong baseline, which is why I refused to
report only MASE. It is not trivial: persistence still has an MAE of 76 to 93,
which is 8–9% MAPE, and the errors are concentrated exactly where an operator cares
— the daily peak. The TCN's 15% improvement is earned entirely in those hours.

**"You said the EDA justified a 144-step lookback, then found receptive field
doesn't matter. Which is it?"**
Both, and the tension is the interesting part. The lag-144 dependency is real —
autocorrelation 0.878, a periodogram peak at exactly 24.000 hours. It is decisive
for SARIMA, which has three parameters and must encode the cycle explicitly. It is
nearly irrelevant to a network that already sees the last several hours, because at
a one-step horizon those observations already imply where in the daily cycle you
are. The EDA correctly identified the structure; my initial inference that every
model therefore *needs* to reach lag 144 was the part the experiment refuted.

**"Hasn't a TCN already been applied to this dataset? What's new here?"**
Yes, and I cite it — Zhang et al. in PLOS ONE 2023 combine a TCN with an attention
module and a Transformer on the Milan data and report a 51% MAE reduction against
an LSTM. Two things separate that from mine and I say so in the report. Theirs is a
3.19-million-parameter spatio-temporal hybrid taking the full 100×100 grid as
input; mine is a 5,601-parameter univariate model on one cell's own history. More
importantly, **every baseline in that paper is another neural network** — so its
percentages show the hybrid beating other deep models, not beating "repeat the last
value". Across all the Milan work I reviewed, I found no study reporting a
persistence or seasonal-naive baseline for single-cell one-step-ahead traffic. That
is the gap my report fills, and it is why my headline finding is a comparison
nobody had published for this data.

**"How do you know your timezone and grid orientation are right?"**
I verified both rather than assuming. The city-wide mean profile bottoms out at
05:00 and peaks at 13:00 local time, and weekend activity is 0.83 of weekday — both
are what human activity should produce, and a wrong offset would have shifted them
visibly. For orientation, I correlated cell centroids from the official
`milano-grid.geojson` against `(id-1)//100` and `(id-1)%100` and got r = 1.000 for
both, so rows run south-to-north and columns west-to-east and my maps are north-up.
A silent vertical flip would have inverted every geographic interpretation.

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
