# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

One-step-ahead forecasting of cellular Internet traffic on the Telecom Italia
Milan grid, comparing a **seasonal ARIMA**, an **LSTM** and a **temporal
convolutional network** across geographical areas with different traffic
characteristics.

**Repository:** https://github.com/Emmanuel-NS/Time-series-forecasting-FA

> **Still to do before submission:** record the video and add its URL — see
> [`report/SUBMIT.md`](report/SUBMIT.md).


> **Research question.** How do different sequential models compare for
> one-step-ahead mobile network traffic forecasting, and how does their
> performance vary across geographical areas with different traffic
> characteristics?

---

## 1. What this project does

| Stage | Script | Output |
|---|---|---|
| Ingest 20.6 GB of raw text into a 341 MB store | `scripts/01_ingest.py` | `data/processed/`, `results/ingest_log.csv` |
| Measure the memory cost of four loading strategies | `scripts/02_memory_benchmark.py` | `results/memory_benchmark.csv` |
| Exploratory and time-series analysis | `scripts/03_eda.py` | `figures/eda_*.png`, `results/eda_*.csv` |
| Iterative hyperparameter experiments | `scripts/04_tune.py` | `results/tuning_*.csv`, `results/tuned_config.json` |
| Final evaluation on 16–22 Dec 2013 | `scripts/05_final_experiments.py` | `figures/forecast_*.png`, `results/metrics_square_*.csv`, `results/timing.csv` |

---

## 1b. Key results

Test week 16–22 December 2013. MAE **relative to the persistence baseline**
(`x̂(t+1) = x(t)`), so 1.00 means "no better than repeating the last observation"
and lower is better. Full tables in `results/metrics_square_*.csv`.

| Model | sq 5161 | sq 5059 | sq 5259 | mean | spread |
|---|---|---|---|---|---|
| Seasonal naive | 3.649 | 2.107 | 6.191 | 3.982 | 4.084 |
| SARIMA(1,0,1)(0,1,0)[144] | 1.164 | 1.095 | 1.076 | 1.112 | 0.088 |
| LSTM (64 units, lookback 144) | 1.003 | 0.825 | 0.886 | 0.905 | 0.179 |
| **TCN (16 ch, 4 levels)** | **0.846** | 0.845 | **0.852** | **0.848** | **0.007** |

Four things are worth pulling out, because they are not what the required metrics
alone would suggest:

* **Seasonal naive is a weak baseline for this series**, so the required MASE
  values (all far below 1) flatter every model. Lag-1 autocorrelation is 0.987
  against 0.878 at lag 144, which makes **persistence** the benchmark that
  actually binds.
* **SARIMA never beats persistence** — it is 7.6–16.4% worse on all three areas.
  It is not a cheap approximation to the networks; it is a worse forecaster that
  happens to fit in about one second with three parameters.
* **The TCN is the most reliable model.** It improves on persistence by 14.8–15.5%
  with a spread of 0.007 across areas that differ fivefold in traffic level and
  invert in weekend behaviour, having been tuned on one area only. The LSTM's mean
  is similar but its spread is 25× larger.
* **Receptive field did not matter.** A controlled study with matched epochs found
  no measurable effect between 61 and 509 steps. At a one-step horizon the strong
  daily cycle is largely redundant — recent observations already imply it.

---

## 2. Dataset

**Telecom Italia Big Data Challenge 2014 — Milan.** Milan is divided into a
100 × 100 grid of 10,000 square areas; telecommunications activity is aggregated
into 10-minute intervals from 1 November 2013 to 1 January 2014 (62 days,
8,928 intervals). Each raw row is one
`(square_id, interval, country_code)` combination with SMS, call and Internet
activity columns.

* Publication: Barlacchi *et al.*, *A multi-source dataset of urban life in the
  city of Milan and the Province of Trentino*, Scientific Data 2, 150055 (2015).
* Files: <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV>

### Getting the data

The dataset is public but Harvard Dataverse attaches a **download guestbook**
(id 96, "Privacy risk assessment"). Since Dataverse 6.10 the plain
`GET /api/access/datafile/{id}` endpoint returns `HTTP 400` unless a guestbook
response accompanies the request, *even for an authenticated user*. The
pipeline handles this automatically using the documented signed-URL flow, but it
needs a free API token:

1. Register or sign in at <https://dataverse.harvard.edu> (any email address; no
   institutional affiliation required).
2. Account menu → **API Token** → **Create Token**.
3. `cp .env.example .env` and paste the token into `DATAVERSE_API_TOKEN`, then
   fill in `DATAVERSE_EMAIL` (the guestbook requires an email address).

`.env` is git-ignored and is never committed.

If you already have the raw `.txt` files, point `MILAN_RAW_DIR` at their
directory instead and no download will occur. Files found locally are **never
deleted** by the pipeline; only files it downloaded itself are discarded after
aggregation.

---

## 3. Setup

Requires Python 3.10+ (developed on 3.12).

```bash
git clone https://github.com/Emmanuel-NS/Time-series-forecasting-FA.git
cd Time-series-forecasting-FA

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` pulls the default PyTorch wheel. On a machine without an
NVIDIA GPU the CPU-only build is smaller and sufficient — this project was
developed and timed entirely on CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify the installation without touching the dataset:

```bash
python -m pytest tests -q      # 27 tests, ~3.5 minutes on 2 CPU cores
```

---

## 4. Reproducing the results

```bash
python scripts/01_ingest.py              # ~2 h: downloads 19.4 GB, keeps 341 MB
python scripts/02_memory_benchmark.py    # ~3 min, needs one raw .txt present
python scripts/03_eda.py                 # ~5 min
python scripts/04_tune.py                # ~1.5 h
python scripts/04_tune.py --models tcn-rf  # ~1.5 h: controlled receptive-field study
python scripts/05_final_experiments.py   # ~1.5 h
python scripts/06_build_report.py        # ~1 min: report.md -> report.pdf
```

Timings are for the 2-core laptop described in Section 9 and are **highly
sensitive to competing load** — the same neural configuration was measured at
10.5 s and 88 s per epoch depending on what else was running. Close other
applications before running stages 4 and 5 if you intend to compare timings.

Stage 1 is **idempotent and resumable**: it records completed days in a manifest
and skips them, so an interrupted download can simply be restarted. Useful
flags:

```bash
python scripts/01_ingest.py --limit 3                    # smoke test
python scripts/01_ingest.py --days 2013-12-16,2013-12-17 # reprocess specific days
python scripts/01_ingest.py --force                      # rebuild everything
```

All randomness is seeded (`RANDOM_SEED = 42` in `src/config.py`) and PyTorch is
pinned to 2 threads, so the neural results and the reported timings reproduce on
the same hardware.

---

## 5. Data handling and memory management

The central constraint is that the raw dataset (19.38 GB across 62 files,
319,896,289 rows) is roughly 2.5× the host's total RAM (7.9 GB usable) and more
than an order of magnitude larger than the memory actually free. Five decisions
make the problem fit.

**1. Aggregate away the unused dimension during parsing.** The forecasting task
is defined per geographical area, so country code is summed out as each chunk is
read. This alone reduces 319,896,289 rows to `62 × 144 × 10,000 = 89,280,000`
cells, a factor of 3.6, and it is the largest single reduction available because
it removes a dimension rather than compressing one.

**2. Project to the target column.** Only `square_id`, `time_interval` and
`internet` are parsed via `usecols`; the four SMS and call columns are never
allocated.

**3. Stream in fixed-size chunks.** Each day is read in 1-million-row chunks and
folded into an accumulator, so peak memory is a function of the chunk size, not
the file size.

**4. Store the result densely as `float32`, and read it back memory-mapped.**
The grid is complete (1,439,8xx of 1,440,000 cells non-zero every day), so a
dense `(8928, 10000)` matrix wastes nothing and needs no index. At `float32`
that is **341 MB**; extracting one area's full series through `numpy.memmap`
touches only 35 KB of resident memory.

**5. Download, process, discard.** Raw files are fetched one at a time and
deleted immediately after aggregation, so peak disk use is
`store + one raw day ≈ 720 MB` rather than 20.6 GB.

Accumulation is done in `float64` and cast to `float32` only once a day is
complete, so summing many small activity values does not accumulate rounding
error while the stored result stays compact.

Measured outcome (see `results/memory_benchmark.csv` for the full table and
`results/ingest_log.csv` for per-day evidence). Four loading strategies were
timed on the same 308 MB day file, each in a **separate subprocess** so that one
variant's heap growth cannot flatter the next:

| Variant | Peak RSS | In-memory payload | RSS reduction |
|---|---|---|---|
| naive `read_csv` | 408.0 MB | 295.6 MB | — |
| + narrow dtypes | 273.9 MB | 157.0 MB | 32.9% |
| + column projection | 165.2 MB | 73.9 MB | 59.5% |
| + chunked streaming | **148.9 MB** | **11.0 MB** | **63.5%** |

Peak RSS falls by 2.7×, but the payload — the data structure itself — falls by
27×. The gap is a ~75 MB interpreter-and-libraries baseline that no optimisation
removes. The payload figure is the one that matters, because it is the only one
that does not grow with the dataset: the first three variants would need roughly
`62 × 296 MB ≈ 18 GB` to process all 62 days at once, while the streamed path
stays at 11 MB regardless of how many days are processed.

Across the full 62-day ingestion, peak RSS averaged 166 MB and never exceeded
**241 MB**, against a 1 GB self-imposed budget. Disk footprint falls from
19.38 GB of raw text to a 341 MB store.

### Trade-offs and limitations

* **Only Internet activity is retained.** Keeping all five activity types would
  make the store 1.7 GB and rule out several in-memory analyses. The cost is
  that a multivariate model using SMS or voice traffic as exogenous inputs would
  require re-running ingestion.
* **The `float32` store loses precision.** Roughly seven significant decimal
  digits remain, which is far below the measurement noise of the activity
  index, but it does mean the store is not a bit-exact copy of the source.
* **The dense layout assumes a complete grid.** This holds for Milan; a sparse
  or ragged grid would waste space and a sparse format would be preferable.
* **Re-ingestion is expensive.** The optimisation is one-directional: any
  analysis needing a discarded column requires downloading 20.6 GB again.
* **Parallel range requests are a workaround, not a right.** Harvard throttles
  each connection to ~0.17 MB/s for this dataset while the host link sustains
  over 5 MB/s, so the downloader issues 8 concurrent range requests per file
  (~4 MB/s aggregate, a 23× improvement). The worker count is deliberately
  modest and configurable via `MILAN_DOWNLOAD_WORKERS` out of courtesy to a
  shared public archive.

---

## 6. Repository layout

```
.
├── src/
│   ├── config.py            # paths, dataset constants, split definitions
│   ├── profiling.py         # peak-RSS probe and hardware summary
│   ├── ingest.py            # guestbook auth, parallel download, streaming aggregation
│   ├── memory_benchmark.py  # four loading strategies, measured in subprocesses
│   ├── data.py              # memmap access, splits, scaling, windowing
│   ├── metrics.py           # MAE, RMSE, MAPE, sMAPE, MASE, R^2
│   ├── eda.py               # exploratory and time-series analysis
│   ├── experiments.py       # tuning harness, final evaluation, figures
│   └── models/
│       ├── base.py          # shared Forecaster interface
│       ├── baselines.py     # persistence, seasonal naive
│       ├── sarima.py        # SARIMA(p,d,q)(0,1,0)[144]
│       └── neural.py        # LSTM and TCN + shared training loop
├── scripts/                 # numbered pipeline stages
├── tests/                   # 23 unit and smoke tests
├── results/                 # CSV tables, logs, tuned config, hardware record
├── figures/                 # all report figures
└── report/                  # research report sources
```

---

## 7. Experimental design

**Chronological split — never shuffled.** Shuffling would let a model see the
future, which invalidates any forecasting evaluation.

| Partition | Dates | Slots | Purpose |
|---|---|---|---|
| Train | 1 Nov – 8 Dec 2013 | 5,472 | parameter estimation and scaler statistics |
| Validation | 9 – 15 Dec 2013 | 1,008 | hyperparameter selection, early stopping |
| Test | **16 – 22 Dec 2013** | 1,008 | reported results, read once |

Validation and test windows may read *history* from the preceding partition,
because at forecast time `t+1` the observations up to `t` genuinely are
available. What never happens is a *target* from a later partition influencing
training, hyperparameter choice, or the scaler statistics.

**Tuning is performed on the highest-traffic area only**, then the selected
configuration is applied unchanged to all three areas. Comparing across areas
therefore measures generalisation rather than per-area overfitting.

**Metrics.** MAE, RMSE and MAPE are required by the assignment. MAPE is
unreliable here because night-time activity approaches zero, so sMAPE (bounded)
and MASE (scaled against the seasonal-naive forecast) are reported alongside it.
All errors are computed in original activity units, after inverting the `log1p`
transform. `RMSE/MAE` is reported because a ratio well above 1 signals that a
model's error is dominated by a few large misses at traffic peaks.

---

## 8. Models

| Model | Family | How it aggregates history | Key hyperparameters |
|---|---|---|---|
| SARIMA(p,d,q)(0,1,0)[144] | linear statistical | explicit seasonal differencing + ARMA terms | `p`, `d`, `q` |
| LSTM | recurrent neural | gated sequential state over the window | lookback, hidden size, layers |
| TCN | convolutional neural | stacked dilated causal convolutions | lookback, channels, levels, kernel |

`Persistence` (`x̂(t+1) = x(t)`) and `Seasonal naive` (`x̂(t+1) = x(t+1−144)`)
are reference baselines, not competitors: they exist so the learned models'
errors can be interpreted, via MASE.

The seasonal ARIMA applies its seasonal difference **explicitly** rather than
through `statsmodels`' state space. A full `SARIMAX(...)(P,D,Q)[144]` carries
~144 state variables, making each Kalman pass over 8,000 observations minutes
long on a 2-core CPU and a grid search unaffordable. Differencing
`log1p` traffic at lag 144 and fitting an `ARIMA(p,d,q)` to the result is the
same model, but the state dimension depends only on `max(p, q+1)`.

All models train on `log1p`-transformed, standardised targets (statistics fitted
on the training partition only) and are evaluated after inverting the transform.
The neural models use a Huber loss rather than MSE, because the series contains
isolated order-of-magnitude spikes that would otherwise dominate the gradient.

---

## 9. Hardware

All timings in the report were recorded on the machine described in
`results/hardware.json`:

* Intel Core i5-6300U @ 2.40 GHz, 2 physical cores / 4 threads
* 8 GB RAM, Windows 10 (19045)
* Python 3.12, PyTorch 2.14 CPU build, pinned to 2 threads
* No GPU

Training time is wall-clock for a complete fit including early stopping.
Inference time is the median of 5 repeated one-step-ahead passes over all 1,008
slots of the test week, averaged over the three study areas.

---

## 10. Use of AI assistance

Disclosed in Section 8 of the report. An AI coding assistant (Cursor) was used for
code scaffolding, debugging (including Dataverse access), literature search, and
report drafting. Every reported number comes from code in this repository. Design
decisions, experiment interpretation, and conclusions are the author's
responsibility; see the report for the full specific disclosure.

Submission steps that still need the author (video + URL): `report/SUBMIT.md`.


---

## 11. References

See the report's reference list. Primary sources:

1. G. Barlacchi *et al.*, "A multi-source dataset of urban life in the city of
   Milan and the Province of Trentino," *Scientific Data*, vol. 2, 150055, 2015.
2. S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural
   Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
3. S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic
   convolutional and recurrent networks for sequence modeling," arXiv:1803.01271,
   2018.
