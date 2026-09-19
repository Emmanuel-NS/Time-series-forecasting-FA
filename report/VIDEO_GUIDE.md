# Video guide (7 to 10 minutes)

This is what you still need to record. The report and code are finished.
After you upload the video, put the link in reference [34] of `report/report.md`
and follow `report/SUBMIT.md`.

**Do not read this aloud word for word.** Use it as a checklist. Speak in your
own words. The marker wants to hear that you understand *your* pipeline and
*your* numbers.

---

## What the video must cover (rubric)

1. The problem and research question  
2. How you handled the large dataset (memory)  
3. Which models you chose and why  
4. Main results (use persistence, not only MASE)  
5. **One technical decision** in depth (recommended: TCN receptive-field study, or SARIMA seasonal difference by hand)  
6. **One limitation or failure** (recommended: Friday to Saturday jump / missing day-of-week)  

Show real files: code, CSVs, and figures from this repo.

---

## Timing plan

| Time | Topic | Show on screen |
|---|---|---|
| 0:00 to 0:40 | Who you are, problem, research question | Title / README |
| 0:40 to 2:10 | Data and memory | `src/ingest.py`, `results/memory_benchmark.csv` |
| 2:10 to 3:30 | EDA that changed design | `figures/eda_01`, `eda_05` |
| 3:30 to 4:30 | Three models + baselines | short table or `src/models/` |
| 4:30 to 6:00 | Tuning and final results | `results/relative_to_persistence.csv`, one forecast plot |
| 6:00 to 7:20 | Deep dive (technical decision) | `src/models/sarima.py` or TCN RF CSV |
| 7:20 to 8:40 | Failure case | `figures/failure_window_sq5161.png` |
| 8:40 to 9:30 | Conclusion + limitation | summary |

Aim for about 8 to 9 minutes. Practise once with a timer.

---

## Opening (about 40 seconds)

Say your name and that this is Formative Assignment 1.

Then, in plain language:

> I forecast Internet traffic for small areas of Milan, one step ahead, which is
> the next 10 minutes. The question is how SARIMA, an LSTM, and a TCN compare,
> and whether that ranking stays the same when the area changes.

Setup facts (say them once):

- 10,000 grid squares, 10-minute slots, 62 days  
- Test week: 16 to 22 December 2013  
- Three evaluation areas: 5161, 5059, 5259 (tuned only on 5161)  
- Repo: https://github.com/Emmanuel-NS/Time-series-forecasting-FA  

---

## Data handling (about 90 seconds)

Start with the constraint:

> The raw data is about 19.4 GB and over 300 million rows. My laptop has about
> 8 GB of RAM and two CPU cores. I cannot load the whole dataset at once.

Then list the five ideas, while scrolling `src/ingest.py`:

1. Sum over country codes while reading  
2. Keep only square, time, and internet columns  
3. Read in chunks of one million rows  
4. Save a dense float32 memmap (341 MB)  
5. Process one day and delete the raw file when the script downloaded it  

Quote these numbers from `results/memory_benchmark.csv`:

- Peak RSS: **408 MB** down to **149 MB** (about 2.7 times)  
- Payload: **296 MB** down to **11 MB** (about 27 times)  
- Full ingest peak never above **241 MB**  

**Explain the 2.7 vs 27 point** (markers like this):

> Peak memory only falls by about 2.7 times because Python itself still uses
> tens of megabytes. The payload, the actual table in memory, falls by about
> 27 times. That is what scales when you process more days.

**Be ready if asked:** why separate subprocesses for the memory test? Because
Python often keeps freed heap, so running all four loaders in one process would
make the later ones look unfairly similar.

Honest trade-off: you only kept Internet traffic. Re-adding SMS or calls means
re-downloading.

---

## EDA (about 80 seconds)

Show `figures/eda_01_spatial_distribution.png`:

> Traffic is very uneven. The top 10% of areas carry almost half the city total.
> The busiest cells sit near the Duomo, so I evaluate three busy areas, not one.

Show `figures/eda_05_autocorrelation.png`:

> Lag-1 autocorrelation is 0.987, so the last value is already a strong forecast.
> Lag 144 (one day) is 0.878. That is why persistence is my hard baseline, and
> why I considered a 144-step lookback for the neural models.

One sentence on land use (optional but strong):

> Square 4159 near Bocconi looks like a campus. Square 4556 near Navigli looks
> like nightlife. Same city, different shapes, so I scale each area on its own
> training data.

---

## Models (about 60 seconds)

| Model | Why you chose it |
|---|---|
| SARIMA | EDA supports seasonal differencing; fair linear baseline |
| LSTM | Standard recurrent model for this kind of series |
| TCN | Parallel convolutions; fixed receptive field; contrast to LSTM |

Also say:

> I also report persistence and seasonal naive. They are not the main models.
> They tell me whether the others are worth the complexity.

One technical detail for SARIMA (short version):

> A full seasonal state-space model with period 144 is too slow on my machine,
> so I difference at lag 144 myself and fit a small ARIMA. That is still
> SARIMA with seasonal D = 1 and P = Q = 0. I say that limit openly.

---

## Results (about 90 seconds)

**Lead with the baseline story.** This is the most important result in the project.

Open `results/relative_to_persistence.csv` (or the table in the PDF):

> If I only look at MASE against seasonal naive, every model looks great.
> Seasonal naive is weak here. Persistence is the baseline that matters.

Then say the three numbers in your own words:

1. **SARIMA** is worse than persistence on all three areas (about 8 to 16%).  
2. **TCN** improves on persistence by about 15% on every area, with almost the same ratio each time (spread 0.007).  
3. **LSTM** is close on average, but the gain jumps around a lot more (spread about 0.179).  

Show one forecast figure (for example `figures/forecast_sq5161_TCN.png`):

> Visually they all track the daily cycle. The difference is at the peaks, which
> is also where absolute error is largest.

Cost, briefly:

> SARIMA trains in about a second. The TCN is slower but still cheaper than the
> LSTM on my machine, with fewer parameters. My timing numbers are noisy because
> the laptop was busy, so I only trust the order of magnitude.

---

## Deep dive: pick ONE (about 80 seconds)

### Option A (recommended): TCN receptive-field study

Show `results/tuning_tcn_sq5161.csv` then `results/tuning_tcn_receptive_field_sq5161.csv`.

Script you can adapt:

> My first TCN search made the shortest receptive field look best. That
> contradicted the daily lag in the ACF. When I checked the logs, only the small
> model finished training. The deeper ones hit a time limit after fewer epochs.
> So receptive field and training effort were mixed.
>
> I ran a second study with the same number of epochs for every depth and no
> early stopping. The errors were 103, 110, 105, and 102. The middle one was
> worst, so noise is at least a few MAE units. The best and worst differ by less
> than one unit. At one step ahead, receptive field did not clearly matter.
>
> I kept the shallow TCN (4 levels) because it is simpler and just as good within
> that noise. At this horizon the last few observations already carry most of
> the signal, because lag-1 correlation is 0.987.

### Option B: SARIMA explicit seasonal difference

Show `src/models/sarima.py`.

> Period 144 makes a full seasonal state-space model expensive. I difference by
> hand, fit ARIMA, then add yesterday back when I invert. Same model class with
> P = Q = 0. Unit tests check the difference and inverse.

---

## Failure case (about 80 seconds)

Show `figures/error_by_hour_sq5161.png`:

> At night the three models are close. In the afternoon the TCN is clearly better.
> So the TCN gain is mostly a peak-hour gain.

Show `figures/failure_window_sq5161.png`:

> The worst six hours on square 5161 are Saturday 21 December afternoon.
>
> I first thought this might be a Christmas peak the models never saw. I checked
> the data. Training already has higher Saturday peaks, up to about 8,000.
> Saturday 21 is about 5,200. So it is not out of range.
>
> What hurts is the jump from Friday to Saturday: about 3,530 to 5,238, a 48%
> rise. None of my models gets day-of-week as an input. SARIMA is tied to
> "yesterday," so it under-predicts (bias about -207). The LSTM overshoots the
> fall (bias about +112). The TCN is nearly unbiased (about +15).

Say the limitation clearly:

> The next improvement I would try is a day-of-week feature. That matches the
> failure I actually observed.

---

## Close (about 50 seconds)

Three takeaways:

1. Persistence, not seasonal naive, is the fair baseline here.  
2. The TCN is the most reliable across areas.  
3. At one step ahead, a long receptive field did not help; day-of-week missingness did hurt.

One limitation:

> I only studied three busy central squares, with one random seed for the neural
> runs, and some training times were cut by a clock limit on a busy laptop.

Thank the viewer. End.

---

## Hard questions (rehearse these)

**"SARIMA lost to a one-line baseline. Is your SARIMA wrong?"**  
No. Eight orders were tried. Validation flatlined. Seasonal differencing helps stationarity but removes short-term information that persistence keeps. At 10-minute horizon that matters.

**"Why keep the TCN that was not the absolute lowest validation MAE?"**  
The controlled study is noisy (middle depth worst). Best and worst differ by 0.7 MAE while the noise floor is about 5. I chose the shallow model for simplicity. That rule is in `scripts/04_tune.py`.

**"Is your MASE the textbook MASE?"**  
Not exactly. Textbook MASE often scales by in-sample one-step naive error. I scale by seasonal-naive error on the same evaluation week. I say that in the report. That is why I also report ratios to persistence.

**"Has a TCN already been used on Milan data?"**  
Yes. Zhang et al. (2023) use a large spatio-temporal TCN hybrid and compare to other neural models. Mine is a small univariate TCN compared to persistence. Different question.

**"Did AI write this project?"**  
Keep this consistent with Section 8: you used docs and papers to learn; any assistant help was for clarifying APIs or language; you own the design, experiments, and conclusions, and you can explain them.

---

## Recording tips

- 1080p, large font in the editor so code is readable.  
- Open tabs before you start: `src/ingest.py`, `src/models/sarima.py`, `results/`, `figures/`.  
- Say numbers slowly. Prefer "about fifteen percent" plus the exact figure once.  
- If something is imperfect (truncated LSTM run, noisy timings), say it. That looks better than hiding it.  
- Smile less important than clarity. Sound like you are explaining to a classmate.

---

## After recording

1. Upload the video (Drive / YouTube unlisted / whatever the course asks).  
2. Put the URL in `report/report.md` reference [34].  
3. Run `python scripts/06_build_report.py`.  
4. Commit and push under your name (commands in `report/SUBMIT.md`).  
5. Submit `report/report.pdf`.
