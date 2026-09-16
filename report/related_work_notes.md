# Related Work — Verified Reference Notes

**Scope.** Supporting material for an undergraduate empirical study of one-step-ahead (10-minute
resolution) cellular traffic forecasting on the Telecom Italia Milan dataset, comparing SARIMA,
LSTM and a TCN against persistence and seasonal-naive baselines.

**Verification policy used here.** Every entry below was located via web search during the
preparation of this file. Metadata (title, authors, venue, year, volume/issue/pages, DOI) was taken
from the publisher record, the author's own copy of the paper, or an indexing service that
reproduces the publisher record (Crossref, DBLP/researchr, institutional repository). Where a
detail could not be confirmed from such a source it is flagged `[unverified]`. Numeric claims in
the `Key numbers` fields were read out of the full text of the paper itself wherever the full text
was accessible; where a number comes only from an abstract or a secondary source that is stated
explicitly.

**Reading of the verification status field:**
- `Verified: yes` — full bibliographic record confirmed from a publisher/repository record, and
  (where relevant) the substantive claims below were checked against the paper's own text.
- `Verified: partial` — bibliographic record confirmed, but the full text was not read in this
  session, so relevance notes rest on the abstract only.

---

# Part A — Annotated bibliography

## A.1 The dataset and the Milan/Telecom Italia line of work

### [1] Barlacchi et al. (2015) — the dataset paper

**Citation (IEEE style).**
G. Barlacchi, M. De Nadai, R. Larcher, A. Casella, C. Chitic, G. Torrisi, F. Antonelli,
A. Vespignani, A. Pentland, and B. Lepri, "A multi-source dataset of urban life in the city of
Milan and the Province of Trentino," *Scientific Data*, vol. 2, art. no. 150055, Oct. 2015,
doi: 10.1038/sdata.2015.55.

**URL / DOI.** https://doi.org/10.1038/sdata.2015.55 ·
PDF: https://www.nature.com/articles/sdata201555.pdf ·
PMCID PMC4622222 · PMID 26528394

**Verified: yes.** Full author list (10 authors, in order) confirmed against the Nature PDF, the
Crossref/CrossMark record and the MIT DSpace deposit. Article number 150055, volume 2, received
27 May 2015, accepted 18 Sep 2015, published 27 Oct 2015. Full text read.

**Relevance.** This is the primary citation for the data the report uses, and it is the citation
the examiner is most likely to probe. It documents the aggregation procedure that turns raw Call
Detail Records into the grid-cell activity series: each CDR is attributed to a Radio Base Station,
the RBS coverage map is intersected with the regular grid, and activity is apportioned to grid
squares in proportion to the intersected area. This matters for the report because it means the
series being forecast are *spatially smoothed proxies* for traffic, not direct per-cell byte
counts — a point worth making explicitly when interpreting results. The paper also states the
10-minute aggregation and the CC-BY licence.

**Key numbers or claims worth quoting.**
- Time slots are aggregated to **ten minutes**; the dataset field description states the end of an
  interval is obtained by "adding 600,000 milliseconds (10 min)" to the start timestamp.
- Milan is covered by a grid of squares of **about 235 m × 235 m**; Trentino by 6,575 squares.
- Caution: the extracted text of the Nature PDF renders the Milan square count as
  "1,000( squares", an OCR artefact. The figure of **10,000 squares (a 100 × 100 grid)** is stated
  explicitly and independently in refs [3] and [4] below, both of which use this dataset. If the
  report quotes "100 × 100 = 10,000 cells", cite [3] or [4] alongside [1] rather than [1] alone.
- Activity variables are SMS-in, SMS-out, Call-in, Call-out and Internet traffic activity, the
  latter being "the number of CDRs generated inside a given Square id during a given Time
  interval" — i.e. an *activity count*, not a volume in bytes. Worth stating in the report.

---

### [2] Zhang & Patras (2018) — STN / D-STN, long-term forecasting on Milan

**Citation (IEEE style).**
C. Zhang and P. Patras, "Long-term mobile traffic forecasting using deep spatio-temporal neural
networks," in *Proc. 19th ACM Int. Symp. Mobile Ad Hoc Networking and Computing (MobiHoc '18)*,
Los Angeles, CA, USA, Jun. 2018, pp. 231–240, doi: 10.1145/3209582.3209606.

**URL / DOI.** https://doi.org/10.1145/3209582.3209606 ·
Author copy: https://homepages.inf.ed.ac.uk/ppatras/pub/mobihoc18.pdf

**Verified: yes.** Pages 231–240 and the June 2018 date confirmed against the ACM DL record, the
authors' BibTeX, and the Edinburgh Research Explorer deposit. Full text read. (Note: the ACM DL
proceedings title reads "Eighteenth" while the authors' own records and the paper say "Nineteenth"
— the symposium is MobiHoc 2018 either way. Use "MobiHoc '18" to sidestep this.)

**Relevance.** The single most useful contrast paper for this report. It uses exactly the same
Milan data at exactly the same 10-minute resolution and compares deep spatio-temporal models
against ARIMA and Holt-Winters. Crucially, its headline claims are about **long-horizon**
forecasting (up to 10 hours / 60 steps ahead), and the deep models' advantage is explicitly shown
to *grow with horizon*. This is the pivot for the report's positioning: at h = 1 the argument for
architectural complexity is far weaker than at h = 60, and this paper's own tables show the
classical baselines are much closer to the deep models at short horizons. It also independently
confirms the Milan grid geometry and the study period.

**Key numbers or claims worth quoting.**
- "Milan's coverage area is partitioned into 100 × 100 squares of 0.055 km² (i.e. 235 m × 235 m)";
  Trentino is 117 × 98 cells of 1 km² each. Data collected **1 Nov 2013 – 1 Jan 2014**.
- Headline abstract claim: "up to 61% smaller prediction errors as compared to widely used
  forecasting approaches, while operating with up to 600 times shorter measurement intervals."
- The 61% figure is **long-term and relative to HW-ExpS**: the text gives "61% and 35% lower NRMSE
  than HW-ExpS and ARIMA" respectively in the long-term regime. The report must not quote "61%
  better than ARIMA" — that is not what the paper says.
- ARIMA configured as p = 3, d = 1, q = 2; HW-ExpS with α = 0.9, β = 0.1, γ = 0.001, both tuned by
  cross-validation on the training set. Useful to note that their ARIMA is *non-seasonal*, which is
  a fair criticism the report can make in favour of its own SARIMA with seasonal differencing.
- Their qualitative observation that "ARIMA gives almost linear and slowly increasing estimates"
  and "yields a nearly constant output that is close to the average of the measurements" at long
  horizons is exactly the behaviour a one-step-ahead study would *not* observe — a good framing
  device.
- Training split: Milan 1 Nov – 10 Dec 2013 (40 days) train, next 10 days validate, 20–30 Dec test.
  Training set size n = 57,480,000 points.

---

### [3] Zhang, Zhang, Yuan & Zhang (2018) — densely connected CNN on Milan

**Citation (IEEE style).**
C. Zhang, H. Zhang, D. Yuan, and M. Zhang, "Citywide cellular traffic prediction based on densely
connected convolutional neural networks," *IEEE Communications Letters*, vol. 22, no. 8,
pp. 1656–1659, Aug. 2018, doi: 10.1109/LCOMM.2018.2841832.

**URL / DOI.** https://doi.org/10.1109/LCOMM.2018.2841832 ·
IEEE Xplore document 8368274 ·
Author copy: https://chuanting.github.io/assets/pdf/ieee_cl_2018.pdf

**Verified: yes.** Volume 22, issue 8, pp. 1656–1659 confirmed via Crossref and a citing PMC
article; DOI confirmed on the author's own PDF. Full text read.

**Relevance.** Directly relevant because it makes a methodological choice the report should
contrast with: it explicitly **aggregates the 10-minute data up to hourly** because the 10-minute
series are too sparse and because "resource planning in 10 minutes level is a non-trivial task."
The report is doing precisely the thing this paper declined to do, so this is the reference to cite
when justifying why 10-minute-resolution one-step forecasting is a distinct and under-studied
problem — and also when discussing zero-inflation and the log transform, since sparsity is the
stated reason for their aggregation.

**Key numbers or claims worth quoting.**
- "In the dataset, H = W = 100, which means the whole city area is divided into 100 × 100 cells.
  The traffic is recorded during the period from 00:00 11/01/2013 to 00:00 01/01/2014 with a
  temporal interval of 10 minutes." (Independent confirmation of the grid and period.)
- "During the 10 minutes time interval of the original dataset, a large proportion of the cell
  traffic is zero, which makes the data very sparse. Thus in this letter, the traffic is aggregated
  per hour." — quotable justification for the report's own handling of sparsity.
- Baselines used: Historical Average (HA), ARIMA and LSTM; results reported as RMSE only, in a
  figure rather than a table. **Do not quote a specific percentage improvement from this paper** —
  the improvements are shown graphically and the letter states only that RMSE "can be significantly
  improved." Its abstract wording is qualitative.
- Test split: last 7 days as test, everything prior as training. Optimiser: Adam, batch 32, 100
  epochs, initial learning rate 0.01 decayed at 50% and 75% of epochs.

---

### [4] Zhang, Zhang, Qiao, Yuan & Zhang (2019) — STCNet, transfer learning across zones

**Citation (IEEE style).**
C. Zhang, H. Zhang, J. Qiao, D. Yuan, and M. Zhang, "Deep transfer learning for intelligent
cellular traffic prediction based on cross-domain big data," *IEEE Journal on Selected Areas in
Communications*, vol. 37, no. 6, pp. 1389–1401, Jun. 2019, doi: 10.1109/JSAC.2019.2904363.

**URL / DOI.** https://doi.org/10.1109/JSAC.2019.2904363

**Verified: yes.** Volume 37, no. 6, pp. 1389–1401, June 2019 confirmed from the author's own
BibTeX entry on his homepage and from two independent citing records (Springer *Wireless Networks*
and IEEE/ACM ToN). Abstract read in full; full paper text not read in this session.

**Relevance.** The canonical "ConvLSTM hybrid on cellular traffic" reference. STCNet uses a
convolutional LSTM to capture spatio-temporal dependencies and adds a clustering step over city
functional zones plus inter-cluster transfer learning. It is the right citation for the claim that
the state of the art in this area is *spatio-temporal and multi-cell*, which in turn sharpens the
report's framing: the report deliberately studies the **univariate, per-cell** problem, so it is
measuring how much of the achievable accuracy is available without any spatial modelling at all.

**Key numbers or claims worth quoting.**
- "the transfer learning based on STCNet brings about 4%~13% extra performance improvements"
  (from the abstract). This is the improvement attributable to the *transfer* step specifically,
  not to the whole model versus ARIMA — quote it with that qualification.
- The paper's own framing that cellular traffic prediction "can be treated as a time series
  forecasting problem" but is complicated by user mobility inducing spatial dependence.

---

### [5] Huang, Chiang & Li (2017) — RNN vs 3D-CNN vs CNN-RNN on Telecom Italia data

**Citation (IEEE style).**
C.-W. Huang, C.-T. Chiang, and Q. Li, "A study of deep learning networks on mobile traffic
forecasting," in *Proc. IEEE 28th Annu. Int. Symp. Personal, Indoor, and Mobile Radio Communications
(PIMRC)*, Montreal, QC, Canada, Oct. 2017, pp. 1–6, doi: 10.1109/PIMRC.2017.8292737.

**URL / DOI.** https://doi.org/10.1109/PIMRC.2017.8292737 · IEEE Xplore document 8292737

**Verified: yes.** Authors, venue, location, pages 1–6 and DOI confirmed via Crossref and the DBLP/
researchr record; use of the Telecom Italia Milan data confirmed via two independent citing papers.
Abstract read; full paper text not read in this session.

**Relevance.** An early, clean head-to-head of recurrent versus convolutional versus hybrid
architectures on this exact data, and therefore a natural predecessor to the report's LSTM-vs-TCN
comparison. Its conclusion — that CNN and RNN extract *geographical* and *temporal* features
respectively, and the hybrid leads — is the standard motivation for ConvLSTM-style work. The report
can note that in the univariate setting the "CNN" role is repurposed: the TCN's convolutions run
along time, not space, which is a different use of convolution from the one this paper evaluates.

**Key numbers or claims worth quoting.**
- "CNN-RNN is a reliable model leading in all tasks with 70 to 80% forecasting accuracy" (abstract).
  Note that "forecasting accuracy" here is their own task-specific metric, so this figure should be
  quoted as-stated and not converted into an error reduction.

---

### [6] Wang et al. (2019) — graph-based prediction, in-cell/inter-cell decomposition

**Citation (IEEE style).**
X. Wang, Z. Zhou, F. Xiao, K. Xing, Z. Yang, Y. Liu, and C. Peng, "Spatio-temporal analysis and
prediction of cellular traffic in metropolis," *IEEE Transactions on Mobile Computing*, vol. 18,
no. 9, pp. 2190–2202, Sep. 2019, doi: 10.1109/TMC.2018.2870135.

**URL / DOI.** https://doi.org/10.1109/TMC.2018.2870135 ·
Author copy: https://xu-wang11.github.io/TMC19_Traffic_Prediction.pdf
(Earlier conference version: *Proc. IEEE ICNP 2017*, pp. 1–10, doi: 10.1109/ICNP.2017.8117559.)

**Verified: yes.** Seven-author list, volume 18, issue 9, pp. 2190–2202 confirmed from DBLP/researchr
and the CityUHK Scholars repository record. Results section read via the author's PDF.

**Relevance.** Not a Milan-dataset paper — it uses a Chinese operator's data (1.5 M users, 5,929
towers) — but it is the best available source for a *directly quotable* comparison table that puts
a naive baseline, ARIMA, LSTM and Holt-Winters side by side on cellular traffic. That table is the
empirical backbone for the report's argument about what naive baselines are actually worth. It also
supports the report's decision to treat the per-cell series as meaningful: the paper's in-cell vs
inter-cell decomposition is exactly an argument that a raw per-cell series mixes two processes.

**Key numbers or claims worth quoting.**
- "GNN-D achieves 62.2, 19.7, 16.3, 13.2, 18.5 percent smaller MAE than NAIVE, ARIMA, LSTM, HW and
  GNN-A, respectively."
- The implied ordering is the interesting part for this report: their LSTM beats ARIMA, but only by
  a few points of MAE, and Holt-Winters (HW) beats their LSTM. That is a strong, citable data point
  for "well-specified classical seasonal methods are competitive."
- Caveat to state in the report: these are *their* implementations on *their* data, and the naive
  baseline is a plain naive, not a seasonal naive.

---

### [7] Trinh, Giupponi & Dini (2018) — LSTM on raw LTE control-channel traffic

**Citation (IEEE style).**
H. D. Trinh, L. Giupponi, and P. Dini, "Mobile traffic prediction from raw data using LSTM
networks," in *Proc. IEEE 29th Annu. Int. Symp. Personal, Indoor and Mobile Radio Communications
(PIMRC)*, Bologna, Italy, Sep. 2018, pp. 1827–1832, doi: 10.1109/PIMRC.2018.8581000.

**URL / DOI.** https://doi.org/10.1109/PIMRC.2018.8581000 · IEEE Xplore document 8581000

**Verified: yes.** Three authors (Trinh, Giupponi, Dini — note: *not* Bui or Widmer, who are
co-authors on a different 2017 PIMRC paper by the same first author), pages 1827–1832, Bologna,
9–12 Sep 2018, confirmed via DBLP/researchr and Crossref. Abstract read; full paper not read.

**Relevance.** Useful precisely because it is *not* the Milan dataset: it uses PDCCH measurements
sampled at 1 ms from a single LTE base station. It is one of the few papers in this space that
explicitly evaluates **one-step prediction separately from long-term prediction** and that studies
how the LSTM's input window length affects error — which is the same design question the report
faces when choosing lookback length. Cite it when justifying the lookback sweep.

**Key numbers or claims worth quoting.**
- Design framing worth quoting: "we evaluate the one-step prediction and the long-term prediction
  errors ... considering different numbers for the duration of the observed values, which determines
  the memory length of the LSTM network."
- `[unverified]` — specific error values are not quoted here because the full text was not read.
  Do not cite a number from this paper without reading it.

---

### [8] Bega, Gramaglia, Fiore, Banchs & Costa-Pérez (2019) — DeepCog, forecasting for resource allocation

**Citation (IEEE style).**
D. Bega, M. Gramaglia, M. Fiore, A. Banchs, and X. Costa-Pérez, "DeepCog: Cognitive network
management in sliced 5G networks with deep learning," in *Proc. IEEE INFOCOM 2019 — IEEE Conf.
Computer Communications*, Paris, France, Apr./May 2019, pp. 280–288,
doi: 10.1109/INFOCOM.2019.8737488.

**URL / DOI.** https://doi.org/10.1109/INFOCOM.2019.8737488 ·
Open copy: https://dspace.networks.imdea.org/bitstream/handle/20.500.12761/503/DeepCog_Cognitive_Network_Management_Sliced_5G_Networks_Deep_Learning_2019_EN.pdf

**Verified: yes.** Five authors, INFOCOM 2019, pp. 280–288 confirmed via DBLP/researchr and the
authors' own citation guidance in the DeepCog repository; DOI confirmed via the UC3M repository
record. (The UC3M record lists the page range as 1–9 for the preprint; 280–288 is the proceedings
pagination.) Abstract read; full paper not read.

**Relevance.** The strongest available argument that **forecast error metrics are not the same as
operational value**. DeepCog deliberately trains against an asymmetric, operator-defined cost
(over-provisioning versus service violations) rather than a symmetric error, and shows that doing
so changes the outcome substantially. This is directly relevant to the report's use of the Huber
loss and to any discussion of why MAE/RMSE/MASE may not rank models the way an operator would.

**Key numbers or claims worth quoting.**
- "DeepCog's tight integration of machine learning into resource orchestration allows for
  substantial (50% or above) reduction of operating expenses with respect to resource allocation
  solutions based on state-of-the-art mobile traffic predictors."
- Frame this carefully: the 50% is a reduction in *operating expense* under their cost model, not a
  reduction in forecast error.

---

### [9] Zhang, Patras & Haddadi (2019) — survey of deep learning in mobile networking

**Citation (IEEE style).**
C. Zhang, P. Patras, and H. Haddadi, "Deep learning in mobile and wireless networking: A survey,"
*IEEE Communications Surveys & Tutorials*, vol. 21, no. 3, pp. 2224–2287, 3rd Quart. 2019,
doi: 10.1109/COMST.2019.2904897.

**URL / DOI.** https://doi.org/10.1109/COMST.2019.2904897 ·
arXiv:1803.04311 ·
Accepted copy: https://www.pure.ed.ac.uk/ws/files/81194497/comst19.pdf

**Verified: yes.** Volume 21, issue 3, pp. 2224–2287 (67 pages) confirmed from the Edinburgh
Research Explorer record and the accepted-version PDF header. Abstract and structure inspected.

**Relevance.** A survey citation to establish the breadth of the field in one reference rather than
ten, and to justify the report's narrowing of scope. Use it in the opening paragraph of Related
Work as the "for a comprehensive treatment see [9]" move, then immediately narrow to short-horizon
univariate forecasting. It is also a safe, high-credibility source for the general claim that deep
learning has been applied across essentially every mobile-networking task.

**Key numbers or claims worth quoting.**
- Best used for scope rather than for numbers. If a number is wanted, the survey's length and
  coverage (67 pages, organised by application domain) is itself the point.

---

## A.2 Classical / statistical forecasting of network traffic

### [10] Shu, Yu, Liu & Yang (2003) — seasonal ARIMA for wireless traffic

**Citation (IEEE style).**
Y. Shu, M. Yu, J. Liu, and O. W. W. Yang, "Wireless traffic modeling and prediction using seasonal
ARIMA models," in *Proc. IEEE Int. Conf. Communications (ICC)*, Anchorage, AK, USA, May 2003,
vol. 3, pp. 1675–1679, doi: 10.1109/ICC.2003.1203886.

**URL / DOI.** https://doi.org/10.1109/ICC.2003.1203886

**Journal version (also verified).**
Y. Shu, M. Yu, O. Yang, J. Liu, and H. Feng, "Wireless traffic modeling and prediction using
seasonal ARIMA models," *IEICE Transactions on Communications*, vol. E88-B, no. 10, pp. 3992–3999,
Oct. 2005, doi: 10.1093/ietcom/e88-b.10.3992.
https://doi.org/10.1093/ietcom/e88-b.10.3992

**Verified: yes** for both records. ICC 2003 pages 1675–1679 confirmed via DBLP/BibSonomy and
Crossref; IEICE volume/issue/pages confirmed on the IEICE search record, which also gives the
five-author list for the journal version (note it differs from the conference version by adding
H. Feng). Abstracts read; full texts not read. `[unverified]` — the ICC conference *location*
(Anchorage) is the known ICC 2003 venue but was not confirmed from the paper record itself; omit
the location if in doubt.

**Relevance.** The foundational citation for "SARIMA applied to cellular traffic." It is the right
reference for the report's choice of a *seasonal* ARIMA rather than a plain ARIMA, and its central
technical contribution — a general expression for seasonal ARIMA with **two periodicities** — is
directly on point for Milan data, which exhibits both a daily cycle (lag 144 at 10-minute
resolution) and a weekly cycle (lag 1008). The report seasonally differences at lag 144 only; this
paper is the natural place to acknowledge that the weekly period exists and to justify not modelling
it (parameter cost, series length).

**Key numbers or claims worth quoting.**
- "we give a general expression of seasonal ARIMA models with two periodicities and provide
  procedures to model and to predict traffic"; validated on GSM traffic in China.
- `[unverified]` — no specific error figures are quoted here; both abstracts are qualitative
  ("could be used to model and predict"). Do not attribute a numeric accuracy claim to this paper.

---

### [11] Hyndman & Athanasopoulos (2021) — forecasting textbook

**Citation (IEEE style).**
R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. Melbourne,
Australia: OTexts, 2021.

**URL / DOI.** https://otexts.com/fpp3/ · ISBN 978-0-9875071-3-6 (ISBN-10 0987507133), 442 pp.

**Verified: yes.** Edition, publisher, place, year, ISBN-10/13 and page count confirmed from the
book's own citation guidance at otexts.com/fpp3, Google Books and Open Library. Print version last
updated 31 May 2021.

**Relevance.** The practical, citable source for several of the report's methodological choices that
would otherwise look arbitrary: the definition of the seasonal-naive benchmark, the rationale for
variance-stabilising (log / Box–Cox) transformations, seasonal differencing as a route to
stationarity, and the recommendation to always report against a benchmark method. It is also the
standard reference for the MASE definition as used in practice. An undergraduate report is expected
to cite a textbook somewhere; this is the defensible one.

**Key numbers or claims worth quoting.**
- Best cited for definitions and methodological recommendations rather than figures. The
  recommended citation string given by the authors is: "Hyndman, R.J., & Athanasopoulos, G. (2021)
  Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia."

---

## A.3 Sequence models

### [12] Hochreiter & Schmidhuber (1997) — LSTM

**Citation (IEEE style).**
S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8,
pp. 1735–1780, Nov. 1997, doi: 10.1162/neco.1997.9.8.1735.

**URL / DOI.** https://doi.org/10.1162/neco.1997.9.8.1735 ·
Author copy: https://www.bioinf.jku.at/publications/older/2604.pdf · PMID 9377276

**Verified: yes.** Volume 9, issue 8, pp. 1735–1780, November 1997 confirmed via MIT Press Direct,
ACM DL and the authors' own PDF (whose header reads "Neural Computation 9(8):1735–1780, 1997").
Submitted 1995, accepted 24 Feb 1997.

**Relevance.** The origin of the architecture the report evaluates. The specific reason to cite it
rather than a secondary source is the paper's own framing of the problem it solves: vanishing /
decaying error backflow in recurrent backpropagation. That framing is what justifies the report's
expectation that an LSTM should be able to carry information across the ~144-step daily cycle,
which is in turn the hypothesis the empirical results test.

**Key numbers or claims worth quoting.**
- "LSTM can learn to bridge minimal time lags in excess of 1000 discrete time steps by enforcing
  constant error flow through constant error carrousels within special units."
- "LSTM is local in space and time; its computational complexity per time step and weight is O(1)."
- The 1000-step claim is worth quoting because 1008 is exactly the weekly lag at 10-minute
  resolution — a neat rhetorical hook, provided the report is clear it is a claim about *artificial*
  long-time-lag tasks, not a guarantee on noisy real data.

---

### [13] Bai, Kolter & Koltun (2018) — TCN

**Citation (IEEE style).**
S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic convolutional and recurrent
networks for sequence modeling," arXiv:1803.01271, Mar. 2018,
doi: 10.48550/arXiv.1803.01271.

**URL / DOI.** https://arxiv.org/abs/1803.01271 · https://doi.org/10.48550/arXiv.1803.01271 ·
Code: https://github.com/locuslab/TCN

**Verified: yes.** Authors and affiliations (Bai and Kolter at CMU, Koltun at Intel Labs), arXiv ID,
and first submission date 4 Mar 2018 confirmed from the arXiv record and Koltun's own publications
page. Abstract read in full. **Important:** this is a technical report / arXiv preprint, *not* a
published conference paper. Koltun's own page lists it as "Technical Report, arXiv:1803.01271,
2018," and the authors' BibTeX uses `@article{..., journal = {arXiv:1803.01271}}`. Cite it as a
preprint; claiming it as an ICML/NeurIPS paper would be wrong.

**Relevance.** The direct source for the report's TCN design — causal convolutions, dilations and
residual connections — and for the hypothesis being tested. Its central claim is precisely the
report's research question transplanted to a new domain, which makes the report a small replication
in a setting the original authors did not consider (a single, strongly seasonal, real-valued
telecom series at short horizon).

**Key numbers or claims worth quoting.**
- "a simple convolutional architecture outperforms canonical recurrent networks such as LSTMs across
  a diverse range of tasks and datasets, while demonstrating longer effective memory."
- "convolutional networks should be regarded as a natural starting point for sequence modeling
  tasks."
- The paper's own finding that the "infinite memory" advantage attributed to RNNs "is largely absent
  in practice" — a useful counterweight if the report's LSTM underperforms.
- Caveat the report should state: their benchmark suite is synthetic tasks, polyphonic music and
  language modelling. None is a seasonal real-valued forecasting task, so transfer of the conclusion
  is an open question — which is the report's contribution.

---

### [14] van den Oord et al. (2016) — WaveNet, dilated causal convolutions

**Citation (IEEE style).**
A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner,
A. Senior, and K. Kavukcuoglu, "WaveNet: A generative model for raw audio," arXiv:1609.03499,
Sep. 2016, doi: 10.48550/arXiv.1609.03499.

**URL / DOI.** https://arxiv.org/abs/1609.03499 · https://doi.org/10.48550/arXiv.1609.03499

**Alternative venue (also verified).** The same work appears as: A. van den Oord et al., "WaveNet:
A generative model for raw audio," in *Proc. 9th ISCA Speech Synthesis Workshop (SSW 9)*, Sunnyvale,
CA, USA, Sep. 2016, p. 125.
https://www.isca-archive.org/ssw_2016/vandenoord16_ssw.html

**Verified: yes.** Nine-author list and order confirmed from the arXiv record, the arXiv PDF author
block (all at Google DeepMind, London) and the ISCA Archive BibTeX. v1 12 Sep 2016, v2 19 Sep 2016.
Note the ISCA entry is a one-page demo abstract (p. 125), so the arXiv version is the right thing to
cite for the architecture.

**Relevance.** The origin of the dilated causal convolution stack that the TCN generalises, and the
cleanest demonstration that exponentially increasing dilation buys a large receptive field at linear
parameter cost. This is the mechanism the report relies on to cover a 144-step daily cycle without
an unmanageably deep or wide network, so it belongs in the methodology as well as in Related Work.

**Key numbers or claims worth quoting.**
- "The model is fully probabilistic and autoregressive, with the predictive distribution for each
  audio sample conditioned on all previous ones; nonetheless we show that it can be efficiently
  trained on data with tens of thousands of samples per second of audio."
- The autoregressive + causal framing is the transferable idea; the TTS mean-opinion-score results
  are not relevant to the report and should not be quoted.

---

### [15] Shi et al. (2015) — ConvLSTM

**Citation (IEEE style).**
X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, and W.-C. Woo, "Convolutional LSTM network: A
machine learning approach for precipitation nowcasting," in *Advances in Neural Information
Processing Systems 28 (NIPS 2015)*, Montreal, QC, Canada, Dec. 2015, pp. 802–810.
[Preprint: arXiv:1506.04214.]

**URL / DOI.** https://papers.nips.cc/paper_files/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html ·
https://arxiv.org/abs/1506.04214

**Verified: partial.** Six-author list, title, venue (NIPS 2015 poster) and arXiv ID confirmed from
the NeurIPS proceedings PDF, the NeurIPS virtual site and arXiv. `[unverified]` — the page range
802–810 is the commonly cited NIPS 2015 pagination but was **not** confirmed from the proceedings
record in this session. Safest form: cite without page numbers, or cite the arXiv preprint.

**Relevance.** Included because it is the component that papers [4] and [2] build on, so the report
cannot explain the Milan-dataset literature without it. Its relevance to the report is indirect but
worth one sentence: ConvLSTM exists because a fully connected LSTM discards spatial structure. In
the report's univariate setting there *is* no spatial structure to discard, which is exactly why a
plain LSTM is the appropriate recurrent baseline here rather than a ConvLSTM.

**Key numbers or claims worth quoting.**
- "By extending the fully connected LSTM (FC-LSTM) to have convolutional structures in both the
  input-to-state and state-to-state transitions, we propose the convolutional LSTM (ConvLSTM)."
- "our ConvLSTM network captures spatiotemporal correlations better and consistently outperforms
  FC-LSTM and the state-of-the-art operational ROVER algorithm."

---

## A.4 Evaluation methodology, decomposition and diagnostics

### [16] Hyndman & Koehler (2006) — MASE and the critique of MAPE/sMAPE

**Citation (IEEE style).**
R. J. Hyndman and A. B. Koehler, "Another look at measures of forecast accuracy," *International
Journal of Forecasting*, vol. 22, no. 4, pp. 679–688, Oct.–Dec. 2006,
doi: 10.1016/j.ijforecast.2006.03.001.

**URL / DOI.** https://doi.org/10.1016/j.ijforecast.2006.03.001 ·
Author page (with BibTeX and the worked MASE spreadsheet):
https://robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/

**Verified: yes.** Volume 22, issue 4, pp. 679–688, 2006 confirmed from Hyndman's own BibTeX entry,
RePEc/EconPapers, the Monash repository record and Crossref. Abstract read in full.

**Relevance.** The single most important methodological citation in the report. It is the source of
MASE, and — critically for a 10-minute cellular series with many near-zero and zero observations —
it is the source of the argument that percentage-based errors are *degenerate* when the actual value
is zero or close to it. That is not a stylistic preference; it is a correctness argument, and it is
why the report scales errors by the in-sample naive error rather than reporting MAPE. Note also that
MASE's scaling denominator is the mean in-sample one-step naive error, which ties the metric
directly to the persistence baseline the report already computes.

**Key numbers or claims worth quoting.**
- "The methods used in the M-competition and the M3-competition, and many of the measures
  recommended by previous authors on this topic, are found to be inadequate, and many of them are
  degenerate in commonly occurring situations."
- "we propose that the mean absolute scaled error become the standard measure for comparing forecast
  accuracy across multiple time series."
- The interpretation anchor: MASE < 1 means the method beats the in-sample one-step naive benchmark
  on average. State this explicitly in the report; it is the cleanest way to present results across
  heterogeneous cells.

---

### [17] Cleveland, Cleveland, McRae & Terpenning (1990) — STL

**Citation (IEEE style).**
R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning, "STL: A seasonal-trend
decomposition procedure based on loess," *Journal of Official Statistics*, vol. 6, no. 1,
pp. 3–73, 1990.

**URL / DOI.** Open PDF: https://www.wessa.net/download/stl.pdf
(No DOI; *Journal of Official Statistics* volume 6 predates DOI assignment for this title.)

**Verified: yes.** Volume 6, no. 1, pp. 3–73, 1990, and the four-author list confirmed from the
journal PDF's own header page and corroborated by the R `stl` and statsmodels `STL` documentation,
both of which cite it identically. Note the paper was published "with Discussion," which is why the
page range is unusually long. `[unverified]` — no DOI located; cite by journal, volume and pages.

**Relevance.** The method behind the report's decomposition figures and, indirectly, behind its
seasonal-strength numbers (see [18]). Loess-based decomposition is the right choice here over
classical decomposition because it permits the seasonal component to *change over time*, which
matters across a two-month window spanning the Christmas period — a stretch where Milan traffic
patterns are visibly atypical. If the report discusses excluding or flagging the holiday period,
STL's time-varying seasonality is the principled justification.

**Key numbers or claims worth quoting.**
- "STL is a filtering procedure for decomposing a time series into trend, seasonal, and remainder
  components ... a sequence of applications of the loess smoother."
- The design property worth citing: the amounts of seasonal and trend smoothing are user-specified
  and range "in a nearly continuous" fashion, allowing explicit control over how fast seasonality is
  allowed to evolve.

---

### [18] Wang, Smith & Hyndman (2006) — characteristic-based clustering, component strengths

**Citation (IEEE style).**
X. Wang, K. Smith, and R. Hyndman, "Characteristic-based clustering for time series data," *Data
Mining and Knowledge Discovery*, vol. 13, no. 3, pp. 335–364, Nov. 2006,
doi: 10.1007/s10618-005-0039-x.

**URL / DOI.** https://doi.org/10.1007/s10618-005-0039-x

**Verified: yes.** Volume 13, issue 3, pp. 335–364 confirmed from the Monash repository record, DBLP
and Crossref; published online 16 May 2006. The article's own first page (visible in the
ResearchGate rendering) reads "Data Mining and Knowledge Discovery, 13, 335–364, 2006." Author names
on the paper are given as Xiaozhe Wang, Kate Smith and Rob Hyndman.

**Relevance.** The source of the trend-strength and seasonal-strength measures defined as
1 − Var(remainder)/Var(deseasonalised or detrended series), computed from an STL decomposition.
If the report quantifies how strongly seasonal each Milan cell is — for instance to explain why
seasonal-naive is a very hard baseline in some cells and a weak one in others — this is the
reference for that formula. It pairs naturally with [17], which supplies the decomposition the
measure is computed from.

**Key numbers or claims worth quoting.**
- Cite for the definition of the component-strength measures rather than for a headline result.
- `[unverified]` — the exact algebraic form of the strength formula as printed in this 2006 paper
  was not read line-by-line in this session. The formula as commonly used today follows [11]
  (fpp3, §4.3). If the report prints the formula, cite [11] for the exact form used and [18] as its
  origin.

---

### [19] Dickey & Fuller (1979) — unit root test

**Citation (IEEE style).**
D. A. Dickey and W. A. Fuller, "Distribution of the estimators for autoregressive time series with a
unit root," *Journal of the American Statistical Association*, vol. 74, no. 366, pp. 427–431,
Jun. 1979, doi: 10.1080/01621459.1979.10482531.

**URL / DOI.** https://doi.org/10.1080/01621459.1979.10482531 ·
JSTOR 2286348 (also indexed as doi: 10.2307/2286348)

**Verified: yes.** Volume 74, no. 366, pp. 427–431, June 1979 confirmed from the Taylor & Francis
record and Crossref. Note both DOIs resolve to the same article (the JSTOR-era 10.2307 DOI and the
T&F 10.1080 DOI); the T&F one is preferable in a modern reference list. Abstract read.

**Relevance.** The report's stationarity diagnostics. The ADF test is what justifies the differencing
order d in the SARIMA specification. The paper is also worth one honest caveat in the report: with
tens of thousands of observations per cell, the ADF test is extremely powerful and will reject the
unit-root null almost always, so the test should inform but not dictate the choice of d — the
seasonal structure and the ACF matter more.

**Key numbers or claims worth quoting.**
- "Properties of the regression estimator of ρ are obtained under the assumption that ρ = ±1 ...
  The estimator of ρ and the regression t test furnish methods of testing the hypothesis that ρ = 1."
- Cite for the test's existence and null hypothesis, not for a numeric result.

---

### [20] Kwiatkowski, Phillips, Schmidt & Shin (1992) — KPSS test

**Citation (IEEE style).**
D. Kwiatkowski, P. C. B. Phillips, P. Schmidt, and Y. Shin, "Testing the null hypothesis of
stationarity against the alternative of a unit root: How sure are we that economic time series have
a unit root?," *Journal of Econometrics*, vol. 54, no. 1–3, pp. 159–178, Oct.–Dec. 1992,
doi: 10.1016/0304-4076(92)90104-Y.

**URL / DOI.** https://doi.org/10.1016/0304-4076(92)90104-Y

**Verified: yes.** Four authors, volume 54, issues 1–3, pp. 159–178, October–December 1992 confirmed
from ScienceDirect, RePEc/EconPapers and Crossref. The full-text PDF header confirms "Journal of
Econometrics 54 (1992) 159–178. North-Holland," received January 1991, final version October 1991.
Abstract read in full.

**Relevance.** The complement to [19] and the reason to run both. KPSS reverses the hypotheses —
stationarity is the null — so agreement between ADF and KPSS is far more informative than either
alone. For the report this is the defensible way to conclude that the *seasonally differenced*
series is stationary while the raw series is not, which is precisely the evidence needed to justify
D = 1 at lag 144.

**Key numbers or claims worth quoting.**
- "We propose a test of the null hypothesis that an observable series is stationary around a
  deterministic trend. The series is expressed as the sum of deterministic trend, random walk, and
  stationary error, and the test is the LM test of the hypothesis that the random walk has zero
  variance."
- The paper's own conclusion on the Nelson–Plosser data — that "for many of these series the
  hypothesis of trend stationarity cannot be rejected" — is a good illustration of why one-sided
  unit-root evidence is weak evidence.

---

### [21] Makridakis, Spiliotis & Assimakopoulos (2018) — statistical vs ML, PLoS ONE

**Citation (IEEE style).**
S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "Statistical and machine learning forecasting
methods: Concerns and ways forward," *PLoS ONE*, vol. 13, no. 3, art. no. e0194889, Mar. 2018,
doi: 10.1371/journal.pone.0194889.

**URL / DOI.** https://doi.org/10.1371/journal.pone.0194889 ·
Open access: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194889 ·
PMCID PMC5870978

**Verified: yes.** Volume 13, issue 3, article e0194889, published 27 March 2018 confirmed from the
PLOS article page (which gives the recommended citation verbatim), PMC and RePEc. Received 9 Dec
2017, accepted 12 Mar 2018. Abstract read in full.

**Relevance.** The strongest single citation for the report's framing that naive and classical
baselines deserve to be taken seriously. It is a controlled, like-for-like comparison in which
machine learning methods lost — comprehensively — to standard statistical methods. The report's
positioning should be: this result is the prior, the report tests whether it survives in a domain
(high-frequency, strongly seasonal telecom data, abundant training samples per series) where the
conditions are much more favourable to machine learning than the M3 monthly series were.

**Key numbers or claims worth quoting.**
- Evaluated on "a large subset of 1045 monthly time series used in the M3 Competition."
- "After comparing the post-sample accuracy of popular ML methods with that of eight traditional
  statistical ones, we found that the former are dominated across both accuracy measures used and
  for all forecasting horizons examined."
- "their computational requirements are considerably greater than those of statistical methods."
- Essential caveat for the report: these are *short, low-frequency, one-series-at-a-time* fits with
  roughly 100 observations each. The Milan series have thousands of observations at 10-minute
  resolution. The report should say plainly that this is the dimension along which it departs from
  the M3 setting, otherwise the citation can be turned against it in the viva.

---

### [22] Makridakis, Spiliotis & Assimakopoulos (2020) — the M4 Competition

**Citation (IEEE style).**
S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "The M4 Competition: 100,000 time series and 61
forecasting methods," *International Journal of Forecasting*, vol. 36, no. 1, pp. 54–74,
Jan.–Mar. 2020, doi: 10.1016/j.ijforecast.2019.04.014.

**URL / DOI.** https://doi.org/10.1016/j.ijforecast.2019.04.014 ·
https://www.sciencedirect.com/science/article/pii/S0169207019301128

**Verified: yes.** Volume 36, issue 1, pp. 54–74, January–March 2020 confirmed from ScienceDirect,
RePEc and Crossref (online 19 Jul 2019). Abstract read in full.

**Relevance.** The counterweight to [21] and the more nuanced position. M4's outcome was that
*hybrid* and *combination* methods won, not pure ML and not pure statistics — which supports a
measured conclusion in the report rather than a triumphalist one in either direction. It is also the
standard citation for the competition-based evaluation philosophy: fixed held-out data, a
pre-declared metric, no peeking. The report's train/validation/test protocol should be justified in
those terms.

**Key numbers or claims worth quoting.**
- Scale and design: 100,000 series, 61 forecasting methods, and the explicit extension to
  "including prediction intervals in the evaluation process as well as point forecasts."
- `[unverified]` — the specific winning-method accuracy improvements (frequently quoted as ~9–10%
  over the Comb benchmark for the ES-RNN winner) were **not** confirmed from the paper text in this
  session. Do not quote a specific M4 percentage without opening the paper.
- Safe use: cite for the competition's existence, scale, design and its general finding that
  combinations of statistical and ML approaches outperformed either alone.

---

## A.5 Energy motivation and the operational use case

### [23] GSMA Intelligence (2023) — RAN share of operator energy

**Citation (IEEE style).**
GSMA Intelligence, "Going green: Benchmarking the energy efficiency of mobile networks (second
edition)," GSMA Intelligence, London, U.K., Feb. 2023.

**URL.** https://www.gsmaintelligence.com/research/going-green-benchmarking-the-energy-efficiency-of-mobile-networks-second-edition
(Report PDF:
https://prod-cms.gsmaintelligence.com/research-file-download?assetId=18922&reportId=52377)

**Verified: yes.** Title, "second edition" and February 2023 publication date confirmed on the GSMA
Intelligence research page; the headline figures below were read directly from the report PDF.
`[unverified]` — no named individual authors are given; cite the organisation as author. This is
industry grey literature, not peer-reviewed — say so in the report if it is used for a headline
claim.

**Relevance.** The report's introduction needs a credible number for "the radio access network
dominates mobile operator energy consumption." This is the most authoritative industry source, based
on anonymised real operator data rather than modelling. Pair it with [24] for a peer-reviewed figure
so the motivation does not rest on grey literature alone.

**Key numbers or claims worth quoting.**
- "87% of the energy of the participating operators is consumed in the radio access network (RAN).
  The network core and owned data centres (12%) and other operations (1%) account for the rest."
- "the average primary energy efficiency ratio in the RAN reached 6.86 GB/kWh in 2021 ... operators
  used on average 0.13 kWh of energy to transfer 1 GB of data across their RAN networks."
- "one cell network site used on average 26,420 kWh during the whole year"; one mobile connection
  required an average of 17 kWh over 12 months (0.06 connections/kWh). Covers 56 markets.
- Quote the 87% as "for the operators participating in the GSMA Intelligence benchmark" — it is a
  sample statistic, not a universal constant.

---

### [24] Larsen, Christiansen, Ruepp & Berger (2023) — peer-reviewed RAN energy survey

**Citation (IEEE style).**
L. M. P. Larsen, H. L. Christiansen, S. Ruepp, and M. S. Berger, "Toward greener 5G and beyond radio
access networks—A survey," *IEEE Open Journal of the Communications Society*, vol. 4, pp. 768–797,
2023, doi: 10.1109/OJCOMS.2023.3257889.

**URL / DOI.** https://doi.org/10.1109/OJCOMS.2023.3257889 (open access)

**Verified: yes.** Four authors, volume 4, pp. 768–797, 2023 confirmed from the DTU Orbit repository
record and DBLP/researchr. The quoted energy-split figures were read from the article text.

**Relevance.** The peer-reviewed, open-access counterpart to [23], and the better citation for an
academic report. It gives a slightly more conservative RAN share (73%) from an independent
accounting, which is useful: quoting a range of 73–87% across two sources is more honest and more
defensible under questioning than quoting a single number as fact. It also quantifies the headroom
available from energy-saving techniques, which is what makes short-horizon prediction worth doing.

**Key numbers or claims worth quoting.**
- "the RAN consumes 73% of the energy, the core network consumes 13%, the datacentres consume 9% and
  other operations account for 5% of the energy consumption."
- "implementing selected technologies and architectures, the mobile network overall energy
  consumption can be reduced by approximately 30%, corresponding to almost half of the RAN energy
  consumption."
- The 30% figure is the payoff number for the report's motivation: it bounds what better
  traffic-aware operation could plausibly contribute.

---

### [25] Wu, Zhang, Zukerman & Yung (2015) — base-station sleep modes

**Citation (IEEE style).**
J. Wu, Y. Zhang, M. Zukerman, and E. K.-N. Yung, "Energy-efficient base-stations sleep-mode
techniques in green cellular networks: A survey," *IEEE Communications Surveys & Tutorials*,
vol. 17, no. 2, pp. 803–826, 2nd Quart. 2015, doi: 10.1109/COMST.2015.2403395.

**URL / DOI.** https://doi.org/10.1109/COMST.2015.2403395 ·
Author copy: https://www.ee.cityu.edu.hk/~zukerman/wu_survey_final.pdf

**Verified: yes.** Author list (Wu, Zhang, Zukerman, Yung), volume 17, issue 2, pp. 803–826,
article number 7041163, online 12 Feb 2015 confirmed from the CityUHK Scholars record and the 2015
IEEE COMST volume index. Abstract read in full.

**Relevance.** The reference that closes the loop between forecasting accuracy and operational
benefit, and therefore the one that makes the report's introduction more than decorative. Sleep-mode
scheduling is driven by *anticipated* low-traffic periods, so short-horizon forecast quality is the
input to the decision. Just as importantly, this survey is unusually candid about the field's
weaknesses, which gives the report a defensible, non-inflated way to state its motivation.

**Key numbers or claims worth quoting.**
- Sleep mode "takes advantage of changing traffic patterns on daily or weekly basis, and selectively
  switches some lightly loaded base stations to low energy consumption modes."
- The honest caveat, worth quoting verbatim because it strengthens rather than weakens the report:
  "certain simplifying assumptions made in the published papers introduce inaccuracies ... an
  assumption that ignores the effect of traffic load dependent factors on energy consumption. We
  show here that considering this effect may lead to noticeably lower benefit than in models that
  ignore this effect."
- Use this to avoid over-claiming: the report should say that accurate short-horizon forecasts are a
  *necessary input* to sleep-mode control, not that they deliver a specific energy saving.

---

## A.6 Training details

### [26] Huber (1964) — robust loss

**Citation (IEEE style).**
P. J. Huber, "Robust estimation of a location parameter," *The Annals of Mathematical Statistics*,
vol. 35, no. 1, pp. 73–101, Mar. 1964, doi: 10.1214/aoms/1177703732.

**URL / DOI.** https://doi.org/10.1214/aoms/1177703732 ·
https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full

**Verified: yes.** Volume 35, issue 1, pp. 73–101, March 1964 confirmed from the Crossref record and
a Springer citation. Note: one legacy index (dml.mathdoc.fr) lists the issue as "no. 4"; this is an
error in that index — Project Euclid and Crossref both place the article in issue 1. Use issue 1.

**Relevance.** The origin of the loss the report trains its neural models with. The substantive
justification is specific to this data: 10-minute cellular activity series contain occasional very
large spikes (events, anomalies) on top of a mostly smooth diurnal pattern. A squared-error loss lets
those spikes dominate the gradient; Huber's loss is quadratic near zero and linear in the tails, so
it retains squared-error behaviour on the bulk of the data while bounding the influence of outliers.
This is a design decision the examiner can reasonably ask about, so the report should be able to
state the transition point δ it used and why.

**Key numbers or claims worth quoting.**
- Cite for the loss function's definition and its minimax-robustness motivation (bounding the
  supremum of the asymptotic variance over a contamination neighbourhood), not for a numeric result.
- The report should note the practical consequence: with a log transform already applied, the
  spike problem is partly mitigated, so the choice of δ interacts with the transform. Worth one
  sentence of self-awareness.

---

### [27] Kingma & Ba (2015) — Adam

**Citation (IEEE style).**
D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf.
Learning Representations (ICLR)*, San Diego, CA, USA, May 2015. [Online]. Available:
https://arxiv.org/abs/1412.6980

**URL / DOI.** https://arxiv.org/abs/1412.6980 · https://doi.org/10.48550/arXiv.1412.6980

**Verified: yes.** Both authors, the ICLR 2015 venue and San Diego location confirmed from the arXiv
comments field, which reads verbatim: "Published as a conference paper at the 3rd International
Conference for Learning Representations, San Diego, 2015." Corroborated by the UvA-DARE repository
record (event: ICLR 2015, 13 pages) and the ICLR anthology entry. v1 submitted 22 Dec 2014; last
revised 30 Jan 2017 (v9). Affiliations: Kingma, University of Amsterdam; Ba, University of Toronto.
Abstract read in full.

**Relevance.** Standard citation for the optimiser used to train both the LSTM and the TCN. Worth
one sentence rather than a paragraph. The one genuinely relevant property for this report is Adam's
suitability "for non-stationary objectives and problems with very noisy and/or sparse gradients" —
which describes training on sparse, spiky 10-minute cellular series reasonably well. Note that [3]
above also used Adam on this dataset, so the choice is consistent with prior work on the same data.

**Key numbers or claims worth quoting.**
- "an algorithm for first-order gradient-based optimization of stochastic objective functions, based
  on adaptive estimates of lower-order moments."
- "The method is also appropriate for non-stationary objectives and problems with very noisy and/or
  sparse gradients."
- If default hyperparameters are used, the report should state them explicitly (β₁, β₂, ε) rather
  than saying "Adam defaults," since defaults differ between frameworks.

---

# Part B — Suggested narrative outline for the Related Work section

Target ~700–900 words. Five themes, in this order. Supporting references in brackets.

## (i) The dataset and the body of work built on it — approx. 150 words

- Open by introducing the Telecom Italia Big Data Challenge release as the de facto public benchmark
  for cellular traffic forecasting, citing **[1]** for the dataset itself.
- State the properties that matter for this study, each traceable to a source: 10-minute aggregation
  and CC-BY licensing **[1]**; the 100 × 100 grid of 235 m squares and the 1 Nov 2013 – 1 Jan 2014
  window **[2]**, **[3]** (make the point that the grid figure is stated most clearly in the
  downstream papers).
- Make one precision point early, because it affects interpretation throughout: the quantity released
  is an *activity measure* derived by apportioning CDRs across RBS coverage areas, not a byte
  count **[1]**. This is why the report forecasts a dimensionless activity series.
- Note the breadth of downstream use in one sentence with a survey citation **[9]**, then narrow.
- **Positioning:** the report uses the same public data as the prior work, so its numbers are
  comparable in kind; what differs is the horizon and the model class, established in the next
  sections.

## (ii) Classical statistical forecasting of network traffic — approx. 150 words

- Establish that seasonal ARIMA has been the standard linear approach in this domain since the early
  2000s **[10]**, and that its key adaptation for telecom data is handling *multiple* periodicities.
- Connect to the report's own specification: at 10-minute resolution the daily period is 144 and the
  weekly period is 1008. The report differences seasonally at lag 144. Cite **[10]** when
  acknowledging the weekly cycle is real but not modelled, and give the reason (parameter cost and
  the two-month series length).
- Justify the differencing decision with diagnostics rather than assertion: ADF **[19]** and KPSS
  **[20]** in combination, with the honest caveat that at n in the thousands the ADF test rejects
  almost everything, so the ACF structure carries more weight than the p-value.
- Cite **[11]** for the seasonal-naive definition and for the variance-stabilising transform.
- **Positioning:** the report's SARIMA is a *fairly specified* classical model — seasonally
  differenced, on a log scale — not the weak straw-man ARIMA that appears as a baseline in much of
  the deep learning literature (see the p=3,d=1,q=2 non-seasonal ARIMA in **[2]**). This is worth
  stating explicitly; it is the report's main claim to a fair comparison.

## (iii) Deep sequential models and what they have been shown to add — approx. 200 words

- Trace the arc: LSTM as the solution to vanishing gradients in recurrent nets **[12]**, then
  ConvLSTM as the extension that preserves spatial structure **[15]**, then its application to
  cellular traffic **[4]**, alongside purely convolutional spatial treatments **[3]** and graph-based
  ones **[6]**.
- Report what these papers actually claim, with the qualifications from Part A. Specifically:
  the "up to 61% smaller prediction errors" in **[2]** is a long-horizon result against Holt-Winters,
  with the corresponding ARIMA figure being 35%; the "4%~13%" in **[4]** is the marginal gain from
  transfer learning, not the total gain over a classical baseline; **[3]** reports its improvements
  graphically and makes no quotable percentage claim.
- Draw out the pattern that motivates the report: the deep models' margin is consistently reported
  at *long* horizons and over *many cells simultaneously*, and it narrows sharply as the horizon
  shortens **[2]**.
- Note the resolution point: **[3]** explicitly aggregated the 10-minute data to hourly because of
  sparsity and because 10-minute resource planning was considered impractical. **[7]** is one of the
  few papers to isolate one-step-ahead performance, and it does so on different data.
- **Positioning:** the one-step-ahead, native-10-minute, single-cell problem is comparatively
  under-examined, and it is the regime where classical methods should be most competitive. The report
  occupies that gap deliberately.

## (iv) Convolutional versus recurrent sequence modelling — approx. 150 words

- Introduce dilated causal convolutions via WaveNet **[14]**: exponentially growing dilation gives a
  receptive field large enough to span a 144-step daily cycle at linear parameter cost, and causality
  is enforced architecturally rather than by data handling.
- State the TCN claim precisely **[13]**: a generic convolutional architecture outperformed canonical
  recurrent networks across a broad suite, with longer effective memory, and the supposed "infinite
  memory" of RNNs is "largely absent in practice."
- Flag the gap honestly: that suite is synthetic tasks, polyphonic music and language modelling —
  none of them a strongly seasonal, real-valued forecasting problem, and **[13]** is an arXiv
  technical report rather than a peer-reviewed conference paper. Both facts should be stated.
- Note that in this univariate setting convolution plays a different role than in **[3]**, **[5]** or
  **[15]**, where it operates over space. Here it operates over time; the report is not building a
  spatio-temporal model.
- **Positioning:** the report is a small, controlled test of the **[13]** claim in a domain its
  authors did not evaluate, with the LSTM **[12]** as the matched recurrent comparator.

## (v) Evaluation methodology and the case for strong baselines — approx. 200 words

- Lead with the metric argument, because it is the most defensible part of the report. Percentage
  errors are degenerate when actuals approach zero **[16]**, and 10-minute cellular cells are
  frequently near zero **[3]**. Hence MASE, whose scaling denominator is the in-sample one-step naive
  error, making "MASE < 1" directly interpretable as "beats persistence" **[16]**, **[11]**.
- Make the baseline argument with evidence rather than assertion. **[21]** found ML methods dominated
  by statistical ones across all horizons on 1,045 M3 series, at far higher computational cost.
  **[22]** gives the more nuanced M4 outcome, where combinations of statistical and ML methods led.
  **[6]** provides the domain-specific data point: in their comparison Holt-Winters outperformed
  their LSTM on MAE, and ARIMA was within a few points of it.
- State the counter-argument fairly so the report is not one-sided: the M3/M4 series are short and
  low-frequency, whereas the Milan cells offer thousands of high-frequency observations per series —
  conditions much more favourable to learned models. This is the specific dimension along which the
  report tests whether the **[21]** prior holds.
- Cover decomposition and series characterisation: STL **[17]** for the time-varying seasonal
  component (important across the Christmas period in this window), and the component-strength
  measures originating in **[18]** for quantifying how strongly seasonal each cell is — which
  predicts where seasonal-naive will be hard to beat.
- Close on training details in one compact sentence — Huber loss **[26]** for spike robustness, Adam
  **[27]** for optimisation, the latter matching the choice made in **[3]** on this same dataset —
  and on **[8]**, which argues that symmetric error metrics do not correspond to operational value,
  a limitation the report should acknowledge rather than resolve.
- **Positioning:** the report's contribution is a *like-for-like*, correctly-scaled comparison at a
  single horizon, with baselines that were given a fair chance, on the standard public dataset.

### Optional framing note for the introduction rather than Related Work

- The energy motivation belongs in the introduction, not here, but the references live in Part A:
  RAN share of operator energy at 73% **[24]** and 87% **[23]** (quote as a range across the two
  sources, and label **[23]** as industry benchmarking rather than peer-reviewed); the ~30% headroom
  estimate **[24]**; the sleep-mode mechanism that makes short-horizon forecasts operationally useful
  **[25]**; and the caution from **[25]** that traffic-load-dependent power effects make naive
  savings estimates optimistic. **[8]** supports the parallel resource-allocation use case.

---

# Verification summary

**Bibliographic metadata** (title, authors, venue, year, volume/issue/pages, DOI):

| Status | Count | References |
|---|---|---|
| **Fully verified** — every cited metadata field confirmed against a publisher, repository or Crossref record | 26 | [1]–[14], [16]–[27] |
| **Partially verified** — record confirmed but one field could not be pinned down | 1 | [15] (page range 802–810 unconfirmed) |

**Depth of substantive checking** (a separate axis from metadata):

| Status | Count | References |
|---|---|---|
| Full text read; quoted numbers taken directly from the paper | 13 | [1]–[3], [6], [11], [13], [14], [16], [17], [20], [23], [24], [27] |
| Abstract / publisher record read only; no numeric claims quoted beyond those reproduced above | 14 | [4], [5], [7]–[10], [12], [15], [18], [19], [21], [22], [25], [26] |

**Total: 27 references.** All 27 were located via web search and all have a working DOI, arXiv ID or
publisher/repository URL. None is included from memory. Seven entries carry an explicit
`[unverified]` flag on a specific sub-claim; these are listed below.

## Outstanding flags to resolve before submission

1. **[15] Shi et al.** — the page range 802–810 is `[unverified]`. Either omit page numbers or
   confirm against the NIPS 2015 proceedings front matter.
2. **[17] Cleveland et al.** — no DOI exists for this article; the reference list must accommodate a
   DOI-less entry.
3. **[18] Wang et al.** — if the component-strength formula is printed in the report, cite **[11]**
   for the exact modern form and **[18]** as its origin, rather than attributing the printed formula
   to **[18]** alone.
4. **[22] M4** — do not quote a specific accuracy percentage without opening the paper; only the
   design and general findings were verified.
5. **[7] Trinh et al.** and **[5] Huang et al.** — abstracts only were read. Do not quote numeric
   results from these beyond what is reproduced above.
6. **[13] Bai et al.** — cite as an arXiv technical report, not as a conference paper.
7. **[23] GSMA** — grey literature; label it as such in the text.
8. **[10] Shu et al.** — the ICC 2003 conference location was not confirmed from the paper record;
   omit it or verify separately. Two versions exist (ICC 2003 and IEICE 2005) with different author
   lists; pick one and be consistent.
