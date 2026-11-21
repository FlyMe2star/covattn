# LCL dToU trial: pre-declared evaluation protocol

Purpose: direct real-world evidence for the what-if claim. In the Low Carbon London 2013 dynamic time-of-use
trial, the experimenters set the price signal (day-ahead High / Low / Normal) and a flat-tariff group serves
as control, so the true effect of the price on the ToU group's load can be estimated without any forecasting
model. We compare Chronos-2's predicted effect of the price (zero-shot and repaired) with that estimate.

This file is committed before any forecasting model is run on LCL data. The model-free stage
(`scripts/lcl_feasibility.py`) fixes two things only: the clock alignment of the tariff file and whether the
ground truth is precise enough to continue. Everything below is fixed now; deviations will be reported as such.

## Data (fixed)

- Consumption: Kaggle mirror `jeanmidev/smart-meters-in-london` (refactor of London Datastore dataset vqm0d,
  whose large files time out at the origin, HTTP 524), half-hourly blocks + `informations_households.csv`.
  Tariff: London Datastore `Tariffs.xlsx` (md5 d64694233f1e3b7241fb1db02ff063f2). `scripts/lcl_download.py`.
- Panel: households with >= 90% finite readings in 2013, per group. Group load = mean kWh over finite panel
  households per half-hour slot.
- Clock: readings are GMT. If the alignment table of the feasibility report shows the strongest High-run
  effect at shift 0 in GMT months and at -2 in BST months, the tariff file is taken to be in local clock time
  and is moved by -2 slots during BST (`--tariff-bst-shift -2`); if both are at 0, no shift. Any other
  pattern is reported and the analysis stops for a decision.

### Amendment 1 (2026-11-21, before any model run): clock alignment

The feasibility report without any shift (`results/lcl/feasibility/report.json`) shows a pattern the rule
above did not anticipate: the strongest High-run effect is at shift +1 in GMT months and at -1 in BST months
(the same holds for Low runs: +1 and -1). The two-slot difference between the seasons is the local-time /
GMT offset foreseen above; the common +1 is consistent with readings being stamped at the end of their half
hour (the first readings of the data are stamped 00:30). Decision (by the author, as the rule required):
primary analysis with the tariff moved by +1 slot everywhere and by a further -2 slots during BST
(`--tariff-shift 1 --tariff-bst-shift -2`, i.e. +1 in GMT months, -1 in BST months); the unshifted analysis
is reported as a sensitivity analysis. The same alignment is used for the covariate given to the models.

## Ground truth (fixed, model-free)

Multiplicative difference-in-differences against the Std group: D = log ToU - log Std; for each event day the
baseline is the per-slot median of D over the 10 nearest all-Normal days of the same day type (Mon-Fri vs
Sat/Sun); counterfactual ToU load under all-Normal prices cf = Std * exp(baseline); true effect
e = ToU - cf over all 48 slots of the day. Noise level: the same estimator on all-Normal days (placebo).

Go / no-go (decided from the feasibility report only): continue if, for High and for Low runs separately, the
median placebo z-score of the run effect has the expected sign and |z| >= 2, and the 2012 schedule placebo
mean relative effect is within +-1/3 of the 2013 effect size. Otherwise stop and report the data as too
noisy for this purpose.

## Forecasting task (fixed)

- Target: ToU group load (kWh per half-hour). Known covariate: price in p/kWh (Low 3.99, Normal 11.76,
  High 67.20; flat Normal before 2013), past and future.
- Origins: 00:00 GMT of every 2013 event day with a complete 48-slot truth; horizon H = 48 (the signal was
  announced a day ahead, so the next day's prices are known at the origin). Context L in {512, 2048}.
- Factual run: actual future prices. Counterfactual run: all future slots set to Normal.
  Predicted effect ê = q50(factual) - q50(counterfactual).
- Methods: zero-shot Chronos-2 with covariate; zero-shot without covariate (accuracy reference only);
  R1 (k = 20, gamma = 2.5, heads from Exp2); R2 (gates of exp3/v2, seed 0); context dither (sigma 0.3, 8
  copies). Nothing is tuned on LCL. Full fine-tune / LoRA from exp3/v2 as references if time permits.

## Metrics (fixed)

Primary (effect size): calibration slope beta = sum <ê, e> / sum ||ê||^2 over event days (ê is noise-free,
so beta is not biased by noise in e; beta = 1 means the right size, beta > 1 means attenuated) and the
effect gain g = sum <ê, e> / (sum ||e||^2 - sum noise), where noise is the placebo mean of ||e||^2 per day.
95% CIs from a block bootstrap over weeks (2000 resamples).

Secondary: per level (High / Low), mean predicted relative effect over the event slots vs the DiD estimate;
sign agreement per event run; Spearman correlation across event runs between predicted and DiD run effects;
factual accuracy on event days (MAE and WQL of the factual forecast against the observed ToU load) and its
ratio to zero-shot.

Reported in full whatever the outcome, including if zero-shot is not attenuated on these data or if the
repairs overshoot.
