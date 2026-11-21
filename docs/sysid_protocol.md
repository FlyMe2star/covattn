# Real system-identification rigs: pre-declared evaluation protocol (Exp13)

Purpose: test whether the repairs improve forecasts of real, input-driven systems where the synthetic results
predict attenuation, and stop this line of work if they do not. Committed before any forecasting model is run on
these data. Context: on the LCL pricing trial (an immediate response, docs/lcl_protocol.md) zero-shot Chronos-2
was not clearly attenuated and the repairs did not help; that result was seen before this protocol was written.

## Data (fixed)

Measured lab rigs with designed, open-loop inputs (`scripts/sysid_prepare.py`, records and model-free ARX memory
in `data/sysid/processed/records.json`): Cascaded Tanks, Wiener-Hammerstein, Silverbox, ParWH, F-16 GVT
(acceleration 1; full multisine and sine-sweep levels 1-7), Fine Steering Mirror (2 inputs as covariates, each of
2 outputs as target). Excluded in advance: EMPS (closed loop: input computed from the output); CED (records of
500 samples cannot hold L = 512 plus H = 64).

## Forecasting task (fixed)

- Target y, known covariate(s) u (past and future). Horizon H = 64, context L in {512, 2048} (a record is used
  for an L only if N >= L + H).
- Windows: per benchmark and L, 64 windows spread evenly over its eligible records (ceil(64 / records) per
  record), origins equally spaced in [L, N - H]. No model is fitted on these data, so estimation and validation
  records are both evaluation data.
- Methods: zero-shot with covariates; zero-shot without covariates (covariate reliance); R1 (k = 20,
  gamma = 2.5); R2 (exp3/v2 gates); dither (sigma 0.3, 8 copies); ARX fitted on the context and simulated over the
  horizon with the known future input (reference, never a candidate). Nothing is tuned on these data.

## Metrics (fixed)

Per window: MAE of the median forecast over the horizon; ratio to zero-shot. Per benchmark: geometric mean of the
window ratios (95% CI: bootstrap over windows, 2000 resamples). Overall: geometric mean of the six benchmark
values (CI: the same bootstrap applied within every benchmark). Also WQL (9 levels) the same way, and per
benchmark the zero-shot covariate reliance = gmean of MAE(no covariate) / MAE(with covariate).

Window periodicity (model-free): p = max over lags P in [16, (L + H) / 2] of corr(u_t, u_{t+P}) over the window's
input [o - L, o + H) (max over input channels); periodic if p >= 0.95.

## Predictions (from Exp3 synthetic test: R2 error ratio 0.93-0.98 for every input type, R1 1.01-1.04)

- P1 (primary): R2 lowers the forecast error of zero-shot: overall MAE ratio < 1 with the 95% CI below 1, at
  both L.
- P2: R1 does not lower the error (overall MAE ratio >= 0.99).
- P3 (conditions): the R2 gain (1 - window ratio) is larger for longer ARX memory (Spearman over records > 0)
  and in non-periodic than in periodic windows.
- P4 (sanity): the inputs drive the outputs: covariate reliance > 1 on every benchmark.

## Stop rule (fixed)

If P1 fails at L = 512 (the primary context), the real-world line of work for the repairs is stopped and the
result is reported as is. If P1 holds, phase 2 (a training-free diagnostic that decides when to apply the
repair, thresholds chosen on synthetic validation data only) gets its own protocol before it is run.
