# Phase 2: a training-free gate that decides when to apply the repair (Exp14), pre-declared

Status when this was written (2026-11-21): ungated R2 lowered forecast error on six real lab rigs (Exp13, P1 met,
docs/sysid_protocol.md) but weakened the price effect on the LCL pricing trial (Exp12, primary truth, L = 512,
where zero-shot was not clearly attenuated), and ungated R1 / R2 were slightly worse than zero-shot on fev-bench
(Exp6). All three were seen before this protocol. The gate below is chosen on simulated / semi-synthetic data
only and confirmed on DaISy data that no model has been run on. This file is committed before the DaISy data are
downloaded and before any gate is computed.

## Gate (fixed form; two thresholds chosen on selection data)

Per forecast window, using the context and the known future input only:
1. For every covariate k fit an ARX model of y on u_k on the context (orders by AIC, the grid of
   `baselines.fit_arx`) and the AR model with the same na and no input. Input share
   s_k = 1 - RSS(ARX) / RSS(AR); k* = argmax s_k, s = s_{k*}.
2. Probe: du = +1 context-std of u_{k*}, a step over the whole horizon. e_arx = zero-state response of the ARX
   model to du; ê = q50(u_future + du) - q50(u_future) of zero-shot Chronos-2.
   Attenuation index rho = <ê, e_arx> / ||e_arx||^2 (rho < 1: the model responds less than the context fit).
3. Apply the repair iff s >= tau_s and rho < tau_rho; otherwise return the zero-shot forecast.

Grid: tau_s in {0, 0.05, 0.1, 0.2, 0.3, 0.5}, tau_rho in {0.6, 0.7, 0.8, 0.9, 1.0, inf}
(tau_s = 0, tau_rho = inf is the ungated repair). Repair: R2 (exp3/v2 gates) primary, R1 (k = 20, gamma = 2.5)
secondary, each with its own thresholds.

## Selection (simulated and semi-synthetic only)

Selection data: (a) the Exp3 selection sets used for R1 and dither (`exp3_repair.val_sets` with the exp3/v2
config: training systems, all four inputs, L in {512, 2048}, seed stream "val"); (b) semi-synthetic pairs on EPF backgrounds as in Exp6 part "semi", but with
kappa in {0.1, 0.25, 0.5, 1} and systems F1 (static gain 1), F2_a0.8, F3_a0.8_d4, F4_P24_z0.2, so that small
effects on a real background, where amplifying is not expected to pay, are represented; validation windows
are those of Exp6 shifted by 13 weeks, so they do not coincide with the Exp6 windows.
Criterion (the R1 selection criterion of the paper, `repair.objective_np` with lambda = 1): mean over pairs of
||ê - e||^2 / ||e||^2 plus the context-std-scaled quantile loss of the factual forecast, computed on (a) and on
(b) and averaged with equal weight. Lowest
criterion wins; ties within 0.002 go to the more conservative setting (larger tau_s, then smaller tau_rho).

## Confirmatory test (DaISy, not yet downloaded)

Inclusion rule (applied to the database descriptions before download): real measured data, at least one input,
not described as closed loop, N >= 576. Included: CD_player_arm, ballbeam, dryer, dryer2, evaporator,
exchanger, flexible_structure, flutter, glassfurnace, heating_system, robot_arm, thermic_res_wall, winding.
Excluded: cstr, destill, distill2, erie, pHdata, steamgen, mrs (simulated); powerplant (N = 200);
foetal_ecg, internet_traffic, tongue (no input). Columns as given in each description (time columns dropped);
every output is a target with all inputs as covariates.

Task and windows exactly as Exp13: H = 64, L = 512 primary (L = 2048 where N >= 2112, secondary), 64 windows
per dataset spread evenly over its outputs / records; MAE of the median, ratio to zero-shot, geometric means
over windows then datasets, bootstrap over windows within datasets (2000). Silent windows (context or horizon
std < 1% of the record std) are removed, as in the Exp13 sensitivity analysis.

- C1 (primary): gated R2 lowers the error of zero-shot at L = 512: overall MAE ratio < 1 with the 95% CI below 1.
- C2 (safety): gated R2 has no more datasets with ratio > 1.01 than ungated R2, and its overall ratio is not
  above the ungated one by more than 0.02.
- C3 (replication of Exp13 P1): ungated R2 overall MAE ratio < 1 with the CI below 1 at L = 512.

## Descriptive checks on data already seen (not confirmatory)

With the same fixed thresholds: Exp3 synthetic test (g, NRMSE, error ratio); Exp13 rigs (MAE ratio); LCL
(all four Exp12 cells: g, beta, relative effects, sign agreement, MAE ratio; expected: the gate mostly stays
off); fev-bench 28 tasks (WQL ratio; expected about 1). Share of windows where the gate fires is reported
everywhere.

## Stop rule

If C3 fails, the real-world accuracy claim of Exp13 does not replicate and the real-world line of work is
stopped. If C3 holds and C1 or C2 fails, the gate is not claimed; ungated results are reported as they are.

## Implementation notes (added 2026-11-21 with the code, before any Exp14 model run)

Details the text above leaves open, fixed in `src/covattn/gate.py` and `scripts/exp14_gate.py`. The code was
checked only with unit tests and a dry run with a stand-in linear "model"; Chronos-2 has not been run on DaISy.
- One gate decision per window, from the factual context and future input; a what-if pair uses it for both runs.
- Gate statistics only (model inputs unchanged): leading positions are dropped until the target and all covariates
  are finite (fev left padding), remaining NaNs are filled by linear interpolation (LCL gaps).
- A context too short for every `fit_arx` order, or an unfittable one, gives s = 0 (gate off). rho is undefined
  (gate off unless tau_rho = inf) when the probe or the ARX response is zero (constant input in the context).
- Selection (b): the objective_np sums are pooled over all (window, step / pulse) pairs of all semi cells, so that
  cells with different L can be combined; (a) is the mean over the Exp3 val batches, as in Exp3.
- Descriptive parts rerun zero-shot only (for the probe difference); the repaired and zero-shot forecasts they
  score are the stored ones of Exp3 / 6 / 12 / 13, and every regenerated window is checked against the stored
  y_true before use.
- DaISy heating_system: the description gives no column order; the file has 3 columns and column 1 is equally
  spaced with step 2.0 (= the stated sampling interval), so it is time, then u, then y
  (`scripts/daisy_prepare.py`). CD_player_arm (N = 2048) has no L = 2048 cell (needs N >= 2112).
