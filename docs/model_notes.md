# Chronos-2 source-verification notes (stage 0)

Verified against the installed package **chronos-forecasting 2.3.2** (`chronos/chronos2/*.py`,
`chronos/chronos_bolt.py`). Line numbers refer to that version. Config values and empirical checks are
in `docs/model_facts.json` (produced by `scripts/verify_model.py`); this file states the claims and where
each one comes from. All hooks in later stages are built on these notes.

## 1. Input format (pipeline)

* `Chronos2Pipeline.predict(inputs, prediction_length, batch_size, context_length, cross_learning=False)`
  (`pipeline.py:468`). We use the list-of-dicts form:
  `{"target": (L,), "past_covariates": {"u": (L,)}, "future_covariates": {"u": (H,)}}`.
  `future_covariates` keys must be a subset of `past_covariates` keys (`preprocess.py:562`).
* `preprocess.from_list_of_dicts` → `_build_prepared_inputs` (`preprocess.py:369`) builds, per task, a
  `context` of shape `(n_targets + n_covariates, L)` with **targets first**, then past-only covariates, then
  known-future covariates (row order `preprocess.py:420-422`), and a `future_covariates` of shape
  `(rows, H)` whose target rows are NaN (`preprocess.py:464`).
* `Chronos2Dataset._build_batch` (`dataset.py:244`) concatenates tasks along the batch axis; all rows of
  one task share a `group_id`; `target_idx_ranges` selects the target rows of the output. No
  normalisation happens in the pipeline / dataset / preprocess layers.
* Return value: list (one per task) of tensors `(n_targets, Q, H)` (`pipeline.py:718`).
* In our wrapper (`model.py`) the target is always row 0 of its group, the covariate(s) rows 1..K.

## 2. Normalisation — **future covariates do NOT enter loc/scale**

* `Chronos2Model._prepare_patched_context` (`model.py:392`): `context, loc_scale = self.instance_norm(context)`
  on the `(rows, L)` context only. `InstanceNorm.forward` (`chronos_bolt.py:105`) computes per row
  `loc = nanmean(x)`, `scale = sqrt(nanmean((x-loc)^2))` (population std; `scale==0 → eps=1e-5`), then
  optional `arcsinh` (controlled by `use_arcsinh`, see facts json).
* `_prepare_patched_future` (`model.py:444`): `future_covariates, _ = self.instance_norm(future_covariates, loc_scale)`
  — future covariates are standardised with the **context** loc/scale of the same row; nothing is
  recomputed from future values.
* Consequence for this study: changing the future covariate (`u_cf = u_fac + Δu` on the horizon) leaves
  every row's loc/scale unchanged. **No normalisation confound; no special mean/variance-preserving
  intervention is needed.** Empirical check in `model_facts.json`:
  `ctx_embedding_input_identical_when_only_future_cov_changes = true` and
  `future_cov_normalised_with_context_loc_scale_maxabs_err ≈ 0` (a +100 shift of the future covariate).
* Note the arcsinh: if `use_arcsinh=True`, normalised values pass through `arcsinh`, i.e. the model sees
  a *compressed* Δu for large standardised values. This is part of the model, not an artefact of ours,
  but it matters for the linearity check (δ sweep).
* Output: `quantile_preds` are un-scaled with the **target row's** context loc/scale
  (`model.py:762`, `InstanceNorm.inverse`, incl. `sinh` if arcsinh). Since fac/cf share the context,
  the same affine (or sinh) inverse map applies to both.

## 3. Token sequence and REG token

Per row (`model.encode`, `model.py:569-635`):

```
[ctx patch 0 ... ctx patch n_ctx-1] [REG] [future patch 0 ... future patch n_fut-1]
 n_ctx = ceil(L/16)  (left NaN-padding to a multiple of 16, patch mask = 0 there)
 REG index = n_ctx ;  future patches at n_ctx+1 ... n_ctx+n_fut ;  n_fut = ceil(H/16)
```

* Each patch embedding input = concat(`time_enc` (16), `values` (16), `mask` (16)) → `input_patch_embedding`
  (ResidualBlock 48 → d_ff → d_model). Context time encoding `[-C..-1]/time_encoding_scale`, future
  `[0..h-1]/time_encoding_scale`.
* REG = `self.shared` embedding row `reg_token_id = 1` (`model.py:232, 601-607`), appended *after* the
  context patches, *before* the future patches. It is present only if `use_reg_token` (true for
  `amazon/chronos-2`, see facts json). Its attention mask is 1.
* **Target future patches**: values = 0 and mask = 0 (target future is NaN → masked,
  `model.py:458-461`). **Covariate future patches**: standardised known values, mask = 1.
  So the target's future tokens differ from each other only by the time encoding; all
  covariate-specific future information has to arrive through attention.
* Output head: `output_patch_embedding` applied to the last `n_fut` hidden states of every row
  (`model.py:731`), reshaped `b n (q p) -> b q (n p)`. Predictions are produced for covariate rows too;
  the pipeline discards them.

## 4. Encoder blocks, attention modules, masks

* `Chronos2Model.encoder.block[i]` (`Chronos2EncoderBlock`, `model.py:48`), `i = 0..num_layers-1`, each with
  `layer = ModuleList([TimeSelfAttention, GroupSelfAttention, FeedForward])`:
  * `block[i].layer[0]` **TimeSelfAttention**: pre-RMSNorm (`layer_norm`, T5 style, no mean/bias) →
    `self_attention` (`MHA`, `use_rope=True`) → residual add. Attention along the time axis within a row.
  * `block[i].layer[1]` **GroupSelfAttention**: transposes `(batch, time, d) -> (time, batch, d)`, pre-norm,
    `self_attention` (`MHA`, `use_rope=False`), residual add, transpose back. At each patch index `t`
    every row attends to rows of **its own group** at the **same** `t`.
  * `block[i].layer[2]` **FeedForward**: pre-norm → `mlp` (`wi` d_model→d_ff, act, `wo`) → residual.
  * Final `encoder.final_layer_norm`.
* `MHA` (`layers.py:191`): `q, k, v, o` are `nn.Linear(bias=False)`, `inner_dim = n_heads * d_kv`.
  Heads are split as `b s (h d) -> b h s d`, i.e. **head h owns columns `h*d_kv:(h+1)*d_kv` of the input of
  `o`** (and rows of q/k/v). **No 1/sqrt(d) scaling** (eager: plain `QK^T + mask`; SDPA called with
  `scale=1.0`). Default implementation is SDPA (no weights returned); `output_attentions=True` forces the
  eager path.
* RoPE (time attention only): `Chronos2RotaryEmbedding` over `d_kv`, `position_ids = arange(seq_len)`
  (shared by context, REG and future tokens), base `rope_theta`.
* Time mask (`_expand_and_invert_time_attention_mask`): `(rows, 1, 1, seq)` additive, `finfo.min` on
  fully-missing context patches. **Bidirectional** (no causal mask): future tokens attend to context,
  REG and other future tokens, and context tokens attend to the future tokens as well.
* Group mask (`_construct_and_invert_group_time_mask`): `(seq, 1, rows, rows)`; key row `b` is visible to
  query row `q` at time `t` iff `group_ids[q]==group_ids[b]` and `attention_mask[b, t]==1`.
  Future positions always have mask 1, so **at future positions the target attends to the covariate rows
  (and itself) even though the target's own future token carries no value.**
* Attention-weight outputs: time `(rows, heads, seq, seq)`, group `(seq, heads, rows, rows)`
  (shapes recorded in facts json).

## 5. Output quantiles

* `chronos_config.quantiles` (list in facts json; Q = its length, the median index is
  `quantiles.index(0.5)`). `quantile_preds`: `(rows, Q, n_fut*16)`; the pipeline truncates to H.
* `max_output_patches` (facts json) bounds single-pass horizons; longer horizons use an autoregressive
  unrolling heuristic (`pipeline.py:419`). For H = 64 / 128 we stay within one forward pass if
  `max_output_patches >= 8` — see facts json; `forward_raw` refuses otherwise.

## 6. Hook points for later stages (stage 1)

| what | module | how |
|---|---|---|
| time-attn output (pre-residual) | `encoder.block[l].layer[0].self_attention` | forward hook on `MHA` output `hidden_states` |
| group-attn per-head output before `o` | `encoder.block[l].layer[1].self_attention.o` | forward-pre-hook on `o` (input `(seq, rows, h*d_kv)`) |
| group-attn split by key source | re-implement `MHA.forward` for `layer[1]` via wrapper (weights × V per key row) | monkey-patch with switch |
| MLP output | `encoder.block[l].layer[2].mlp` | forward hook |
| residual stream | `encoder.block[l]` output | forward hook |
| REG logit bias | time-attn of `layer[0]`, query = target future rows, key = REG index | wrapper adds bias to the additive mask |

Univariate guarantee (repair stage): when a group contains only the target row, the group-attention
softmax has a single key, so any term "target-query × covariate-key" is identically zero.
