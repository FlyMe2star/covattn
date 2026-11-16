"""Stage-0 source verification: dump Chronos-2 config / module tree and check empirically
  (1) loc/scale of every row come from its context only (future covariates do not enter);
  (2) future covariates are normalised with the *context* loc/scale of the same row;
  (3) token layout [ctx patches][REG][future patches] and output shape (B, Q, H).
Writes docs/model_facts.json (consumed by docs/model_notes.md)."""

from __future__ import annotations

import inspect
import json
import math

import numpy as np
import torch

from covattn.io_utils import REPO_ROOT, env_info
from covattn.model import Chronos2Wrapper, CovBatch


def main() -> None:
    w = Chronos2Wrapper("amazon/chronos-2", device="auto", amp="off", batch_size=256)
    m = w.model
    cc = m.chronos_config
    facts: dict = {"env": env_info(), "chronos_module_file": inspect.getsourcefile(type(m))}
    facts["core_config"] = {k: getattr(m.config, k) for k in
                            ["d_model", "d_kv", "d_ff", "num_layers", "num_heads", "dropout_rate",
                             "feed_forward_proj", "rope_theta", "layer_norm_epsilon"]}
    facts["attn_implementation"] = m.config._attn_implementation
    facts["chronos_config"] = {k: getattr(cc, k) for k in
                               ["context_length", "input_patch_size", "output_patch_size", "input_patch_stride",
                                "quantiles", "use_reg_token", "use_arcsinh", "max_output_patches",
                                "time_encoding_scale"]}
    facts["n_params"] = sum(p.numel() for p in m.parameters())
    facts["module_names_block0"] = [n for n, _ in m.encoder.block[0].named_modules() if n]
    facts["top_level_modules"] = [n for n, _ in m.named_children()]

    # ---------------- (1)+(2): normalisation of future covariates
    rng = np.random.default_rng(0)
    L, H = 300, 64
    y = rng.standard_normal(L).cumsum() + 5
    u = rng.standard_normal(L) * 3 + 10
    uf_a = rng.standard_normal(H) * 3 + 10
    uf_b = uf_a + 100.0  # huge shift of the future covariate only
    captured: list[torch.Tensor] = []
    h = m.input_patch_embedding.register_forward_hook(lambda mod, inp, out: captured.append(inp[0].detach().clone()))
    for uf in (uf_a, uf_b):
        b = CovBatch(y[None], u[None, None], uf[None, None])
        w.forward_raw(b)
    h.remove()
    # captured: [ctx_a, fut_a, ctx_b, fut_b]; feature layout per patch = [time_enc(16), values(16), mask(16)]
    ctx_a, fut_a, ctx_b, fut_b = captured
    P = cc.input_patch_size
    facts["ctx_embedding_input_identical_when_only_future_cov_changes"] = bool(torch.equal(ctx_a, ctx_b))
    loc_u, scale_u = u.mean(), u.std()
    fut_vals_row1 = fut_a[1, :, P : 2 * P].reshape(-1)[:H].cpu().numpy()
    expected = (uf_a - loc_u) / scale_u
    if cc.use_arcsinh:
        expected = np.arcsinh(expected)
    facts["future_cov_normalised_with_context_loc_scale_maxabs_err"] = float(np.abs(fut_vals_row1 - expected).max())
    facts["future_target_row_values_zero_and_mask_zero"] = bool(
        torch.all(fut_a[0, :, P:] == 0).item()
    )
    facts["future_cov_mask_ones"] = bool(torch.all(fut_a[1, :, 2 * P :] == 1).item())
    facts["num_context_patches_for_L300"] = int(ctx_a.shape[1])
    facts["expected_ceil_L_over_P"] = math.ceil(L / P)

    # ---------------- (3): layout and output shapes
    seen: dict = {}

    def _shape_hook(mod, inp, out) -> None:  # must return None, else the output is replaced
        seen["time_attn_out_shape"] = tuple(out[0].shape)

    hk = m.encoder.block[0].layer[0].register_forward_hook(_shape_hook)
    b = CovBatch(y[None], u[None, None], uf_a[None, None])
    q, out = w.forward_raw(b, output_attentions=True)
    hk.remove()
    facts["time_attn_hidden_shape_(rows,seq,d)"] = seen["time_attn_out_shape"]
    facts["seq_len_expected_ctx+1+fut"] = math.ceil(L / P) + 1 + math.ceil(H / P)
    facts["time_attn_weights_shape_(rows,heads,q,k)"] = tuple(out.enc_time_self_attn_weights[0].shape)
    facts["group_attn_weights_shape_(seq,heads,rows,rows)"] = tuple(out.enc_group_self_attn_weights[0].shape)
    facts["quantile_preds_shape_(rows,Q,H)"] = tuple(out.quantile_preds.shape)
    # REG token: input embedding at position n_ctx equals shared embedding row reg_token_id
    facts["reg_token_id"] = getattr(m.config, "reg_token_id", None)

    # pipeline vs forward_raw equivalence on this toy input
    q_pipe = w.predict(b, H)
    facts["pipeline_vs_forward_raw_maxabs"] = float(np.abs(q_pipe - q.cpu().numpy()).max())

    out_file = REPO_ROOT / "docs" / "model_facts.json"
    out_file.write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    print(json.dumps(facts, indent=2, default=str))


if __name__ == "__main__":
    main()
