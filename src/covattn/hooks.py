"""Hooked re-execution of Chronos2Model with read / replace / rescale access to internals.

The runner calls the installed model's own sub-modules and helper methods in the same order as
`Chronos2Model.forward` (see docs/model_notes.md); nothing in the package is modified. With an
empty `HookSpec` the official modules are called unchanged (SDPA path), so the output equals the
official forward up to float error (tested in tests/test_hooks.py).

Components (all recorded / patched at **target rows** only, shape conventions):
  time_out[l]  : time-attention output before the residual add     (N, S, d)
  group_out[l] : group-attention output (after W_O) before residual  (N, S, d)
  mlp_out[l]   : feed-forward output before residual                (N, S, d)
  cross[l]     : per-head group-attention context from covariate keys, before W_O   (S, H, N, dk)
  self[l]      : per-head group-attention context from the target's own key          (S, H, N, dk)
  head[l]      : per-head group-attention context of the target query = self + cross (S, H, N, dk)
  block_in[l]  : residual stream entering block l, all rows          (B, S, d)  (for restarting at l)
  resid_fut[l] : residual stream after block l at target future positions (N, n_fut, d)
  time_w[l]    : time-attention mass of target future queries on [REG, future, context] (N, H, 3)
  group_w[l]   : group-attention mass of target queries on covariate keys              (S, H, N)

Exact decomposition: for a target query at patch t and head h,
  ctx_h = sum_j A[t,h,i,j] v[t,h,j] = self (j=i) + cross (j covariate rows of the group),
and group_out = W_O [ctx_1; ...; ctx_H], so per-head contributions are W_O[:, h] ctx_h.
With one target per group (our setting) there is no third term.

Note (softmax competition): covariate keys also enter the softmax denominator, so a covariate
change alters the target's *self* weight A[t,h,i,i] too. Patching only `cross` therefore does not
transmit the full covariate influence; `head` (self + cross) does. Zeroing `cross` removes every
covariate *value* read by the target, but not the normalisation effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from einops import rearrange

POSITION_SETS = ("future", "context", "all", "reg")


@dataclass
class Patch:
    component: str  # time_out | group_out | mlp_out | cross
    layer: int
    positions: str = "future"
    heads: list[int] | None = None  # cross only; None = all heads
    source: dict | None = None  # record dict of the source run (uses source[component][layer])


@dataclass
class HookSpec:
    record: set[str] = field(default_factory=set)
    patches: list[Patch] = field(default_factory=list)
    cross_gain: torch.Tensor | None = None  # (n_layers, n_heads), multiplies the cross part of target queries
    cross_gain_positions: str = "all"
    reg_bias: torch.Tensor | None = None  # (n_layers,), added to time-attn logits (target future query -> REG)
    start_layer: int = 0
    start_hidden: torch.Tensor | None = None  # block-`start_layer` input (B, S, d); skips earlier computation


@dataclass
class Prepared:
    emb: torch.Tensor  # (B, S, d) input embeddings
    tmask: torch.Tensor  # (B, 1, 1, S) additive time mask
    gmask: torch.Tensor  # (S, 1, B, B) additive group mask
    pos_ids: torch.Tensor
    loc_scale: tuple[torch.Tensor, torch.Tensor]
    tgt: torch.Tensor  # (N,) target row indices
    covkey: torch.Tensor  # (B,) 1.0 for covariate rows, 0.0 for target rows
    n_ctx: int
    n_fut: int
    H_out: int  # requested horizon

    @property
    def S(self) -> int:
        return self.emb.shape[1]

    @property
    def reg(self) -> int:
        return self.n_ctx

    def pos(self, which: str) -> torch.Tensor:
        dev = self.emb.device
        if which == "future":
            return torch.arange(self.n_ctx + 1, self.S, device=dev)
        if which == "context":
            return torch.arange(0, self.n_ctx, device=dev)
        if which == "reg":
            return torch.tensor([self.n_ctx], device=dev)
        if which == "all":
            return torch.arange(0, self.S, device=dev)
        raise ValueError(which)


class HookedChronos2:
    def __init__(self, model) -> None:
        self.m = model
        self.n_layers = model.config.num_layers
        self.n_heads = model.config.num_heads
        self.dk = model.config.d_kv
        self.Q = len(model.chronos_config.quantiles)
        self.P = model.chronos_config.output_patch_size

    # ------------------------------------------------------------------ inputs
    @torch.no_grad()
    def prepare(self, inp: dict) -> Prepared:
        """inp = Chronos2Wrapper.build_model_inputs(batch) (+ 'rows', 'H')."""
        m = self.m
        context, fut, gids = inp["context"], inp["future_covariates"], inp["group_ids"]
        nop, rows = int(inp["num_output_patches"]), int(inp["rows"])
        B = context.shape[0]
        patched_context, attention_mask, loc_scale = m._prepare_patched_context(context=context, context_mask=None)
        n_ctx = attention_mask.shape[-1]
        emb = m.input_patch_embedding(patched_context)
        assert m.chronos_config.use_reg_token, "layout assumes a REG token (true for amazon/chronos-2)"
        reg_ids = torch.full((B, 1), m.config.reg_token_id, device=emb.device)
        emb = torch.cat([emb, m.shared(reg_ids)], dim=-2)
        attention_mask = torch.cat([attention_mask.to(m.dtype), torch.ones_like(reg_ids).to(m.dtype)], dim=-1)
        patched_future, _ = m._prepare_patched_future(
            future_covariates=fut, future_covariates_mask=None, loc_scale=loc_scale, num_output_patches=nop,
            batch_size=B,
        )
        emb = torch.cat([emb, m.input_patch_embedding(patched_future)], dim=-2)
        attention_mask = torch.cat([attention_mask, torch.ones(B, nop, dtype=m.dtype, device=emb.device)], dim=-1)
        enc = m.encoder
        tmask = enc._expand_and_invert_time_attention_mask(attention_mask, emb.dtype)
        gmask = enc._construct_and_invert_group_time_mask(gids, attention_mask, emb.dtype)
        S = emb.shape[1]
        tgt = torch.arange(0, B, rows, device=emb.device)
        covkey = torch.ones(B, device=emb.device, dtype=emb.dtype)
        covkey[tgt] = 0.0
        return Prepared(emb=emb, tmask=tmask, gmask=gmask,
                        pos_ids=torch.arange(S, device=emb.device).unsqueeze(0), loc_scale=loc_scale, tgt=tgt,
                        covkey=covkey, n_ctx=n_ctx, n_fut=nop, H_out=int(inp.get("H", nop * self.P)))

    # ------------------------------------------------------------------ main loop
    @torch.no_grad()
    def run(self, P: Prepared, spec: HookSpec | None = None) -> tuple[torch.Tensor, dict]:
        """Returns (target quantiles (N, Q, H) in data units, record dict)."""
        spec = spec or HookSpec()
        rec: dict = {k: {} for k in spec.record}
        if spec.start_hidden is not None:
            h, l0 = spec.start_hidden, spec.start_layer
        else:
            h, l0 = P.emb, 0
        blocks = self.m.encoder.block
        for l in range(l0, self.n_layers):
            blk = blocks[l]
            if "block_in" in rec:
                rec["block_in"][l] = h.clone()
            h = self._time(l, blk.layer[0], h, P, spec, rec)
            h = self._group(l, blk.layer[1], h, P, spec, rec)
            h = self._mlp(l, blk.layer[2], h, P, spec, rec)
            if "resid_fut" in rec:
                rec["resid_fut"][l] = h[P.tgt][:, P.n_ctx + 1 :].clone()
        q = self.decode(h[P.tgt][:, P.n_ctx + 1 :], P, final_norm=True)
        return q, rec

    @torch.no_grad()
    def decode(self, fut_hidden: torch.Tensor, P: Prepared, final_norm: bool = True) -> torch.Tensor:
        """Output head on target future hidden states (N, n_fut, d) -> quantiles (N, Q, H) in data units.
        Applying it to intermediate residual streams gives a 'logit-lens' style readout."""
        m = self.m
        x = m.encoder.final_layer_norm(fut_hidden) if final_norm else fut_hidden
        qp = m.output_patch_embedding(x)
        n = fut_hidden.shape[1]
        qp = rearrange(qp, "b n (q p) -> b (q n p)", n=n, q=self.Q, p=self.P)
        loc, scale = P.loc_scale
        qp = m.instance_norm.inverse(qp, (loc[P.tgt], scale[P.tgt]))
        qp = rearrange(qp, "b (q h) -> b q h", q=self.Q)
        return qp[..., : P.H_out].float()

    # ------------------------------------------------------------------ helpers
    def _apply(self, comp: str, l: int, a: torch.Tensor, P: Prepared, spec: HookSpec, rec: dict) -> torch.Tensor:
        """Record / patch a (B, S, d) component at target rows."""
        if comp in rec:
            rec[comp][l] = a[P.tgt].clone()
        for p in spec.patches:
            if p.component == comp and p.layer == l:
                pos = P.pos(p.positions)
                src = p.source[comp][l]
                a = a.clone()
                a[P.tgt[:, None], pos[None, :]] = src[:, pos]
        return a

    def _time(self, l: int, mod, h: torch.Tensor, P: Prepared, spec: HookSpec, rec: dict) -> torch.Tensor:
        normed = mod.layer_norm(h)
        mask = P.tmask
        if spec.reg_bias is not None and float(spec.reg_bias[l]) != 0.0:
            B, S = h.shape[:2]
            bias = torch.zeros(B, 1, S, S, device=h.device, dtype=mask.dtype)
            fut = P.pos("future")
            bias[P.tgt[:, None], 0, fut[None, :], P.reg] = float(spec.reg_bias[l])
            mask = mask + bias
        need_w = "time_w" in rec
        out = mod.self_attention(normed, position_ids=P.pos_ids, mask=mask, output_attentions=need_w)
        a = out.hidden_states
        if need_w:
            w = out.attn_weights[P.tgt][:, :, P.n_ctx + 1 :, :]  # (N, H, n_fut, S)
            reg = w[..., P.reg]
            futm = w[..., P.n_ctx + 1 :].sum(-1)
            ctxm = w[..., : P.n_ctx].sum(-1)
            rec["time_w"][l] = torch.stack([reg, futm, ctxm], -1).mean(2)  # (N, H, 3)
        a = self._apply("time_out", l, a, P, spec, rec)
        return h + mod.dropout(a)

    def _needs_decomp(self, l: int, spec: HookSpec, rec: dict) -> bool:
        if any(k in rec for k in ("cross", "self", "head", "group_w")):
            return True
        if any(p.component in ("cross", "head") and p.layer == l for p in spec.patches):
            return True
        return spec.cross_gain is not None and bool((spec.cross_gain[l] != 1.0).any())

    def _group(self, l: int, mod, h: torch.Tensor, P: Prepared, spec: HookSpec, rec: dict) -> torch.Tensor:
        x = rearrange(h, "b t d -> t b d")
        normed = mod.layer_norm(x)
        mha = mod.self_attention
        if not self._needs_decomp(l, spec, rec):
            a = mha(normed, mask=P.gmask).hidden_states  # official path
        else:
            Hh = self.n_heads
            q = rearrange(mha.q(normed), "t b (h k) -> t h b k", h=Hh)
            k = rearrange(mha.k(normed), "t b (h k) -> t h b k", h=Hh)
            v = rearrange(mha.v(normed), "t b (h k) -> t h b k", h=Hh)
            scores = torch.matmul(q, k.transpose(-1, -2)) + P.gmask  # no 1/sqrt(d) scaling (as in Chronos-2)
            A = torch.softmax(scores.float(), dim=-1).type_as(scores)  # (T, H, B, B)
            ctx = torch.matmul(A, v)  # (T, H, B, dk)
            At = A[:, :, P.tgt, :]  # target queries (T, H, N, B)
            A_cross = At * P.covkey
            cross = torch.matmul(A_cross, v)  # (T, H, N, dk)
            if "group_w" in rec:
                rec["group_w"][l] = A_cross.sum(-1).clone()
            if "cross" in rec:
                rec["cross"][l] = cross.clone()
            if "self" in rec:
                a_self = torch.gather(At, -1, P.tgt.view(1, 1, -1, 1).expand(At.shape[0], Hh, -1, 1))
                rec["self"][l] = a_self * v[:, :, P.tgt, :]
            ctx_t = ctx[:, :, P.tgt]  # (T, H, N, dk) copy (advanced indexing)
            if "head" in rec:
                rec["head"][l] = ctx_t.clone()
            changed = False
            cur_cross = cross
            if spec.cross_gain is not None and bool((spec.cross_gain[l] != 1.0).any()):
                fac = torch.ones(cross.shape[0], Hh, 1, 1, device=h.device, dtype=cross.dtype)
                fac[P.pos(spec.cross_gain_positions)] = spec.cross_gain[l].to(device=h.device, dtype=cross.dtype).view(1, Hh, 1, 1)
                cur_cross = cross * fac
                ctx_t = ctx_t + (cur_cross - cross)
                changed = True
            for p in spec.patches:
                if p.layer != l or p.component not in ("cross", "head"):
                    continue
                pos = P.pos(p.positions)
                heads = torch.arange(Hh, device=h.device) if p.heads is None else torch.as_tensor(p.heads, device=h.device)
                ix = (pos[:, None], heads[None, :])
                src = p.source[p.component][l]
                if p.component == "cross":  # swap only the value-from-covariate term
                    ctx_t[ix] = ctx_t[ix] + (src[ix] - cur_cross[ix])
                else:  # whole head output of the target (self + cross)
                    ctx_t[ix] = src[ix]
                changed = True
            if changed:
                ctx = ctx.clone()
                ctx[:, :, P.tgt] = ctx_t
            a = mha.o(rearrange(ctx, "t h b k -> t b (h k)"))
        a = rearrange(a, "t b d -> b t d")
        a = self._apply("group_out", l, a, P, spec, rec)
        return h + mod.dropout(a)

    def _mlp(self, l: int, mod, h: torch.Tensor, P: Prepared, spec: HookSpec, rec: dict) -> torch.Tensor:
        f = mod.mlp(mod.layer_norm(h))
        f = self._apply("mlp_out", l, f, P, spec, rec)
        return h + mod.dropout(f)

    # ------------------------------------------------------------------ W_O per head
    def head_out_weight(self, l: int) -> torch.Tensor:
        """(H, d, dk) slices of the group-attention output projection of layer l."""
        W = self.m.encoder.block[l].layer[1].self_attention.o.weight  # (d, H*dk)
        return rearrange(W, "d (h k) -> h d k", h=self.n_heads)
