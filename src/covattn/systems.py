"""Synthetic forced systems with exact counterfactual pairs.

Observation model:  y_t = x_t + s_t + eps_t
  x_t   : state driven by covariate u_t (families F1-F5)
  s_t   : background independent of u (random sinusoidal season + slow trend / random walk)
  eps_t : measurement noise (std = 1/snr, relative to std(x) on the context = 1)

A counterfactual pair shares the system, all noise realisations and the whole context;
only the *future* covariate is changed: u_cf[L:] = u_fac[L:] + du. The true effect
e = y_cf - y_fac is obtained by re-simulation with the same noise. For linear families it
also equals conv(impulse_response, du), which tests cross-check.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from scipy.signal import lfilter

Family = Literal["F1", "F2", "F3", "F4", "F5"]
InputType = Literal["prbs", "multisine", "ar1", "steps"]
Shape = Literal["step", "pulse", "ramp"]
SPLIT_OFFSETS = {"train": 0, "val": 1, "test": 2}
PATCH = 16


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SystemSpec:
    """One system configuration (family + dynamics parameters)."""

    family: Family
    a: float = 0.0  # pole for F2/F3/F5
    d: int = 0  # input delay for F3
    period: float = 0.0  # natural period (steps) for F4
    zeta: float = 0.0  # damping ratio for F4
    proc_noise: float = 0.0  # std of process noise w_t (relative to unit-std x)

    @property
    def key(self) -> str:
        if self.family == "F1":
            return "F1"
        if self.family == "F2":
            return f"F2_a{self.a:g}"
        if self.family == "F3":
            return f"F3_a{self.a:g}_d{self.d}"
        if self.family == "F4":
            return f"F4_P{self.period:g}_z{self.zeta:g}"
        return f"F5_a{self.a:g}"

    @property
    def is_linear(self) -> bool:
        return self.family != "F5"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SampleSpec:
    """Everything except the system that defines one factual sample."""

    context_length: int = 512
    horizon: int = 64
    input_type: InputType = "prbs"
    snr: float = 10.0  # std(x_ctx)/std(eps); inf -> no measurement noise
    bg_scale: float = 1.0  # amplitude scale of the background s_t
    burn_in: int = 500


@dataclass
class Sample:
    """A factual sample plus everything needed to build counterfactuals."""

    system: SystemSpec
    spec: SampleSpec
    u: np.ndarray  # (T,) factual covariate over [0, L+H)
    y: np.ndarray  # (T,) factual observation
    x: np.ndarray  # (T,) factual state
    s: np.ndarray  # (T,) background
    eps: np.ndarray  # (T,) measurement noise
    w: np.ndarray  # (T,) process noise (already scaled)
    b: float  # input gain (normalised so std(x_ctx) = 1)
    u_burn: np.ndarray  # covariate during burn-in (needed to re-simulate)
    w_burn: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def L(self) -> int:
        return self.spec.context_length

    @property
    def H(self) -> int:
        return self.spec.horizon


# --------------------------------------------------------------------------------------
# seeding helpers
# --------------------------------------------------------------------------------------
def stable_hash(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def make_rng(base_seed: int, split: str, *keys: str | int) -> np.random.Generator:
    """Deterministic RNG from (seed, split, keys). Train/val/test use disjoint streams."""
    entropy = [int(base_seed), SPLIT_OFFSETS[split]] + [
        stable_hash(k) if isinstance(k, str) else int(k) for k in keys
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


# --------------------------------------------------------------------------------------
# input signals
# --------------------------------------------------------------------------------------
def gen_input(kind: InputType, T: int, rng: np.random.Generator) -> np.ndarray:
    if kind.startswith("mix"):
        # "mix<a>": a * multisine + (1 - a) * PRBS (each standardised), interpolating between the two regimes (Exp9)
        a = float(kind[3:])
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"mixing weight out of [0, 1]: {kind}")
        u = a * gen_input("multisine", T, rng) + (1.0 - a) * gen_input("prbs", T, rng)
        sd = u.std()
        return (u - u.mean()) / (sd if sd > 0 else 1.0)
    if kind == "prbs":
        hold = rng.uniform(4, 24)  # mean hold length
        switches = rng.random(T) < 1.0 / hold
        level = np.cumsum(switches) % 2
        u = 2.0 * level - 1.0
    elif kind == "multisine":
        k = int(rng.integers(3, 6))
        freqs = np.exp(rng.uniform(np.log(1 / 256), np.log(1 / 8), size=k))
        phases = rng.uniform(0, 2 * np.pi, size=k)
        amps = rng.uniform(0.5, 1.0, size=k)
        t = np.arange(T)
        u = (amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * t[None, :] + phases[:, None])).sum(0)
    elif kind == "ar1":
        rho = rng.uniform(0.8, 0.98)
        eta = rng.standard_normal(T) * np.sqrt(1 - rho**2)
        u = lfilter([1.0], [1.0, -rho], eta)
    elif kind == "steps":
        hold = rng.uniform(20, 80)
        switches = rng.random(T) < 1.0 / hold
        seg = np.cumsum(switches)
        levels = rng.standard_normal(seg[-1] + 1)
        u = levels[seg]
    else:
        raise ValueError(f"unknown input type {kind}")
    u = np.asarray(u, dtype=np.float64)
    sd = u.std()
    return (u - u.mean()) / (sd if sd > 0 else 1.0)


def gen_background(T: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    t = np.arange(T)
    period = rng.uniform(16, 128)
    phase = rng.uniform(0, 2 * np.pi)
    amp = rng.uniform(0.3, 1.0)
    season = amp * np.sin(2 * np.pi * t / period + phase)
    if rng.random() < 0.5:
        trend = rng.uniform(-1.0, 1.0) * t / T  # slow linear trend, total drift <= 1
    else:
        trend = np.cumsum(rng.standard_normal(T)) * 0.02  # slow random walk
    level = rng.uniform(-2, 2)
    return scale * (season + trend) + level


# --------------------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------------------
def f4_coeffs(period: float, zeta: float) -> tuple[float, float]:
    """Discrete AR(2) coefficients for an under-damped oscillator (pole mapping z = exp(s))."""
    wn = 2 * np.pi / period
    r = np.exp(-zeta * wn)
    theta = wn * np.sqrt(max(1 - zeta**2, 1e-12))
    return 2 * r * np.cos(theta), -(r**2)


def transfer_coeffs(sys: SystemSpec) -> tuple[np.ndarray, np.ndarray]:
    """(B, A) for x = B(q)/A(q) u (unit input gain) for linear families; F5 uses the F2 part."""
    if sys.family == "F1":
        return np.array([1.0]), np.array([1.0])
    if sys.family in ("F2", "F5"):
        return np.array([1.0]), np.array([1.0, -sys.a])
    if sys.family == "F3":
        B = np.zeros(sys.d + 1)
        B[sys.d] = 1.0
        return B, np.array([1.0, -sys.a])
    if sys.family == "F4":
        a1, a2 = f4_coeffs(sys.period, sys.zeta)
        return np.array([1.0]), np.array([1.0, -a1, -a2])
    raise ValueError(sys.family)


def simulate_state(sys: SystemSpec, u: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """x_t from u_t (whole series incl. burn-in), zero initial state. w enters additively in the
    state recursion (process noise), so it is filtered by 1/A(q)."""
    B, A = transfer_coeffs(sys)
    drive = np.tanh(u) if sys.family == "F5" else u
    return lfilter(B * b, A, drive) + lfilter([1.0], A, w)


def impulse_response(sys: SystemSpec, n: int, b: float = 1.0) -> np.ndarray:
    B, A = transfer_coeffs(sys)
    imp = np.zeros(n)
    imp[0] = 1.0
    return lfilter(B * b, A, imp)


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------
def make_sample(sys: SystemSpec, spec: SampleSpec, rng: np.random.Generator) -> Sample:
    L, H, burn = spec.context_length, spec.horizon, spec.burn_in
    T_all = burn + L + H
    u_all = gen_input(spec.input_type, T_all, rng)
    w_all = rng.standard_normal(T_all) * sys.proc_noise
    # normalise the input gain so that the forced response has unit std on the context
    x_unit = simulate_state(sys, u_all, np.zeros(T_all), 1.0)
    sd = x_unit[burn : burn + L].std()
    b = 1.0 / sd if sd > 1e-8 else 1.0
    x_all = simulate_state(sys, u_all, w_all, b)
    x = x_all[burn:]
    s = gen_background(L + H, rng, spec.bg_scale)
    eps_std = 0.0 if not np.isfinite(spec.snr) else 1.0 / spec.snr
    eps = rng.standard_normal(L + H) * eps_std
    y = x + s + eps
    return Sample(
        system=sys,
        spec=spec,
        u=u_all[burn:].copy(),
        y=y,
        x=x,
        s=s,
        eps=eps,
        w=w_all[burn:].copy(),
        b=b,
        u_burn=u_all[:burn].copy(),
        w_burn=w_all[:burn].copy(),
    )


def delta_u(shape: Shape, H: int, delta: float) -> np.ndarray:
    du = np.zeros(H)
    if shape == "step":
        du[:] = delta
    elif shape == "pulse":
        du[:PATCH] = delta
    elif shape == "ramp":
        du[:] = delta * np.arange(1, H + 1) / H
    else:
        raise ValueError(shape)
    return du


@dataclass
class Counterfactual:
    shape: Shape
    delta_mult: float  # delta / std(u_ctx)
    du: np.ndarray  # (H,)
    u_future_cf: np.ndarray  # (H,)
    y_future_cf: np.ndarray  # (H,)
    effect: np.ndarray  # (H,) exact y_cf - y_fac by re-simulation


def make_counterfactual(sample: Sample, shape: Shape, delta_mult: float) -> Counterfactual:
    L, H = sample.L, sample.H
    std_u = sample.u[:L].std()
    du = delta_u(shape, H, delta_mult * std_u)
    u_full_fac = np.concatenate([sample.u_burn, sample.u])
    u_full_cf = u_full_fac.copy()
    u_full_cf[-H:] += du
    w_full = np.concatenate([sample.w_burn, sample.w])
    x_cf = simulate_state(sample.system, u_full_cf, w_full, sample.b)[len(sample.u_burn) :]
    y_cf = x_cf + sample.s + sample.eps
    effect = y_cf[L:] - sample.y[L:]
    return Counterfactual(
        shape=shape,
        delta_mult=delta_mult,
        du=du,
        u_future_cf=u_full_cf[-H:].copy(),
        y_future_cf=y_cf[L:].copy(),
        effect=effect,
    )


def effect_by_convolution(sample: Sample, du: np.ndarray) -> np.ndarray:
    """Linear-system effect = impulse response * du (zero-state response). Cross-check only."""
    h = impulse_response(sample.system, len(du), sample.b)
    return np.convolve(h, du)[: len(du)]


# --------------------------------------------------------------------------------------
# standard grids
# --------------------------------------------------------------------------------------
def default_systems(include_f5: bool = True) -> list[SystemSpec]:
    out: list[SystemSpec] = [SystemSpec("F1")]
    out += [SystemSpec("F2", a=a) for a in (0.5, 0.8, 0.95, 0.99)]
    out += [SystemSpec("F3", a=0.8, d=d) for d in (1, 4, 16, 32)]
    out += [SystemSpec("F4", period=p, zeta=z) for p in (8, 24, 96) for z in (0.05, 0.2, 0.5)]
    if include_f5:
        out += [SystemSpec("F5", a=0.8)]
    return out


def quick_systems() -> list[SystemSpec]:
    return [
        SystemSpec("F1"),
        SystemSpec("F2", a=0.8),
        SystemSpec("F2", a=0.95),
        SystemSpec("F3", a=0.8, d=4),
        SystemSpec("F4", period=24, zeta=0.2),
    ]
