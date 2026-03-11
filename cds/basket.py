# Basket CDS / k-th to default pricing engine
# One-factor Gaussian copula, Monte Carlo simulation

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# -- Conversion utilities --

def hazard_from_spread(spread_bps: float, recovery: float) -> float:
    # Approximate hazard rate from CDS spread: λ ≈ s / (1 - R)
    lgd = 1.0 - recovery
    if lgd <= 0:
        return 0.0
    return (spread_bps / 10_000.0) / lgd


def spread_from_hazard(lam: float, recovery: float) -> float:
    # Approximate CDS spread (bps) from hazard rate: s ≈ λ * (1 - R)
    return lam * (1.0 - recovery) * 10_000.0



# -- Monte Carlo simulation --

def simulate_default_times(
    hazard_rates: list[float],
    correlation: float,
    n_simulations: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    # Simulate correlated default times via one-factor Gaussian copula
    if rng is None:
        rng = np.random.default_rng(42)

    n_names = len(hazard_rates)
    lam = np.array(hazard_rates)

    rho = np.clip(correlation, -0.999, 0.999)
    sqrt_rho2 = np.sqrt(max(1.0 - rho ** 2, 0.0))

    # Draw all random variables at once (vectorised)
    M = rng.standard_normal(n_simulations)                    # (n_sims,)
    eps = rng.standard_normal((n_simulations, n_names))       # (n_sims, n_names)

    # Correlated latent variables
    Z = rho * M[:, np.newaxis] + sqrt_rho2 * eps              # (n_sims, n_names)

    # Transform to uniform via Φ
    U = norm.cdf(Z)

    # Clip to avoid log(0)
    U = np.clip(U, 1e-15, 1.0 - 1e-15)

    # Invert exponential: τ = -ln(U) / λ
    tau = -np.log(U) / lam[np.newaxis, :]

    return tau


# -- k-th to default pricing --

def basket_pricing(
    hazard_rates: list[float],
    recoveries: list[float],
    correlation: float,
    maturity: float,
    risk_free: float,
    n_simulations: int = 50_000,
    frequency: int = 4,
    notional: float = 10_000_000.0,
) -> dict:
    # Price all k-th to default tranches for a basket
    n_names = len(hazard_rates)
    rec = np.array(recoveries)

    # Simulate default times
    tau = simulate_default_times(hazard_rates, correlation, n_simulations)

    # Sort default times per path and track which name defaults at each rank
    sorted_indices = np.argsort(tau, axis=1)               # (n_sims, n_names)
    sorted_tau = np.take_along_axis(tau, sorted_indices, axis=1)

    # Recovery of the name that defaults at each rank
    sorted_rec = rec[sorted_indices]                        # (n_sims, n_names)

    # Payment schedule for premium leg
    dt = 1.0 / frequency
    payment_dates = np.arange(dt, maturity + 1e-9, dt)
    if len(payment_dates) > 0 and payment_dates[-1] > maturity + 1e-9:
        payment_dates = payment_dates[:-1]

    results = []

    for k in range(n_names):
        tau_k = sorted_tau[:, k]                            # (n_sims,)
        rec_k = sorted_rec[:, k]                            # (n_sims,)

        defaulted = tau_k < maturity                        # bool (n_sims,)

        # -- Protection leg --
        # (1 - R_k) * N * DF(τ_k) if default before maturity
        prot_payoff = np.where(
            defaulted,
            (1.0 - rec_k) * notional * np.exp(-risk_free * tau_k),
            0.0,
        )
        protection_pv = np.mean(prot_payoff)

        # -- Premium leg (annuity for 1bp) --
        # Pay dt * N * 1bp * DF(t_i) at each payment date t_i < min(τ_k, T)
        stop_time = np.minimum(tau_k, maturity)             # (n_sims,)

        # For each path, sum DF at payment dates before stop_time
        # Vectorised: (n_sims, n_dates) bool mask
        mask = payment_dates[np.newaxis, :] <= stop_time[:, np.newaxis]
        df_dates = np.exp(-risk_free * payment_dates)       # (n_dates,)
        annuity_per_path = dt * notional * (1.0 / 10_000.0) * np.sum(
            mask * df_dates[np.newaxis, :], axis=1
        )
        annuity_1bp = np.mean(annuity_per_path)

        # -- Probability of k-th default before maturity --
        prob_default = np.mean(defaulted)

        # -- Fair spread --
        if annuity_1bp > 0:
            fair_spd = protection_pv / annuity_1bp
        else:
            fair_spd = 0.0

        results.append({
            "rank": k + 1,
            "label": _ordinal(k + 1) + "-to-default",
            "prob_default": prob_default,
            "fair_spread_bps": fair_spd,
            "premium_leg_pv": annuity_1bp * fair_spd,
            "protection_leg_pv": protection_pv,
        })

    return {
        "tranches": results,
        "tau": tau,
        "sorted_indices": sorted_indices,
    }


# -- Additional analytics --

def default_count_distribution(tau: np.ndarray, maturity: float) -> list[dict]:
    # Distribution of total defaults before maturity
    defaulted = tau < maturity                              # (n_sims, n_names)
    n_defaults_per_path = np.sum(defaulted, axis=1)         # (n_sims,)
    n_names = tau.shape[1]

    dist = []
    for k in range(n_names + 1):
        prob = np.mean(n_defaults_per_path == k)
        dist.append({"n_defaults": k, "probability": prob})
    return dist


def expected_losses(tau: np.ndarray, sorted_indices: np.ndarray,
                    recoveries: list[float], maturity: float) -> list[dict]:
    # Expected loss for each k-th to default tranche
    rec = np.array(recoveries)
    sorted_rec = rec[sorted_indices]                        # (n_sims, n_names)
    n_names = tau.shape[1]

    sorted_tau = np.take_along_axis(tau, sorted_indices, axis=1)

    results = []
    for k in range(n_names):
        tau_k = sorted_tau[:, k]
        rec_k = sorted_rec[:, k]
        defaulted = tau_k < maturity
        prob = np.mean(defaulted)
        avg_lgd = np.mean((1.0 - rec_k)[defaulted]) if np.any(defaulted) else (1.0 - np.mean(rec))
        el = prob * avg_lgd
        results.append({
            "label": _ordinal(k + 1) + "-to-default",
            "expected_loss": el,
            "prob_default": prob,
            "avg_lgd": avg_lgd,
        })
    return results


def _ordinal(n: int) -> str:
    # Convert integer to ordinal string
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
