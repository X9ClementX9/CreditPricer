# Bootstrap piecewise-constant hazard rates from market CDS spreads
# For each tenor, solve for the hazard rate that matches the quoted spread

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from cds.pricing import survival_probability_piecewise
from cds.utils import payment_schedule, accrual_fraction, discount_factors


# -- Premium leg with piecewise hazard --

def _premium_leg_pw(spread_bps, notional, maturity, hazard_rates, knots, r, frequency=4):
    spread = spread_bps / 10_000
    dt = accrual_fraction(frequency)
    dates = payment_schedule(maturity, frequency)
    df = discount_factors(r, dates)

    q = np.array([
        survival_probability_piecewise(hazard_rates, knots, t) for t in dates
    ])

    # Regular + accrual
    pv = notional * spread * dt * np.sum(df * q)
    q_prev = np.concatenate([[1.0], q[:-1]])
    delta_pd = q_prev - q
    pv += notional * spread * (dt / 2.0) * np.sum(df * delta_pd)
    return pv


# -- Protection leg with piecewise hazard --

def _protection_leg_pw(notional, maturity, hazard_rates, knots, r, recovery, integration_steps=400):
    lgd = 1.0 - recovery
    times = np.linspace(0, maturity, integration_steps + 1)

    q = np.array([
        survival_probability_piecewise(hazard_rates, knots, t) for t in times
    ])
    df = discount_factors(r, times)

    # Find which hazard rate applies at each time
    lam_t = np.zeros_like(times)
    for idx, t in enumerate(times):
        for k in range(len(hazard_rates)):
            if t <= knots[k + 1] + 1e-12:
                lam_t[idx] = hazard_rates[k]
                break

    integrand = lam_t * q * df
    pv = lgd * notional * np.trapezoid(integrand, times)
    return pv


# -- Fair spread with piecewise hazard --

def _fair_spread_pw(notional, maturity, hazard_rates, knots, r, recovery, frequency=4):
    prot = _protection_leg_pw(notional, maturity, hazard_rates, knots, r, recovery)
    rpv01 = _premium_leg_pw(1.0, notional, maturity, hazard_rates, knots, r, frequency)
    if rpv01 == 0:
        return 0.0
    return prot / rpv01


# -- Bootstrap --

def bootstrap_hazard_rates(market_spreads_bps, recovery, r, notional=10_000_000, frequency=4):
    # market_spreads_bps: dict {maturity: spread_bps}, e.g. {1: 50, 3: 80, ...}
    # Returns DataFrame with tenor, market_spread, hazard_rate, model_spread, survival_prob
    tenors = sorted(market_spreads_bps.keys())
    knots = np.array([0.0] + tenors)
    hazard_rates = []
    results = []

    for idx, tenor in enumerate(tenors):
        target_spread = market_spreads_bps[tenor]

        def objective(lam_k):
            trial_rates = np.array(hazard_rates + [lam_k])
            trial_knots = knots[: len(trial_rates) + 1]
            fs = _fair_spread_pw(notional, tenor, trial_rates, trial_knots, r, recovery, frequency)
            return fs - target_spread

        # Find lam_k via Brent's method
        lam_k = brentq(objective, 1e-6, 2.0, xtol=1e-10)
        hazard_rates.append(lam_k)

        # Check model spread matches market
        trial_rates = np.array(hazard_rates)
        trial_knots = knots[: len(trial_rates) + 1]
        model_spread = _fair_spread_pw(notional, tenor, trial_rates, trial_knots, r, recovery, frequency)
        q_T = survival_probability_piecewise(trial_rates, trial_knots, tenor)

        results.append({
            "tenor": tenor,
            "market_spread_bps": target_spread,
            "hazard_rate": lam_k,
            "model_spread_bps": model_spread,
            "survival_prob": q_T,
        })

    return pd.DataFrame(results)


# -- Survival curve from bootstrap --

def bootstrap_survival_curve(hazard_rates, knots, max_time=None, n_points=300):
    # Fine-grained survival curve from bootstrapped rates
    if max_time is None:
        max_time = knots[-1]
    times = np.linspace(0, max_time, n_points)
    q = np.array([
        survival_probability_piecewise(hazard_rates, knots, t) for t in times
    ])
    return pd.DataFrame({"time": times, "survival_prob": q, "default_prob": 1.0 - q})
