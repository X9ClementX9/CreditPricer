# CDS pricing engine
# Reduced-form model: constant or piecewise hazard rate,
# flat risk-free rate, fixed recovery.

from __future__ import annotations
import numpy as np
from typing import Optional

from cds.utils import (
    payment_schedule,
    accrual_fraction,
    discount_factors,
    validate_positive,
    validate_rate,
    validate_nonnegative,
)


# -- Survival / default probs (constant hazard) --

def survival_probability(lam, t):
    # Q(t) = exp(-lam * t)
    return np.exp(-lam * t)


def default_probability(lam, t):
    # PD(t) = 1 - Q(t)
    return 1.0 - survival_probability(lam, t)


# -- Survival / default probs (piecewise-constant hazard) --

def survival_probability_piecewise(hazard_rates, knots, t):
    # hazard_rates[i] applies on (knots[i], knots[i+1]]
    # knots must start with 0
    scalar = np.isscalar(t)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    result = np.ones_like(t_arr)

    for idx, ti in enumerate(t_arr):
        integral = 0.0
        for k in range(len(hazard_rates)):
            t_start = knots[k]
            t_end = knots[k + 1]
            if ti <= t_start:
                break
            dt = min(ti, t_end) - t_start
            integral += hazard_rates[k] * dt
        result[idx] = np.exp(-integral)

    return float(result[0]) if scalar else result


# -- Premium leg PV --

def premium_leg_pv(spread_bps, notional, maturity, lam, r, frequency=4, include_accrual=True):
    # PV of the spread payments by the protection buyer
    spread = spread_bps / 10_000.0
    dt = accrual_fraction(frequency)
    dates = payment_schedule(maturity, frequency)
    df = discount_factors(r, dates)
    q = survival_probability(lam, dates)

    # Regular payments
    pv = notional * spread * dt * np.sum(df * q)

    # Accrual on default (half-period approx)
    if include_accrual:
        q_prev = np.concatenate([[1.0], q[:-1]])
        delta_pd = q_prev - q
        pv += notional * spread * (dt / 2.0) * np.sum(df * delta_pd)

    return pv


# -- Protection leg PV --

def protection_leg_pv(notional, maturity, lam, r, recovery, frequency=4, integration_steps=200):
    # PV of (1-R)*N paid at default, computed by trapezoidal integration
    lgd = 1.0 - recovery
    times = np.linspace(0, maturity, integration_steps + 1)
    q = survival_probability(lam, times)
    df = discount_factors(r, times)
    integrand = lam * q * df
    pv = lgd * notional * np.trapezoid(integrand, times)
    return pv


# -- Risky PV01 --

def risky_pv01(notional, maturity, lam, r, frequency=4, include_accrual=True):
    # PV of 1 bp of spread (risky annuity)
    return premium_leg_pv(
        spread_bps=1.0, notional=notional, maturity=maturity,
        lam=lam, r=r, frequency=frequency, include_accrual=include_accrual,
    )


# -- Fair spread --

def fair_spread(notional, maturity, lam, r, recovery, frequency=4, include_accrual=True):
    # Par spread (in bps) that makes PV = 0 at inception
    prot = protection_leg_pv(notional, maturity, lam, r, recovery, frequency)
    rpv01 = risky_pv01(notional, maturity, lam, r, frequency, include_accrual)
    if rpv01 == 0:
        return 0.0
    return prot / rpv01


# -- Mark-to-market --

def cds_mtm(contractual_spread_bps, market_spread_bps, notional, maturity, lam, r, recovery, frequency=4, include_accrual=True):
    # MTM from protection buyer side: prot - prem(contractual)
    prot = protection_leg_pv(notional, maturity, lam, r, recovery, frequency)
    prem = premium_leg_pv(contractual_spread_bps, notional, maturity, lam, r, frequency, include_accrual)
    return prot - prem


# -- CS01 --

def cs01(notional, maturity, lam, r, recovery, spread_bps, frequency=4, include_accrual=True, bump=1.0):
    # Change in PV for a 1bp spread bump (approx = -risky_pv01)
    rpv01 = risky_pv01(notional, maturity, lam, r, frequency, include_accrual)
    return -rpv01


# -- Full pricing report --

def pricing_report(notional, maturity, lam, r, recovery, spread_bps,
                   market_spread_bps=None, frequency=4, include_accrual=True):
    # Compute all key metrics, return as dict
    dates = payment_schedule(maturity, frequency)
    q_final = float(survival_probability(lam, maturity))
    pd_final = 1.0 - q_final

    prem = premium_leg_pv(spread_bps, notional, maturity, lam, r, frequency, include_accrual)
    prot = protection_leg_pv(notional, maturity, lam, r, recovery, frequency)
    fs = fair_spread(notional, maturity, lam, r, recovery, frequency, include_accrual)
    rpv01 = risky_pv01(notional, maturity, lam, r, frequency, include_accrual)

    mkt_spread = market_spread_bps if market_spread_bps is not None else spread_bps
    mtm = cds_mtm(spread_bps, mkt_spread, notional, maturity, lam, r, recovery, frequency, include_accrual)
    credit_cs01 = cs01(notional, maturity, lam, r, recovery, spread_bps, frequency, include_accrual)

    # Recovery sens: bump R +1pp
    prot_up = protection_leg_pv(notional, maturity, lam, r, recovery + 0.01, frequency)
    recovery_sens = prot_up - prot

    # Rate sens: bump r +1bp
    prem_r_up = premium_leg_pv(spread_bps, notional, maturity, lam, r + 0.0001, frequency, include_accrual)
    prot_r_up = protection_leg_pv(notional, maturity, lam, r + 0.0001, recovery, frequency)
    rate_sens = (prot_r_up - prem_r_up) - (prot - prem)

    return {
        "survival_prob_at_maturity": q_final,
        "default_prob_at_maturity": pd_final,
        "premium_leg_pv": prem,
        "protection_leg_pv": prot,
        "fair_spread_bps": fs,
        "cds_pv": mtm,
        "risky_pv01": rpv01,
        "cs01": credit_cs01,
        "recovery_sensitivity": recovery_sens,
        "rate_sensitivity": rate_sens,
    }
