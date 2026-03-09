# Sensitivity analysis helpers
# Sweep one or two params and return DataFrames for charts

from __future__ import annotations
import numpy as np
import pandas as pd

from cds.pricing import (
    fair_spread,
    cds_mtm,
    risky_pv01,
    survival_probability,
    default_probability,
    protection_leg_pv,
    premium_leg_pv,
)


# -- 1D sweeps --

def sweep_hazard_rate(notional, maturity, r, recovery, spread_bps, frequency=4,
                      lam_range=(0.001, 0.30), n_points=100):
    lams = np.linspace(*lam_range, n_points)
    rows = []
    for lam in lams:
        fs = fair_spread(notional, maturity, lam, r, recovery, frequency)
        pv = cds_mtm(spread_bps, spread_bps, notional, maturity, lam, r, recovery, frequency)
        rpv = risky_pv01(notional, maturity, lam, r, frequency)
        rows.append({"hazard_rate": lam, "fair_spread_bps": fs, "cds_pv": pv, "risky_pv01": rpv})
    return pd.DataFrame(rows)


def sweep_recovery(notional, maturity, lam, r, spread_bps, frequency=4,
                    rec_range=(0.0, 0.80), n_points=80):
    recs = np.linspace(*rec_range, n_points)
    rows = []
    for rec in recs:
        fs = fair_spread(notional, maturity, lam, r, rec, frequency)
        pv = cds_mtm(spread_bps, spread_bps, notional, maturity, lam, r, rec, frequency)
        rpv = risky_pv01(notional, maturity, lam, r, frequency)
        rows.append({"recovery_rate": rec, "fair_spread_bps": fs, "cds_pv": pv, "risky_pv01": rpv})
    return pd.DataFrame(rows)


def sweep_risk_free_rate(notional, maturity, lam, recovery, spread_bps, frequency=4,
                         r_range=(0.0, 0.10), n_points=80):
    rates = np.linspace(*r_range, n_points)
    rows = []
    for r in rates:
        fs = fair_spread(notional, maturity, lam, r, recovery, frequency)
        pv = cds_mtm(spread_bps, spread_bps, notional, maturity, lam, r, recovery, frequency)
        rpv = risky_pv01(notional, maturity, lam, r, frequency)
        rows.append({"risk_free_rate": r, "fair_spread_bps": fs, "cds_pv": pv, "risky_pv01": rpv})
    return pd.DataFrame(rows)


def sweep_maturity(notional, lam, r, recovery, spread_bps, frequency=4,
                   mat_range=(0.5, 15.0), n_points=80):
    mats = np.linspace(*mat_range, n_points)
    rows = []
    for mat in mats:
        fs = fair_spread(notional, mat, lam, r, recovery, frequency)
        pv = cds_mtm(spread_bps, spread_bps, notional, mat, lam, r, recovery, frequency)
        rpv = risky_pv01(notional, mat, lam, r, frequency)
        rows.append({"maturity": mat, "fair_spread_bps": fs, "cds_pv": pv, "risky_pv01": rpv})
    return pd.DataFrame(rows)


# -- Probability curves --

def probability_curves(lam, maturity, n_points=200):
    # Survival and default prob over time (constant hazard)
    times = np.linspace(0, maturity, n_points)
    q = survival_probability(lam, times)
    pd_arr = default_probability(lam, times)
    return pd.DataFrame({"time": times, "survival_prob": q, "default_prob": pd_arr})


# -- 2D heatmaps --

def heatmap_fair_spread(notional, maturity, r, frequency=4,
                        lam_range=(0.005, 0.20), rec_range=(0.10, 0.70),
                        n_lam=25, n_rec=25):
    lams = np.linspace(*lam_range, n_lam)
    recs = np.linspace(*rec_range, n_rec)
    Z = np.zeros((n_rec, n_lam))
    for i, rec in enumerate(recs):
        for j, lam in enumerate(lams):
            Z[i, j] = fair_spread(notional, maturity, lam, r, rec, frequency)
    return lams, recs, Z


def heatmap_cds_pv(notional, maturity, recovery, spread_bps, frequency=4,
                   lam_range=(0.005, 0.20), r_range=(0.0, 0.08),
                   n_lam=25, n_r=25):
    lams = np.linspace(*lam_range, n_lam)
    rates = np.linspace(*r_range, n_r)
    Z = np.zeros((n_r, n_lam))
    for i, r_val in enumerate(rates):
        for j, lam in enumerate(lams):
            Z[i, j] = cds_mtm(spread_bps, spread_bps, notional, maturity, lam, r_val, recovery, frequency)
    return lams, rates, Z
