# Analytics helpers
# Probability curves for CDS pricing

from __future__ import annotations
import numpy as np
import pandas as pd

from cds.pricing import survival_probability, default_probability


# -- Probability curves --

def probability_curves(lam, maturity, n_points=200):
    # Survival and default prob over time (constant hazard)
    times = np.linspace(0, maturity, n_points)
    q = survival_probability(lam, times)
    pd_arr = default_probability(lam, times)
    return pd.DataFrame({"time": times, "survival_prob": q, "default_prob": pd_arr})
