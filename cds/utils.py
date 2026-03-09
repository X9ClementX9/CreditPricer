# Shared utilities: schedules, discount factors, validation

import numpy as np


# -- Payment schedule --

def payment_schedule(maturity, frequency=4):
    # Returns array of payment dates as year fractions
    dt = 1.0 / frequency
    dates = np.arange(dt, maturity + 1e-9, dt)
    if dates[-1] > maturity + 1e-9:
        dates = dates[:-1]
    return dates


def accrual_fraction(frequency=4):
    # Year fraction between two premium dates
    return 1.0 / frequency


# -- Discount factors --

def discount_factor(r, t):
    # Single discount factor: exp(-r * t)
    return np.exp(-r * t)


def discount_factors(r, times):
    # Vectorised discount factors
    return np.exp(-r * times)


# -- Input validation --

def validate_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive, got {value}")


def validate_rate(value, name):
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def validate_nonnegative(value, name):
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
