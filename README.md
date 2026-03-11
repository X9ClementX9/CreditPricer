# CDS Pricer

A pedagogical credit default swap pricing tool built with Python and Streamlit.

## Pricer

Single-name CDS pricing using a reduced-form model with constant hazard rate. Compute **fair spread**, **PV (mark-to-market)**, **risky PV01**, **premium and protection leg PVs**, and **survival/default probabilities**. The hazard rate can be set directly or implied from a market spread.

## Bootstrap

Calibrate a **piecewise-constant hazard rate curve** from market CDS spreads at standard maturities (1Y, 3Y, 5Y, 7Y, 10Y) using a sequential bootstrap. Visualise the hazard term structure and the bootstrapped survival curve.

## Basket CDS

Price **Nth-to-default basket CDS** using a **one-factor Gaussian copula** Monte Carlo simulation. Define a basket of 2 to 6 names with individual hazard rates and recoveries, set a common correlation, and compute fair spreads and default probabilities for each tranche (1st-to-default, 2nd-to-default, etc.).