# CDS Pricer — run with: streamlit run app.py

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from cds.pricing import (
    pricing_report,
    survival_probability,
    default_probability,
    fair_spread,
)
from cds.analytics import probability_curves
from cds.bootstrap import bootstrap_hazard_rates, bootstrap_survival_curve
from cds.plots import (
    plot_survival_default,
    plot_hazard_term_structure,
    plot_bootstrap_survival,
    plot_spread_comparison,
)

# Page setup
st.set_page_config(
    page_title="CDS Pricer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Sidebar inputs --

with st.sidebar:
    st.markdown("## Parameters")

    st.markdown("### Contract")
    notional = st.number_input(
        "Notional", value=1_000, step=1_000_000,
        format="%d", help="CDS notional amount",
    )
    maturity = st.number_input(
        "Maturity (years)", value=5.0, step=0.5,
        min_value=0.5, max_value=30.0, format="%.1f",
    )
    frequency = st.selectbox(
        "Payment frequency", [1, 2, 4, 12],
        index=2, format_func=lambda x: {
            1: "Annual", 2: "Semi-annual", 4: "Quarterly", 12: "Monthly"
        }[x],
    )

    st.markdown("### Market")
    spread_bps = st.number_input(
        "Contractual Spread (bps)", value=100.0, step=5.0,
        help="CDS spread agreed at contract inception",
    )
    recovery = st.slider(
        "Recovery Rate", min_value=0.0, max_value=0.95,
        value=0.40, step=0.01,
    )
    risk_free = st.slider(
        "Risk-Free Rate", min_value=0.0, max_value=0.15,
        value=0.03, step=0.005,
        help="Continuously compounded flat rate",
    )

    st.markdown("### Credit")
    mode = st.radio(
        "Mode",
        ["Use hazard rate",
         "Use market spread"],
        help="Choose whether to specify the hazard rate directly "
             "or derive it from the CDS spread.",
    )

    if mode.startswith("Use hazard"):
        hazard_rate = st.slider(
            "Hazard Rate λ", min_value=0.001, max_value=0.50,
            value=0.02, step=0.001, format="%.4f",
        )
    else:
        # Solve for hazard rate that matches the spread
        from scipy.optimize import brentq

        def _spread_diff(lam: float) -> float:
            return fair_spread(
                notional, maturity, lam, risk_free, recovery, frequency
            ) - spread_bps

        try:
            hazard_rate = brentq(_spread_diff, 1e-6, 2.0, xtol=1e-10)
            st.info(f"Implied λ = {hazard_rate:.6f}")
        except Exception:
            hazard_rate = 0.02
            st.warning("Could not imply hazard rate — using default 0.02")

    enable_mtm = False
    market_spread_bps = None
    mtm_hazard_rate = None

    if not mode.startswith("Use hazard"):
        st.markdown("### Mark to Market")
        enable_mtm = st.checkbox("Enable MTM calculation", value=False)
        if enable_mtm:
            market_spread_bps = st.number_input(
                "Market Spread (bps)", value=spread_bps + 20, step=5.0,
                help="Current market spread (different from contractual)",
            )
            # Solve for hazard rate implied by market spread
            from scipy.optimize import brentq as _brentq

            def _mtm_spread_diff(lam: float) -> float:
                return fair_spread(
                    notional, maturity, lam, risk_free, recovery, frequency
                ) - market_spread_bps

            try:
                mtm_hazard_rate = _brentq(_mtm_spread_diff, 1e-6, 2.0, xtol=1e-10)
                st.info(f"MTM implied λ = {mtm_hazard_rate:.6f}")
            except Exception:
                mtm_hazard_rate = hazard_rate
                st.warning("Could not imply MTM hazard rate — using current λ")




# -- Main content --

st.title("CDS Single-Name Pricer")
st.caption("A pedagogical credit default swap pricing tool")

tab1, tab2 = st.tabs(["Pricer", "Bootstrap"])


# -- Tab 1: Pricer --

with tab1:
    st.markdown("### Pricing Results")

    # Pick hazard rate: MTM one if enabled, base otherwise
    pricing_lam = mtm_hazard_rate if (enable_mtm and mtm_hazard_rate is not None) else hazard_rate

    report = pricing_report(
        notional=notional,
        maturity=maturity,
        lam=pricing_lam,
        r=risk_free,
        recovery=recovery,
        spread_bps=spread_bps,
        market_spread_bps=market_spread_bps,
        frequency=frequency,
        include_accrual=True,
    )

    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Fair Spread", f"{report['fair_spread_bps']:.2f} bps")
    col2.metric("CDS PV (MTM)", f"${report['cds_pv']:,.2f}")
    col3.metric("Risky PV01", f"${report['risky_pv01']:,.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Premium Leg PV", f"${report['premium_leg_pv']:,.2f}")
    col6.metric("Protection Leg PV", f"${report['protection_leg_pv']:,.2f}")
    col7.metric("Survival Q(T)", f"{report['survival_prob_at_maturity']:.2%}")
    col8.metric("Default PD(T)", f"{report['default_prob_at_maturity']:.2%}")



    # Probability chart
    st.markdown("---")
    st.markdown("#### Survival & Default Probabilities")
    prob_df = probability_curves(pricing_lam, maturity)
    st.plotly_chart(plot_survival_default(prob_df), use_container_width=True)


# -- Tab 2: Bootstrap --

with tab2:
    st.markdown("### Hazard Curve Bootstrap")
    st.markdown(
        "We calibrate a piecewise-constant hazard rate curve from market CDS spreads at standard maturities using a sequential bootstrap."
    )

    st.markdown("---")
    st.markdown("#### Market CDS Spreads")

    in_cols = st.columns(5)
    s1y = in_cols[0].number_input("1Y (bps)", value=50.0, step=5.0)
    s3y = in_cols[1].number_input("3Y (bps)", value=80.0, step=5.0)
    s5y = in_cols[2].number_input("5Y (bps)", value=120.0, step=5.0)
    s7y = in_cols[3].number_input("7Y (bps)", value=150.0, step=5.0)
    s10y = in_cols[4].number_input("10Y (bps)", value=180.0, step=5.0)

    st.markdown("#### Bootstrap Parameters")
    bc1, bc2 = st.columns(2)
    bs_recovery = bc1.slider("Recovery Rate (Bootstrap)", 0.0, 0.95, 0.40, 0.01, key="bs_rec")
    bs_rate = bc2.slider("Risk-Free Rate (Bootstrap)", 0.0, 0.15, 0.03, 0.005, key="bs_rate")

    market_spreads = {1: s1y, 3: s3y, 5: s5y, 7: s7y, 10: s10y}

    if st.button("Run Bootstrap", type="primary"):
        with st.spinner("Bootstrapping hazard rates …"):
            try:
                bs_df = bootstrap_hazard_rates(
                    market_spreads, bs_recovery, bs_rate, notional, frequency
                )

                st.success("Bootstrap completed successfully!")

                # Results table
                st.markdown("#### Results")
                display_df = pd.DataFrame([
                    {
                        "Metric": f"{row['tenor']:.0f}Y",
                        "Market Spread (bps)": f"{row['market_spread_bps']:.2f}",
                        "Hazard Rate λ": f"{row['hazard_rate']:.2f}",
                        "Model Spread (bps)": f"{row['model_spread_bps']:.2f}",
                        "Survival Q(T)": f"{row['survival_prob']:.2f}",
                    }
                    for _, row in bs_df.iterrows()
                ])
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Default probs per interval
                st.markdown("#### Default Probabilities per Interval")
                raw_hr = bs_df["hazard_rate"].values.astype(float)
                knots = np.array([0.0] + list(bs_df["tenor"].values))
                q_values = [1.0] + [float(x) for x in bs_df["survival_prob"].values]
                interval_df = pd.DataFrame([
                    {
                        "Interval": f"{knots[i]:.0f}Y – {knots[i+1]:.0f}Y",
                        "λ": f"{raw_hr[i]:.2f}",
                        "Q(start)": f"{q_values[i]:.2f}",
                        "Q(end)": f"{q_values[i+1]:.2f}",
                        "Default Prob": f"{q_values[i] - q_values[i+1]:.2f}",
                    }
                    for i in range(len(raw_hr))
                ])
                st.dataframe(interval_df, use_container_width=True, hide_index=True)

                # Charts
                st.markdown("---")
                ch1, ch2 = st.columns(2)
                with ch1:
                    st.plotly_chart(
                        plot_hazard_term_structure(bs_df),
                        use_container_width=True,
                    )
                with ch2:
                    raw_hr = bs_df["hazard_rate"].values.astype(float)
                    curve_df = bootstrap_survival_curve(
                        raw_hr, knots, max_time=float(knots[-1]),
                    )
                    st.plotly_chart(
                        plot_bootstrap_survival(curve_df),
                        use_container_width=True,
                    )

            except Exception as exc:
                st.error(f"Bootstrap failed: {exc}")
