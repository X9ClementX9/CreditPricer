# CDS Pricer — run with: streamlit run app.py

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from cds.pricing import (
    pricing_report,
    fair_spread,
)
from cds.analytics import probability_curves
from cds.bootstrap import bootstrap_hazard_rates, bootstrap_survival_curve
from cds.plots import (
    plot_survival_default,
    plot_hazard_term_structure,
    plot_bootstrap_survival,
    plot_basket_pd,
    plot_basket_spread,
    plot_basket_default_count,
)
from cds.basket import (
    basket_pricing,
    hazard_from_spread,
    spread_from_hazard,
    default_count_distribution,
    expected_losses,
)

# Page setup
st.set_page_config(
    page_title="Credit Tool",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from scipy.optimize import brentq

# -- Main content --

st.title("Credit Pedagogical Tool")
st.caption("A pedagogical credit default swap pricing tool")

tab1, tab2, tab3 = st.tabs(["Pricer", "Bootstrap", "Basket CDS"])


# -- Tab 1: Pricer --

with tab1:
    st.markdown("### CDS Pricing")
    st.markdown(
        "Price a single-name credit default swap and visualise survival and default "
        "probabilities under flat hazard rate or market-implied calibration."
    )

    st.markdown("---")
    st.markdown("#### Contract & Market")

    # Reset counter for pricer widgets
    if "pr_reset_counter" not in st.session_state:
        st.session_state.pr_reset_counter = 0
    pr_rc = st.session_state.pr_reset_counter

    if st.button("↻ Reset to Defaults", key="pr_reset"):
        st.session_state.pr_reset_counter = pr_rc + 1
        st.rerun()

    p_c1, p_c2, p_c3 = st.columns(3)
    notional = p_c1.number_input(
        "Notional", value=1_000, step=1_000_000,
        format="%d", help="CDS notional amount", key=f"pr_notional_{pr_rc}",
    )
    maturity = p_c2.number_input(
        "Maturity (years)", value=5.0, step=0.5,
        min_value=0.5, max_value=30.0, format="%.1f", key=f"pr_maturity_{pr_rc}",
    )
    frequency = p_c3.selectbox(
        "Payment frequency", [1, 2, 4, 12],
        index=2, format_func=lambda x: {
            1: "Annual", 2: "Semi-annual", 4: "Quarterly", 12: "Monthly"
        }[x], key=f"pr_freq_{pr_rc}",
    )

    p_c4, p_c5, p_c6 = st.columns(3)
    spread_bps = p_c4.selectbox(
        "Contractual Spread (bps)", [25, 100, 500, 1000],
        index=1,
        format_func=lambda x: f"{x} bps", key=f"pr_spread_{pr_rc}",
    )
    recovery = p_c5.slider(
        "Recovery Rate", min_value=0.0, max_value=0.95,
        value=0.40, step=0.01, key=f"pr_rec_{pr_rc}",
    )
    risk_free = p_c6.slider(
        "Risk-Free Rate", min_value=0.0, max_value=0.15,
        value=0.03, step=0.005,
        help="Continuously compounded flat rate", key=f"pr_rf_{pr_rc}",
    )

    st.markdown("#### Credit")

    p_c7, p_c8 = st.columns(2)
    mode = p_c7.radio(
        "Mode",
        ["Use hazard rate", "Use market spread"],
        help="Choose whether to specify the hazard rate directly "
             "or derive it from the CDS spread.",
        horizontal=True, key=f"pr_mode_{pr_rc}",
    )

    if mode.startswith("Use hazard"):
        hazard_rate = p_c8.slider(
            "Hazard Rate λ", min_value=0.001, max_value=0.50,
            value=0.02, step=0.001, format="%.4f", key=f"pr_lam_{pr_rc}",
        )
    else:
        def _spread_diff(lam: float) -> float:
            return fair_spread(
                notional, maturity, lam, risk_free, recovery, frequency
            ) - spread_bps

        try:
            hazard_rate = brentq(_spread_diff, 1e-6, 2.0, xtol=1e-10)
            p_c8.info(f"Implied λ = {hazard_rate:.6f}")
        except Exception:
            hazard_rate = 0.02
            p_c8.warning("Could not imply hazard rate — using default 0.02")

    enable_mtm = False
    market_spread_bps = None
    mtm_hazard_rate = None

    if not mode.startswith("Use hazard"):
        st.markdown("#### Mark to Market")
        mtm_c1, mtm_c2 = st.columns(2)
        enable_mtm = mtm_c1.checkbox("Enable MTM calculation", value=False)
        if enable_mtm:
            market_spread_bps = mtm_c2.number_input(
                "Market Spread (bps)", value=float(spread_bps + 20), step=5.0,
                help="Current market spread (different from contractual)",
            )

            def _mtm_spread_diff(lam: float) -> float:
                return fair_spread(
                    notional, maturity, lam, risk_free, recovery, frequency
                ) - market_spread_bps

            try:
                mtm_hazard_rate = brentq(_mtm_spread_diff, 1e-6, 2.0, xtol=1e-10)
                st.info(f"MTM implied λ = {mtm_hazard_rate:.6f}")
            except Exception:
                mtm_hazard_rate = hazard_rate
                st.warning("Could not imply MTM hazard rate — using current λ")

    # -- Pricing Results --
    st.markdown("---")
    st.markdown("#### Pricing Results")

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
    st.plotly_chart(plot_survival_default(prob_df), width="stretch")


# -- Tab 2: Bootstrap --

with tab2:
    st.markdown("### Hazard Curve Bootstrap")
    st.markdown(
        "We calibrate a piecewise-constant hazard rate curve from market CDS spreads at standard maturities using a sequential bootstrap."
    )

    st.markdown("---")
    st.markdown("#### Market CDS Spreads")

    # Reset counter for bootstrap widgets
    if "bs_reset_counter" not in st.session_state:
        st.session_state.bs_reset_counter = 0
    bs_rc = st.session_state.bs_reset_counter

    if st.button("↻ Reset to Defaults", key="bs_reset"):
        st.session_state.bs_reset_counter = bs_rc + 1
        st.rerun()

    in_cols = st.columns(5)
    s1y = in_cols[0].number_input("1Y (bps)", value=50.0, step=5.0, key=f"bs_1y_{bs_rc}")
    s3y = in_cols[1].number_input("3Y (bps)", value=80.0, step=5.0, key=f"bs_3y_{bs_rc}")
    s5y = in_cols[2].number_input("5Y (bps)", value=120.0, step=5.0, key=f"bs_5y_{bs_rc}")
    s7y = in_cols[3].number_input("7Y (bps)", value=150.0, step=5.0, key=f"bs_7y_{bs_rc}")
    s10y = in_cols[4].number_input("10Y (bps)", value=180.0, step=5.0, key=f"bs_10y_{bs_rc}")

    st.markdown("#### Bootstrap Parameters")
    bc1, bc2, bc3 = st.columns(3)
    bs_recovery = bc1.slider("Recovery Rate", 0.0, 0.95, 0.40, 0.01, key=f"bs_rec_{bs_rc}")
    bs_rate = bc2.slider("Risk-Free Rate", 0.0, 0.15, 0.03, 0.005, key=f"bs_rate_{bs_rc}")
    bs_frequency = bc3.selectbox(
        "Payment Frequency", [1, 2, 4, 12],
        index=2, format_func=lambda x: {
            1: "Annual", 2: "Semi-annual", 4: "Quarterly", 12: "Monthly"
        }[x],
        key=f"bs_freq_{bs_rc}",
    )

    market_spreads = {1: s1y, 3: s3y, 5: s5y, 7: s7y, 10: s10y}
    bs_notional = 10_000_000  # Cancels out in spread calculation

    @st.cache_data(show_spinner="Bootstrapping hazard rates…")
    def _cached_bootstrap(spreads_tuple, rec, rate, notional, freq):
        return bootstrap_hazard_rates(
            dict(spreads_tuple), rec, rate, notional, freq
        )

    try:
        bs_df = _cached_bootstrap(
            tuple(market_spreads.items()), bs_recovery, bs_rate, bs_notional, bs_frequency
        )

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
        st.dataframe(display_df, width="stretch", hide_index=True)

        raw_hr = bs_df["hazard_rate"].values.astype(float)
        knots = np.array([0.0] + list(bs_df["tenor"].values))

        # Charts
        st.markdown("---")
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(
                plot_hazard_term_structure(bs_df),
                width="stretch",
            )
        with ch2:
            curve_df = bootstrap_survival_curve(
                raw_hr, knots, max_time=float(knots[-1]),
            )
            st.plotly_chart(
                plot_bootstrap_survival(curve_df),
                width="stretch",
            )

    except Exception as exc:
        st.error(f"Bootstrap failed: {exc}")

# -- Tab 3: Basket CDS --

with tab3:
    st.markdown("### Basket CDS - Nth to Default Pricing")
    st.markdown(
        "Price basket credit default swaps using a **one-factor Gaussian copula** "
        "Monte Carlo simulation. Choose the number of names, set individual "
        "credit parameters, and explore how correlation affects Nth to default spreads."
    )

    st.markdown("---")

    # -- Basket setup --
    st.markdown("#### Basket Setup")

    # Reset counter: incrementing changes all widget keys → fresh defaults
    if "bk_reset_counter" not in st.session_state:
        st.session_state.bk_reset_counter = 0
    rc = st.session_state.bk_reset_counter

    if st.button("↻ Reset to Defaults", key="bk_reset"):
        st.session_state.bk_reset_counter = rc + 1
        st.rerun()

    bk_c1, bk_c2, bk_c3, bk_c4 = st.columns(4)
    n_names = bk_c1.slider("Number of names", 2, 6, 3, key=f"bk_n_{rc}")
    bk_rate = bk_c2.slider("Risk-Free Rate", 0.0, 0.15, 0.03, 0.005, key=f"bk_rate_{rc}")
    bk_corr = bk_c3.slider("Correlation ρ", 0.0, 0.99, 0.30, 0.01, key=f"bk_corr_{rc}")
    bk_maturity = bk_c4.number_input(
        "Maturity (years)", value=5.0, step=0.5,
        min_value=0.5, max_value=30.0, format="%.1f", key=f"bk_mat_{rc}",
    )

    mc_c1, mc_c2, mc_c3 = st.columns(3)
    bk_n_sims = mc_c1.select_slider(
        "MC Simulations",
        options=[50_000, 100_000, 500_000, 1_000_000],
        value=100_000,
        format_func=lambda x: f"{x:,}",
        key=f"bk_sims_{rc}",
    )
    bk_freq = mc_c2.selectbox(
        "Payment Frequency", [1, 2, 4],
        index=2, format_func=lambda x: {
            1: "Annual", 2: "Semi-annual", 4: "Quarterly"
        }[x],
        key=f"bk_freq_{rc}",
    )
    bk_input_mode = mc_c3.radio(
        "Input mode", ["Hazard Rate", "Spread (bps)"],
        key=f"bk_input_mode_{rc}", horizontal=True,
    )

    # -- Per-name inputs --
    st.markdown("#### Credit Parameters per Name")

    name_cols = st.columns(n_names)
    bk_hazard_rates = []
    bk_recoveries = []
    bk_spreads_display = []

    default_lambdas = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
    default_recoveries = [0.40, 0.40, 0.40, 0.35, 0.35, 0.30]

    for i in range(n_names):
        with name_cols[i]:
            st.markdown(f"**Name {i + 1}**")

            rec_i = st.slider(
                "Recovery", 0.0, 0.95, default_recoveries[i], 0.01,
                key=f"bk_rec_{i}_{rc}",
            )

            if bk_input_mode == "Hazard Rate":
                lam_i = st.number_input(
                    "λ", value=default_lambdas[i],
                    min_value=0.001, max_value=0.50,
                    step=0.005, format="%.4f", key=f"bk_lam_{i}_{rc}",
                )
                spd_i = spread_from_hazard(lam_i, rec_i)
                st.caption(f"≈ {spd_i:.0f} bps")
            else:
                default_spd = spread_from_hazard(default_lambdas[i], default_recoveries[i])
                spd_i = st.number_input(
                    "Spread (bps)", value=round(default_spd, 0),
                    min_value=1.0, max_value=5000.0,
                    step=5.0, format="%.0f", key=f"bk_spd_{i}_{rc}",
                )
                lam_i = hazard_from_spread(spd_i, rec_i)
                st.caption(f"λ ≈ {lam_i:.4f}")

            bk_hazard_rates.append(lam_i)
            bk_recoveries.append(rec_i)
            bk_spreads_display.append(spread_from_hazard(lam_i, rec_i))

    # -- Results (auto-update on parameter change) --
    st.markdown("---")

    @st.cache_data(show_spinner="Running Monte Carlo simulations…")
    def _cached_basket_pricing(hr_tuple, rec_tuple, corr, mat, rf, n_sims, freq):
        return basket_pricing(
            hazard_rates=list(hr_tuple),
            recoveries=list(rec_tuple),
            correlation=corr,
            maturity=mat,
            risk_free=rf,
            n_simulations=n_sims,
            frequency=freq,
        )

    try:
        # Basket pricing (cached)
        pricing_out = _cached_basket_pricing(
            hr_tuple=tuple(bk_hazard_rates),
            rec_tuple=tuple(bk_recoveries),
            corr=bk_corr,
            mat=bk_maturity,
            rf=bk_rate,
            n_sims=bk_n_sims,
            freq=bk_freq,
        )
        results = pricing_out["tranches"]
        tau = pricing_out["tau"]
        sorted_idx = pricing_out["sorted_indices"]

        # Compute expected losses to merge into results table
        el_data = expected_losses(tau, sorted_idx, bk_recoveries, bk_maturity)
        el_map = {e["label"]: e for e in el_data}

        # Compute default count distribution
        dist = default_count_distribution(tau, bk_maturity)

        st.markdown("#### Nth to Default Results")
        res_df = pd.DataFrame([
            {
                "Product": r["label"],
                "P(default < T)": f"{r['prob_default']:.2%}",
                "Avg LGD": f"{el_map[r['label']]['avg_lgd']:.2%}",
                "Expected Loss": f"{el_map[r['label']]['expected_loss']:.2%}",
                "Fair Spread (bps)": f"{r['fair_spread_bps']:.2f}",
                "Premium Leg PV": f"${r['premium_leg_pv']:,.2f}",
                "Protection Leg PV": f"${r['protection_leg_pv']:,.2f}",
            }
            for r in results
        ])
        st.dataframe(res_df, width="stretch", hide_index=True)

        # -- Three charts side by side --
        ch1, ch2, ch3 = st.columns(3)
        short_labels = [f"{r['rank']}{['th','st','nd','rd'][r['rank'] if r['rank'] < 4 else 0]}" for r in results]
        tranche_pds = [r["prob_default"] for r in results]
        tranche_spreads = [r["fair_spread_bps"] for r in results]

        with ch1:
            st.plotly_chart(
                plot_basket_pd(short_labels, tranche_pds),
                width="stretch",
            )
        with ch2:
            st.plotly_chart(
                plot_basket_spread(short_labels, tranche_spreads),
                width="stretch",
            )
        with ch3:
            dist_ns = [d["n_defaults"] for d in dist]
            dist_ps = [d["probability"] for d in dist]
            st.plotly_chart(
                plot_basket_default_count(dist_ns, dist_ps),
                width="stretch",
            )

    except Exception as exc:
        st.error(f"Basket pricing failed: {exc}")
