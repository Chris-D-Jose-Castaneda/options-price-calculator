"""
Options Pricing App (Streamlit)
- Hyphenated modules supported: "Black-Scholes.py", "Monte-Carlo.py"
- BSM (with dividend yield q), Greeks, Implied Vol
- Monte-Carlo (antithetic + control variate) with SE and 95% CI
- Tabs: Overview / Sensitivities / Distribution / Heatmaps
- Heatmaps: vertical layout, diverging red↔white↔green (white only near 0), crisp labels
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import streamlit as st

# ======================= Dynamic import (supports hyphenated filenames) =======================
ROOT = Path(__file__).resolve().parent

def _import_by_filename(module_name: str, filename: str):
    file_path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

try:
    from black_scholes import black_scholes_price, greeks, implied_volatility, put_call_parity_gap  # type: ignore
except Exception:
    BS = _import_by_filename("Black_Scholes_mod", "Black-Scholes.py")
    black_scholes_price = BS.black_scholes_price
    greeks = BS.greeks
    implied_volatility = BS.implied_volatility
    put_call_parity_gap = BS.put_call_parity_gap

try:
    from monte_carlo import monte_carlo_option_price, sample_discounted_payoffs  # type: ignore
except Exception:
    MC = _import_by_filename("Monte_Carlo_mod", "Monte-Carlo.py")
    monte_carlo_option_price = MC.monte_carlo_option_price
    sample_discounted_payoffs = MC.sample_discounted_payoffs

# ======================= Page & light styling =======================
st.set_page_config(page_title="Options Pricing — BS & Monte Carlo", page_icon="📈", layout="wide")

st.markdown("""
<style>
:root{ --green:#16a34a; --red:#dc2626; --muted:#64748b; }
.block-container { max-width: 1200px; }
.card{ background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,.05); }
.card .label{ color:var(--muted); font-size:.9rem; margin-bottom:.25rem; }
.card .value{ font-size:1.8rem; font-weight:700; line-height:1.1; }
hr{ border:none; height:1px; background:#e5e7eb; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#0b1324", "axes.labelcolor": "#0b1324",
    "xtick.color": "#334155", "ytick.color": "#334155",
    "axes.grid": True, "grid.color": "#e2e8f0", "grid.alpha": 0.7,
    "font.size": 11
})

st.title("📈 Options Pricing — Black-Scholes–Merton vs Monte-Carlo (Advanced)")

# ======================= Sidebar =======================
st.sidebar.header("Parameters")
c1, c2 = st.sidebar.columns(2)
S = c1.number_input("Spot S", value=100.0, min_value=0.01, step=1.0)
K = c2.number_input("Strike K", value=100.0, min_value=0.01, step=1.0)

c3, c4 = st.sidebar.columns(2)
T = c3.number_input("Maturity T (years)", value=1.0, min_value=0.0001, step=0.05, format="%.4f")
r = c4.number_input("Risk-free r", value=0.05, step=0.005, format="%.4f")

c5, c6 = st.sidebar.columns(2)
sigma = c5.number_input("Vol σ", value=0.20, min_value=0.0001, step=0.01, format="%.4f")
q = c6.number_input("Dividend yield q", value=0.00, step=0.005, format="%.4f")

option_type = st.sidebar.selectbox("Option type", ["call", "put"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Monte-Carlo Settings")
n_sims = st.sidebar.slider("Simulations", 10_000, 1_000_000, 200_000, step=10_000)
antithetic = st.sidebar.checkbox("Antithetic variates", True)
control_var = st.sidebar.checkbox("Control variate", True)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
market_price = st.sidebar.number_input("Market price (for IV)", value=0.0, min_value=0.0, format="%.4f")

# ======================= Core calcs =======================
bs_price = black_scholes_price(S, K, T, r, sigma, option_type, q)
mc = monte_carlo_option_price(S, K, T, r, sigma, option_type, q, n_sims, antithetic, control_var, int(seed))
diff = mc["price"] - bs_price

# ======================= Tabs =======================
tab_overview, tab_sens, tab_dist, tab_heat = st.tabs(["Overview", "Sensitivities", "Distribution", "Heatmaps"])

# ---------- Overview ----------
with tab_overview:
    left, right = st.columns([1.2, 1])
    with left:
        a, b = st.columns(2)
        with a:
            st.markdown(f"""
                <div class="card">
                  <div class="label">Black-Scholes–Merton</div>
                  <div class="value">${bs_price:,.4f}</div>
                </div>
            """, unsafe_allow_html=True)
        with b:
            color = "var(--green)" if diff >= 0 else "var(--red)"
            st.markdown(f"""
                <div class="card">
                  <div class="label">Monte-Carlo (95% CI)</div>
                  <div class="value">${mc['price']:,.4f}</div>
                  <div style="color:{color}; font-weight:600; margin-top:.35rem;">Δ {diff:+.4f}</div>
                  <div style="color:#64748b; font-size:.9rem;">CI [{mc['ci_low']:.4f}, {mc['ci_high']:.4f}] • SE {mc['stderr']:.6f} • n={mc['n_sims']:,}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        if market_price > 0:
            iv = implied_volatility(market_price, S, K, T, r, option_type, q)
            if np.isfinite(iv):
                st.markdown(f"""
                    <div class="card">
                      <div class="label">Implied Volatility from ${market_price:.4f}</div>
                      <div class="value">{iv*100:.3f}%</div>
                      <div style="color:#64748b; font-size:.9rem;">BSM price at IV: ${black_scholes_price(S, K, T, r, iv, option_type, q):.4f}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No IV solution in [1e-6, 5]. Check inputs.", icon="ℹ️")

    with right:
        st.subheader("Greeks (BSM)")
        g = greeks(S, K, T, r, sigma, option_type, q)
        st.markdown(f"""
            <div class="card">
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:.25rem .75rem;">
                <div>Δ</div><div style="text-align:right;">{g['delta']:.6f}</div>
                <div>Γ</div><div style="text-align:right;">{g['gamma']:.6f}</div>
                <div>Vega</div><div style="text-align:right;">{g['vega']:.6f}</div>
                <div>Θ (annualized)</div><div style="text-align:right;">{g['theta']:.6f}</div>
                <div>ρ</div><div style="text-align:right;">{g['rho']:.6f}</div>
              </div>
            </div>
        """, unsafe_allow_html=True)

        call_px = black_scholes_price(S, K, T, r, sigma, "call", q)
        put_px  = black_scholes_price(S, K, T, r, sigma, "put",  q)
        gap = put_call_parity_gap(call_px, put_px, S, K, T, r, q)
        st.caption(f"Put-call parity gap (BSM): {gap:+.6e} (≈0 within numerical tolerance).")

# ---------- Sensitivities ----------
with tab_sens:
    st.subheader("Price vs Volatility (σ)")

    @st.cache_data(show_spinner=False)
    def _bs_curve_for_sigmas(S, K, T, r, option_type, q, vols: np.ndarray) -> np.ndarray:
        return np.array([black_scholes_price(S, K, T, r, v, option_type, q) for v in vols])

    vols = np.linspace(max(0.01, sigma * 0.3), sigma * 2.0, 60)
    bs_curve = _bs_curve_for_sigmas(S, K, T, r, option_type, q, vols)

    mc_curve = np.array([
        monte_carlo_option_price(S, K, T, r, v, option_type, q,
                                 n_sims=max(20_000, n_sims // 10),
                                 antithetic=True, control_variate=True, seed=int(seed))["price"]
        for v in vols
    ])
    diff_curve = mc_curve - bs_curve

    fig1, ax1 = plt.subplots(figsize=(9.5, 5.2))
    ax1.plot(vols, bs_curve, label="BSM", linewidth=2)
    ax1.plot(vols, mc_curve, label="Monte-Carlo", linewidth=2, linestyle="--")
    above = diff_curve >= 0
    ax1.fill_between(vols, bs_curve, mc_curve, where=above, alpha=0.25, interpolate=True, color="#16a34a")
    ax1.fill_between(vols, bs_curve, mc_curve, where=~above, alpha=0.25, interpolate=True, color="#dc2626")
    ax1.set_xlabel("Volatility σ"); ax1.set_ylabel("Option Price"); ax1.set_title("Option Price vs σ")
    ax1.legend(loc="best")
    st.pyplot(fig1, use_container_width=True)

# ---------- Distribution ----------
with tab_dist:
    st.subheader("Discounted Payoff Histogram (Monte-Carlo samples)")
    samples = sample_discounted_payoffs(
        S, K, T, r, sigma, option_type, q,
        n_sims=min(60_000, max(20_000, n_sims // 5)), seed=int(seed) + 3
    )
    threshold = mc["price"]
    fig2, ax2 = plt.subplots(figsize=(9.5, 5.0))
    counts, bins, patches = ax2.hist(samples, bins=50, alpha=0.9, edgecolor="#ffffff")
    for i, p in enumerate(patches):
        center = 0.5 * (bins[i] + bins[i + 1])
        p.set_facecolor("#16a34a" if center >= threshold else "#dc2626")
    ax2.axvline(threshold, linestyle="--", linewidth=2, color="#2563eb")
    ax2.set_title("Discounted Payoff Histogram"); ax2.set_xlabel("Discounted payoff"); ax2.set_ylabel("Frequency")
    st.pyplot(fig2, use_container_width=True)

# ---------- Heatmaps (vertical, diverging colors, labels) ----------
# ---------- Heatmaps ----------
with tab_heat:
    st.subheader("Heatmaps")

    # Concise controls
    with st.expander("Settings", expanded=True):
        e1, e2, e3 = st.columns(3)
        S_min = e1.number_input("S min", value=float(max(1.0, S * 0.5)), step=1.0)
        S_max = e2.number_input("S max", value=float(S * 1.5), step=1.0)
        nS    = e3.slider("S grid (cols)", 5, 25, 11, 1)

        e4, e5, e6 = st.columns(3)
        v_min = e4.number_input("σ min", value=0.10, step=0.01, format="%.2f")
        v_max = e5.number_input("σ max", value=0.50, step=0.01, format="%.2f")
        nV    = e6.slider("σ grid (rows)", 5, 25, 9, 1)

        metric = st.selectbox("Metric", ["BSM price", "MC price (fast)", "MC − BSM difference"], index=2)
        per_cell_sims = st.slider("MC sims / cell", 2000, 25000, 8000, 1000)
        zero_eps = st.slider("Zero ε (|x|≤ε → white)", 0.0, 0.01, 0.001, 0.001, format="%.3f")
        show_labels = st.checkbox("Show values", value=True)

    # ----- helpers (unchanged logic) -----
    @st.cache_data(show_spinner=False)
    def _bsm_grids(K, T, r, q, S_min, S_max, nS, v_min, v_max, nV):
        S_vals = np.linspace(S_min, S_max, nS)
        V_vals = np.linspace(v_min, v_max, nV)
        call_grid = np.zeros((nV, nS))
        put_grid  = np.zeros((nV, nS))
        for i, vv in enumerate(V_vals):
            for j, ss in enumerate(S_vals):
                call_grid[i, j] = black_scholes_price(ss, K, T, r, vv, "call", q)
                put_grid[i, j]  = black_scholes_price(ss, K, T, r, vv, "put",  q)
        return S_vals, V_vals, call_grid, put_grid

    @st.cache_data(show_spinner=False)
    def _mc_grids(S_min, S_max, nS, v_min, v_max, nV, K, T, r, q, per_cell_sims, seed):
        S_vals = np.linspace(S_min, S_max, nS)
        V_vals = np.linspace(v_min, v_max, nV)
        call_grid = np.zeros((nV, nS))
        put_grid  = np.zeros((nV, nS))
        ctr = 0
        for i, vv in enumerate(V_vals):
            for j, ss in enumerate(S_vals):
                ctr += 1
                sd = int(seed + 10 * ctr)
                call_grid[i, j] = monte_carlo_option_price(
                    ss, K, T, r, vv, "call", q,
                    n_sims=per_cell_sims, antithetic=True, control_variate=True, seed=sd
                )["price"]
                put_grid[i, j] = monte_carlo_option_price(
                    ss, K, T, r, vv, "put", q,
                    n_sims=per_cell_sims, antithetic=True, control_variate=True, seed=sd + 1
                )["price"]
        return S_vals, V_vals, call_grid, put_grid

    S_vals, V_vals, call_bsm, put_bsm = _bsm_grids(K, T, r, q, S_min, S_max, nS, v_min, v_max, nV)
    if metric == "BSM price":
        call_grid, put_grid = call_bsm, put_bsm
    elif metric == "MC price (fast)":
        _, _, call_mc, put_mc = _mc_grids(S_min, S_max, nS, v_min, v_max, nV, K, T, r, q, per_cell_sims, int(seed))
        call_grid, put_grid = call_mc, put_mc
    else:  # MC − BSM difference
        _, _, call_mc, put_mc = _mc_grids(S_min, S_max, nS, v_min, v_max, nV, K, T, r, q, per_cell_sims, int(seed))
        call_grid, put_grid = call_mc - call_bsm, put_mc - put_bsm

    # Zero threshold
    def _apply_zero_eps(Z, eps):
        Z2 = Z.copy()
        Z2[np.abs(Z2) <= eps] = 0.0
        return Z2
    callZ, putZ = _apply_zero_eps(call_grid, zero_eps), _apply_zero_eps(put_grid, zero_eps)

    # Diverging palette centered at 0
    from matplotlib.colors import TwoSlopeNorm
    div_cmap = LinearSegmentedColormap.from_list(
        "red_white_green",
        [(0.00, "#b91c1c"), (0.35, "#fecaca"), (0.50, "#ffffff"),
         (0.65, "#bbf7d0"), (1.00, "#15803d")]
    )
    abs_max = float(max(np.abs(callZ).max(), np.abs(putZ).max(), 1e-9))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    # Edges / centers
    S_edges = np.linspace(S_vals.min(), S_vals.max(), nS + 1)
    V_edges = np.linspace(V_vals.min(), V_vals.max(), nV + 1)
    S_centers = (S_edges[:-1] + S_edges[1:]) / 2
    V_centers = (V_edges[:-1] + V_edges[1:]) / 2

    # ----- draw (labels fixed to 16 pt) -----
    LABEL_FONTSIZE = 16  # << requested fixed size for cell labels

    def _draw_single_heat(Z, title):
        fig, ax = plt.subplots(figsize=(12.5, 6.0))  # one heatmap per row
        im = ax.pcolormesh(
            S_edges, V_edges, Z,
            cmap=div_cmap, norm=norm,
            shading="auto", edgecolors="white", linewidth=0.7
        )
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_xlabel("Spot S"); ax.set_ylabel("Volatility σ")
        ax.set_xticks(np.linspace(S_vals.min(), S_vals.max(), min(8, nS)))
        ax.set_yticks(np.linspace(V_vals.min(), V_vals.max(), min(8, nV)))
        ax.tick_params(labelsize=16)

        if show_labels and (nS * nV <= 400):
            for i, y in enumerate(V_centers):
                for j, x in enumerate(S_centers):
                    val = Z[i, j]
                    if not np.isfinite(val): 
                        continue
                    strength = abs(val) / (abs_max + 1e-12)
                    txt_color  = "#0b1324" if strength < 0.35 else "#ffffff"
                    stroke_col = "#ffffff" if txt_color == "#0b1324" else "#0b1324"
                    t = ax.text(x, y, f"{val:.2f}", ha="center", va="center",
                                fontsize=LABEL_FONTSIZE, color=txt_color, fontweight="bold")
                    t.set_path_effects([pe.withStroke(linewidth=2.2, foreground=stroke_col)])

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Value (centered at 0)")
        cbar.set_ticks(np.linspace(-abs_max, abs_max, 5))
        cbar.ax.tick_params(labelsize=14)
        st.pyplot(fig, use_container_width=True)

    title_call = "CALL (MC–BSM)" if metric.endswith("difference") else ("CALL — MC" if "MC price" in metric else "CALL — BSM")
    title_put  = "PUT  (MC–BSM)" if metric.endswith("difference") else ("PUT  — MC"  if "MC price" in metric else "PUT  — BSM")

    _draw_single_heat(callZ, title_call)
    _draw_single_heat(putZ,  title_put)

    st.markdown("""
        <div style="text-align:center; color:#64748b; font-size:.9rem;">
            Note: White cells indicate values close to zero (|value| ≤ ε).
            Adjust ε to control the threshold for white coloring.
        </div>
    """, unsafe_allow_html=True)
