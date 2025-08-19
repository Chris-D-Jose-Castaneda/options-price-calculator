"""
Black-Scholes–Merton (with dividend yield q)
Greeks + Implied Volatility + Put-Call Parity check
Improvements vs typical demos:
- Includes dividend yield q
- Full Greeks, robust IV via brentq, parity diagnostics
- Safe guards for T→0 and bad inputs
"""
from __future__ import annotations
from typing import Dict, Literal, Tuple
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

Option = Literal["call", "put"]

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> Tuple[float, float]:
    # Guard small T to avoid division by zero (treat ~instantaneous)
    T_eff = max(T, 1e-12)
    if sigma <= 0:
        # Degenerate sigma → treat as almost zero to keep continuity
        sigma = 1e-12
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T_eff) / (sigma * np.sqrt(T_eff))
    d2 = d1 - sigma * np.sqrt(T_eff)
    return d1, d2

def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: Option = "call", q: float = 0.0
) -> float:
    """European option price under BSM with continuous dividend yield q."""
    if T <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)
    if option_type == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    elif option_type == "put":
        return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: Option = "call", q: float = 0.0
) -> Dict[str, float]:
    """Return Δ, Γ, Θ, Vega, ρ (annualized Θ). Vega is per 1.00 vol (not %)."""
    if T <= 0:
        # At expiry, Greeks are undefined/dirac; return zeros
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_m_d1 = norm.cdf(-d1)
    N_d2 = norm.cdf(d2)
    N_m_d2 = norm.cdf(-d2)

    gamma = df_q * n_d1 / (S * sigma * np.sqrt(T))
    vega = S * df_q * n_d1 * np.sqrt(T)

    if option_type == "call":
        delta = df_q * N_d1
        theta = (-S * df_q * n_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * df_r * N_d2
                 + q * S * df_q * N_d1)
        rho = K * T * df_r * N_d2
    else:
        delta = df_q * (N_d1 - 1.0)
        theta = (-S * df_q * n_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * df_r * N_m_d2
                 - q * S * df_q * N_m_d1)
        rho = -K * T * df_r * N_m_d2

    return {"delta": float(delta), "gamma": float(gamma), "theta": float(theta),
            "vega": float(vega), "rho": float(rho)}

def implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float, option_type: Option = "call", q: float = 0.0
) -> float:
    """
    Solve for sigma via brentq. Returns np.nan if no solution exists in [1e-6, 5].
    """
    if T <= 0:
        return np.nan
    def f(sig: float) -> float:
        return black_scholes_price(S, K, T, r, sig, option_type, q) - market_price
    # Bracket
    lo, hi = 1e-6, 5.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return np.nan
    try:
        return float(brentq(f, lo, hi, maxiter=200, xtol=1e-12))
    except Exception:
        return np.nan

def put_call_parity_gap(call: float, put: float, S: float, K: float, T: float, r: float, q: float = 0.0) -> float:
    """
    Gap = C - P - (S e^{-qT} - K e^{-rT}). Should be ~0 for European options.
    """
    return float(call - put - (S * np.exp(-q * T) - K * np.exp(-r * T)))
