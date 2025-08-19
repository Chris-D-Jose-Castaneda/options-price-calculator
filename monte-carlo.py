"""
Monte Carlo European Option Pricer (with variance reduction)
- Antithetic variates
- Control variate using ST (known E[ST] = S e^{(r-q)T})
- Returns price, stderr, and 95% CI
"""
from __future__ import annotations
from typing import Literal, Dict, Tuple
import numpy as np

Option = Literal["call", "put"]

def _terminal_prices(
    S: float, T: float, r: float, sigma: float, q: float, n_sims: int, antithetic: bool, seed: int | None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if antithetic:
        m = n_sims // 2
        Z = rng.standard_normal(m)
        Z = np.concatenate([Z, -Z])
        if Z.size < n_sims:  # odd case
            Z = np.append(Z, rng.standard_normal(1))
    else:
        Z = rng.standard_normal(n_sims)
    mu = (r - q - 0.5 * sigma * sigma) * T
    sig = sigma * np.sqrt(T)
    ST = S * np.exp(mu + sig * Z)
    return ST

def monte_carlo_option_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: Option = "call",
    q: float = 0.0, n_sims: int = 100_000, antithetic: bool = True, control_variate: bool = True,
    seed: int | None = 42
) -> Dict[str, float]:
    """
    Returns dict: price, stderr, ci_low, ci_high, n_sims
    """
    if T <= 0:
        price = max(0.0, (S - K) if option_type == "call" else (K - S))
        return {"price": float(price), "stderr": 0.0, "ci_low": float(price), "ci_high": float(price), "n_sims": 0}

    df_r = np.exp(-r * T)
    ST = _terminal_prices(S, T, r, sigma, q, n_sims, antithetic, seed)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    Y = df_r * payoff  # discounted payoff

    if control_variate:
        # Control: X = ST, with known E[X] = S*exp((r-q)T)
        X = ST
        EX = S * np.exp((r - q) * T)
        # Optimal b* = Cov(Y, X) / Var(X)
        covYX = np.cov(Y, X, ddof=1)[0, 1]
        varX = np.var(X, ddof=1)
        b_star = 0.0 if varX == 0 else covYX / varX
        Y = Y - b_star * (X - EX)

    price = float(np.mean(Y))
    # standard error of the mean
    stderr = float(np.std(Y, ddof=1) / np.sqrt(len(Y)))
    ci_low = price - 1.96 * stderr
    ci_high = price + 1.96 * stderr
    return {"price": price, "stderr": stderr, "ci_low": float(ci_low), "ci_high": float(ci_high), "n_sims": int(len(Y))}

def sample_discounted_payoffs(
    S: float, K: float, T: float, r: float, sigma: float, option_type: Option = "call",
    q: float = 0.0, n_sims: int = 20_000, seed: int | None = 7
) -> np.ndarray:
    """Utility for plotting payoff distribution (no variance reduction)."""
    df_r = np.exp(-r * T)
    ST = _terminal_prices(S, T, r, sigma, q, n_sims, antithetic=False, seed=seed)
    payoff = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)
    return df_r * payoff
