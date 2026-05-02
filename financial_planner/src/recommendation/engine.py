"""
Recommendation Engine - Data-Driven Stock Selection
Based on historical performance: return, volatility, drawdown, and goal alignment
"""

import pandas as pd
import numpy as np
from typing import Dict


def safe_normalize(series: pd.Series) -> pd.Series:
    """
    Safe min-max normalization that handles constant series.
    Returns 0.5 for constant series to avoid division by zero.
    """
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def classify_risk(volatility: float) -> str:
    """
    Classify stock risk based on annualized volatility (in percentage).
    
    Criteria:
        Low: < 18%
        Medium: 18% - 30%
        High: > 30%
    """
    if volatility < 18.0:
        return "Low Risk"
    elif volatility < 30.0:
        return "Medium Risk"
    else:
        return "High Risk"


def compute_stock_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract key features from a stock's historical data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    
    Returns
    -------
    dict
        Contains annual_return (%), volatility (%), max_drawdown (%)
    """
    df = df.copy()
    
    # Daily returns
    df["daily_ret"] = df["close"].pct_change()
    
    # Annualized metrics (in percentage)
    annual_return = df["daily_ret"].mean() * 252 * 100
    volatility = df["daily_ret"].std() * np.sqrt(252) * 100
    
    # Maximum drawdown (already in percentage)
    cummax = df["close"].cummax()
    drawdown = (df["close"] - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    return {
        "annual_return": annual_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown
    }


def build_stock_features_dataframe(all_stocks_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a dataframe with features for all stocks."""
    features_list = []
    
    for stock, df in all_stocks_data.items():
        # Skip non-dataframe entries (e.g., metadata dictionaries)
        if not isinstance(df, pd.DataFrame):
            continue
        if "close" not in df.columns:
            continue
        
        try:
            features = compute_stock_features(df)
            features["stock"] = stock
            features_list.append(features)
        except Exception as e:
            print(f"Warning: Could not process {stock}: {e}")
            continue
    
    df_features = pd.DataFrame(features_list)
    df_features = df_features.set_index("stock")
    
    return df_features


def normalize_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features to 0-1 scale.
    For volatility and drawdown, lower values are better (inverted normalization).
    """
    df_norm = df_features.copy()
    
    df_norm["return_norm"] = safe_normalize(df_features["annual_return"])
    df_norm["vol_norm"] = 1 - safe_normalize(df_features["volatility"])
    df_norm["drawdown_norm"] = 1 - safe_normalize(df_features["max_drawdown"])
    
    return df_norm


def get_goal_weights(goal: str) -> Dict[str, float]:
    """
    Return weights based on investment goal.
    
    Wedding (1-3 years): prioritize stability
    KPR (3-5 years): balance between return and stability
    Education (10+ years): prioritize growth
    """
    if "Wedding" in goal:
        return {"return": 0.30, "volatility": 0.45, "drawdown": 0.25}
    elif "KPR" in goal:
        return {"return": 0.45, "volatility": 0.35, "drawdown": 0.20}
    else:
        return {"return": 0.55, "volatility": 0.25, "drawdown": 0.20}


def adjust_weights_for_risk(weights: Dict[str, float], risk_profile: str) -> Dict[str, float]:
    """Adjust weights based on user's risk tolerance."""
    adjusted = weights.copy()
    
    if risk_profile == "Conservative":
        adjusted["volatility"] += 0.10
        adjusted["drawdown"] += 0.10
        adjusted["return"] -= 0.20
    elif risk_profile == "Aggressive":
        adjusted["return"] += 0.15
        adjusted["volatility"] -= 0.10
        adjusted["drawdown"] -= 0.05
    
    # Re-normalize to sum to 1
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()}


def score_stocks(df_norm: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Calculate final score for each stock based on weights."""
    df_score = df_norm.copy()
    
    df_score["final_score"] = (
        weights["return"] * df_score["return_norm"] +
        weights["volatility"] * df_score["vol_norm"] +
        weights["drawdown"] * df_score["drawdown_norm"]
    )
    
    return df_score.sort_values("final_score", ascending=False)


def get_smooth_allocations(scores: pd.Series, top_n: int = 5) -> Dict[str, float]:
    """
    Convert scores to allocations using softmax for smoother distribution.
    """
    # Take top N scores
    top_scores = scores.head(top_n)
    
    std_val = top_scores.std()
    if std_val == 0:
        exp_scores = np.ones(len(top_scores))
    else:
        exp_scores = np.exp(top_scores / std_val)
    
    softmax_alloc = exp_scores / exp_scores.sum()
    
    allocations = {stock: round(alloc * 100, 1) for stock, alloc in zip(top_scores.index, softmax_alloc)}
    
    # Adjust rounding to sum to 100
    total = sum(allocations.values())
    if total != 100:
        first_stock = list(allocations.keys())[0]
        allocations[first_stock] += round(100 - total, 1)
    
    return allocations


def get_strategy_explanation(goal: str, risk_profile: str) -> str:
    """Generate strategy explanation based on goal and risk profile."""
    if "Wedding" in goal:
        base_strategy = "80% low risk + 20% medium risk stocks"
        focus = "capital protection"
    elif "KPR" in goal:
        base_strategy = "60% low risk + 30% medium risk + 10% high risk"
        focus = "balanced return with moderate risk"
    else:
        base_strategy = "50% low risk + 30% medium risk + 20% high risk"
        focus = "long-term growth"
    
    risk_note = ""
    if risk_profile == "Conservative":
        risk_note = " with extra weight on stability"
    elif risk_profile == "Aggressive":
        risk_note = " with extra weight on returns"
    
    return f"Strategy: {base_strategy}. Focus: {focus}{risk_note}."


def generate_recommendation(
    goal: str,
    risk_profile: str,
    df_features: pd.DataFrame,
    df_norm: pd.DataFrame,
    top_n: int = 5
) -> Dict:
    """
    Generate stock recommendation based on goal and risk profile.
    
    Parameters
    ----------
    goal : str
        Investment goal (Wedding, KPR, Education)
    risk_profile : str
        Risk tolerance (Conservative, Moderate, Aggressive)
    df_features : pd.DataFrame
        Raw feature values for all stocks
    df_norm : pd.DataFrame
        Normalized feature values for all stocks
    top_n : int, default 5
        Number of stocks to recommend
    
    Returns
    -------
    dict
        Contains recommended stocks, allocations, scores, and strategy explanation
    """
    # Get base weights and adjust for risk
    weights = get_goal_weights(goal)
    weights = adjust_weights_for_risk(weights, risk_profile)
    
    # Score all stocks
    df_scored = score_stocks(df_norm, weights)
    top_stocks = df_scored.head(top_n)
    
    # Get smooth allocations
    allocations = get_smooth_allocations(top_stocks["final_score"], top_n)
    
    # Add risk levels for each recommended stock
    risk_levels = {}
    for stock in top_stocks.index:
        vol = df_features.loc[stock, "volatility"]
        risk_levels[stock] = classify_risk(vol)
    
    strategy_text = get_strategy_explanation(goal, risk_profile)
    
    # Generate explanation and risk warning based on goal
    if "Wedding" in goal:
        explanation = f"Based on your wedding goal with {risk_profile.lower()} risk profile, these stocks prioritize stability and capital protection. Selected from {len(df_features)} LQ45 stocks (2019-2025 historical data)."
        risk_warning = "High volatility stocks are not recommended for short-term goals. Consider shifting to lower risk assets if less than 2 years remaining."
    elif "KPR" in goal:
        explanation = f"For your KPR goal with {risk_profile.lower()} risk profile, these stocks offer balanced return with moderate risk. Bank stocks in this list can provide 3-4% dividend yield to help cover monthly payments."
        risk_warning = "Avoid high volatility stocks for down payment savings. Consider moving to safer instruments as your deadline approaches."
    else:
        explanation = f"Your education goal with {risk_profile.lower()} risk profile prioritizes long-term growth. Consumer stocks (ICBP, INDF, UNVR) serve as natural inflation hedge."
        risk_warning = "Education fund allows taking calculated risks. Long time horizon benefits from market growth despite short-term volatility."
    
    return {
        "recommended_stocks": top_stocks.index.tolist(),
        "allocations": allocations,
        "scores": top_stocks["final_score"].to_dict(),
        "risk_levels": risk_levels,
        "strategy": strategy_text,
        "weights_used": weights,
        "explanation": explanation,
        "risk_warning": risk_warning
    }