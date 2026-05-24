"""
================================================================================
LQ45 FINANCIAL PLANNING ENGINE
================================================================================
Professional Financial Planning Tool
Based on LQ45 Indonesia Stock Exchange Data (July 2019 - February 2025)
Methodology: Crisis-Weighted Bootstrap Simulation (10,000 paths)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date, datetime
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import load_all_stocks
from src.features.returns import calculate_returns
from src.features.volatility import calculate_volatility
from src.features.drawdown import calculate_drawdown
from src.simulation.bootstrap import BootstrapSimulator, add_crisis_weights
from src.utils.constants import SECTOR_MAP
from src.advisory.lq45_overview import LQ45_OVERVIEW

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="LQ45 Financial Planning Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    return load_all_stocks()

EMITEN, STOCK_DATA = load_data()

# ============================================================================
# COLOR SYSTEM
# ============================================================================
COLORS = {
    "primary": "#1a73e8",
    "success": "#34a853",
    "warning": "#fbbc04",
    "danger": "#ea4335",
    "dark": "#1e293b",
    "text": "#334155",
    "text_secondary": "#64748b",
    "background": "#f8fafc",
    "surface": "#ffffff",
    "border": "#e2e8f0"
}

# ============================================================================
# PLOTLY CUSTOM TEMPLATE FOR DARK TEXT
# ============================================================================
custom_template = pio.templates["simple_white"]
custom_template.layout.title.font.color = COLORS["dark"]
custom_template.layout.font.color = COLORS["text"]
custom_template.layout.xaxis.title.font.color = COLORS["text"]
custom_template.layout.yaxis.title.font.color = COLORS["text"]
custom_template.layout.xaxis.tickfont.color = COLORS["text_secondary"]
custom_template.layout.yaxis.tickfont.color = COLORS["text_secondary"]
custom_template.layout.legend.font.color = COLORS["text"]
pio.templates["custom"] = custom_template
pio.templates.default = "custom"

# ============================================================================
# CSS STYLING - COMPLETE FIX FOR ALL TEXT
# ============================================================================
st.markdown(f"""
<style>
    /* Base container */
    .stApp {{
        background-color: {COLORS["background"]};
    }}
    
    /* Force all text to be dark */
    .stApp, .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, .stMarkdown li, .stMarkdown ul,
    .stMarkdown ol, .stMarkdown blockquote, label, .stTextInput label, .stNumberInput label,
    .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label,
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] div,
    div[data-testid="stMarkdown"] p {{
        color: {COLORS["dark"]} !important;
    }}
    
    /* Keep secondary text lighter */
    .stCaption, caption, .stTextInput input::placeholder,
    .stNumberInput input::placeholder {{
        color: {COLORS["text_secondary"]} !important;
    }}
    
    /* Buttons - keep white text on colored background */
    .stButton button, .stButton button * {{
        color: white !important;
    }}
    .stButton button {{
        background-color: {COLORS["primary"]} !important;
        border-radius: 2rem !important;
        padding: 0.4rem 1.2rem !important;
        font-weight: 500 !important;
        border: none !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }}
    .stButton button:hover {{
        background-color: #0d5bb9 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }}
    
    /* Input fields */
    .stTextInput input, .stNumberInput input, 
    div[data-baseweb="select"] > div,
    .stDateInput input {{
        background-color: {COLORS["surface"]} !important;
        border-radius: 0.5rem !important;
        border: 1px solid {COLORS["border"]} !important;
        color: {COLORS["dark"]} !important;
    }}
    
    .stTextInput input:focus, .stNumberInput input:focus {{
        border-color: {COLORS["primary"]} !important;
        box-shadow: 0 0 0 2px rgba(26,115,232,0.1) !important;
        outline: none !important;
    }}
    
    /* Selectbox dropdown */
    div[data-baseweb="select"] div[role="button"] {{
        color: {COLORS["dark"]} !important;
    }}
    
    div[data-baseweb="select"] ul {{
        background-color: {COLORS["surface"]} !important;
    }}
    
    div[data-baseweb="select"] li {{
        color: {COLORS["dark"]} !important;
        background-color: {COLORS["surface"]} !important;
    }}
    
    div[data-baseweb="select"] li:hover {{
        background-color: #f1f5f9 !important;
    }}
    
    /* Radio buttons */
    .stRadio div[role="radiogroup"] {{
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }}
    
    .stRadio div[role="radiogroup"] label {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 2rem;
        padding: 0.4rem 1rem;
        color: {COLORS["dark"]} !important;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    
    .stRadio div[role="radiogroup"] label:hover {{
        background-color: #f1f5f9;
        border-color: {COLORS["primary"]};
    }}
    
    .stRadio div[role="radiogroup"] label[data-state="checked"] {{
        background-color: {COLORS["primary"]};
        border-color: {COLORS["primary"]};
    }}
    
    .stRadio div[role="radiogroup"] label[data-state="checked"] span {{
        color: white !important;
    }}
    
    .stRadio div[role="radiogroup"] label span {{
        color: {COLORS["dark"]} !important;
    }}
    
    /* Checkbox */
    .stCheckbox {{
        margin: 0.25rem 0;
    }}
    
    .stCheckbox label {{
        color: {COLORS["dark"]} !important;
    }}
    
    .stCheckbox label span {{
        color: {COLORS["dark"]} !important;
    }}
    
    .stCheckbox label div {{
        color: {COLORS["dark"]} !important;
    }}
    
    div[data-testid="stCheckbox"] label {{
        color: {COLORS["dark"]} !important;
    }}
    
    div[data-testid="stCheckbox"] label span {{
        color: {COLORS["dark"]} !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {COLORS["surface"]} !important;
        color: {COLORS["dark"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
    }}
    
    .streamlit-expanderHeader p {{
        color: {COLORS["dark"]} !important;
    }}
    
    .streamlit-expanderHeader:hover {{
        background-color: #f1f5f9 !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: {COLORS["surface"]} !important;
        padding: 0.5rem 0 !important;
    }}
    
    .streamlit-expanderContent .stCheckbox label {{
        color: {COLORS["dark"]} !important;
    }}
    
    .streamlit-expanderContent .stCheckbox label span {{
        color: {COLORS["dark"]} !important;
    }}
    
    .streamlit-expanderContent p {{
        color: {COLORS["dark"]} !important;
    }}
    
    /* Expander details */
    details.streamlit-expander {{
        background-color: {COLORS["surface"]} !important;
    }}
    
    details.streamlit-expander summary {{
        color: {COLORS["dark"]} !important;
    }}
    
    details.streamlit-expander div {{
        color: {COLORS["dark"]} !important;
    }}
    
    /* Slider */
    .stSlider .stSlider > div {{
        color: {COLORS["primary"]} !important;
    }}
    
    .stSlider label {{
        color: {COLORS["dark"]} !important;
    }}
    
    /* Metric cards - BLUE BACKGROUND WITH WHITE TEXT */
    .stat-card {{
        background-color: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}
    .stat-value {{
        font-size: 1.5rem;
        font-weight: 500;
        color: black !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }}
    .stat-label {{
        font-size: 0.7rem;
        color: #cbd5e1 !important;
        text-transform: uppercase;
        font-weight: 500;
        letter-spacing: 0.3px;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1.5rem;
        border-bottom: 1px solid {COLORS["border"]};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 500;
        color: {COLORS["text_secondary"]} !important;
        padding: 0.5rem 0;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS["primary"]} !important;
        border-bottom: 2px solid {COLORS["primary"]};
    }}
    .stTabs [data-baseweb="tab"] p {{
        color: inherit !important;
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS["surface"]};
        border-right: 1px solid {COLORS["border"]};
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div[data-testid="stMetric"] label,
    section[data-testid="stSidebar"] div[data-testid="stMetric"] div {{
        color: {COLORS["dark"]} !important;
    }}
    
    /* Alert boxes */
    .stAlert {{
        background-color: {COLORS["surface"]};
        border-left: 4px solid {COLORS["primary"]};
        border-radius: 0.5rem;
    }}
    .stAlert p {{
        color: {COLORS["dark"]} !important;
    }}
    
    .stSuccess {{
        border-left-color: {COLORS["success"]} !important;
    }}
    .stWarning {{
        border-left-color: {COLORS["warning"]} !important;
    }}
    .stError {{
        border-left-color: {COLORS["danger"]} !important;
    }}
    .stInfo {{
        border-left-color: {COLORS["primary"]} !important;
    }}
    
    
    /* Dataframe */
    .stDataFrame {{
        border-radius: 0.5rem;
        overflow: hidden;
    }}
    .dataframe th {{
        background-color: #f1f5f9 !important;
        color: {COLORS["dark"]} !important;
        font-weight: 600 !important;
    }}
    .dataframe td {{
        color: {COLORS["text"]} !important;
    }}
    
    /* Divider */
    hr {{
        border-color: {COLORS["border"]};
        margin: 1rem 0;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        padding: 1.5rem;
        color: {COLORS["text_secondary"]};
        font-size: 0.7rem;
        border-top: 1px solid {COLORS["border"]};
        margin-top: 2rem;
    }}
    
    /* Main title */
    .main-title {{
        font-size: 1.75rem;
        font-weight: 600;
        color: {COLORS["dark"]};
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }}
    .subtitle {{
        font-size: 0.85rem;
        color: {COLORS["text_secondary"]};
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {COLORS["primary"]};
        display: inline-block;
    }}
    
    /* Number input value */
    .stNumberInput input {{
        color: {COLORS["dark"]} !important;
    }}
    
    /* Selectbox label */
    .stSelectbox label {{
        color: {COLORS["dark"]} !important;
    }}
    
    /* Caption */
    .stCaption {{
        color: {COLORS["text_secondary"]} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def format_idr(value):
    """Format number as Indonesian Rupiah with thousands separator"""
    return f"Rp {value:,.0f}".replace(",", ".")

def format_number(value):
    return f"{value:,.0f}".replace(",", ".")

def prepare_simulation_data(df, use_crisis_weight):
    df = df.copy()
    df = calculate_returns(df)
    if use_crisis_weight:
        df = add_crisis_weights(df)
        df_clean = df.dropna(subset=['daily_return', 'bootstrap_weight'])
        if len(df_clean) == 0:
            return None, None
        returns = df_clean["daily_return"].values
        weights = df_clean["bootstrap_weight"].values
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        return returns, weights
    returns = df["daily_return"].dropna().values
    return returns, None

# ============================================================================
# SESSION STATE
# ============================================================================
if "plan_created" not in st.session_state:
    st.session_state.plan_created = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "financial_goal" not in st.session_state:
    st.session_state.financial_goal = "Wedding Fund"
if "target_nominal" not in st.session_state:
    st.session_state.target_nominal = 100_000_000
if "investment_horizon" not in st.session_state:
    st.session_state.investment_horizon = 2.0
if "monthly_contribution" not in st.session_state:
    st.session_state.monthly_contribution = 2_000_000
if "risk_tolerance" not in st.session_state:
    st.session_state.risk_tolerance = "Moderate"
if "crisis_weight_enabled" not in st.session_state:
    st.session_state.crisis_weight_enabled = True
if "portfolio_stocks" not in st.session_state:
    st.session_state.portfolio_stocks = ["BBCA", "BBRI"]
if "stock_weights" not in st.session_state:
    st.session_state.stock_weights = {"BBCA": 50, "BBRI": 50}
if "calculation_complete" not in st.session_state:
    st.session_state.calculation_complete = False

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### Analysis Parameters")
    st.divider()
    
    st.markdown("**Dataset Summary**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Stocks", len(EMITEN))
    with col_b:
        st.metric("Trading Days", "1,355")
    
    col_c, col_d = st.columns(2)
    with col_c:
        st.metric("Period", "5.5 years")
    with col_d:
        st.metric("Simulations", "10,000")
    
    st.divider()
    
    st.markdown("**Simulation Settings**")
    crisis_toggle = st.checkbox("Crisis Weighting (3x COVID Period)", value=st.session_state.crisis_weight_enabled)
    st.session_state.crisis_weight_enabled = crisis_toggle
    st.caption("COVID-19 crisis period (March-June 2020) receives 3x sampling weight in simulations")
    
    st.divider()
    
    st.markdown("**Documentation**")
    with st.expander("About LQ45 Index"):
        st.markdown(LQ45_OVERVIEW)
    
    st.caption("Data source: Indonesia Stock Exchange (IDX)")
    st.caption("Data compiled by: wildangunawan/Dataset-Saham-IDX (GitHub)")
    st.caption("Methodology: Bootstrap Simulation")

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<p class="main-title">LQ45 Financial Planning Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Crisis-Weighted Bootstrap Simulation | 2019-2025</p>', unsafe_allow_html=True)

# ============================================================================
# MAIN TABS
# ============================================================================
tab_planning, tab_explorer, tab_analysis = st.tabs([
    "Financial Planning",
    "Stock Data Explorer",
    "Statistic Analysis"
])

# ============================================================================
# TAB 1: FINANCIAL PLANNING
# ============================================================================
with tab_planning:
    st.markdown("### Goal Configuration")
    
    input_col, summary_col = st.columns([1, 1])
    
    with input_col:
        st.markdown("#### Personal Information")
        user_name = st.text_input("Investor Name", value=st.session_state.user_name, placeholder="Enter your name")
        st.session_state.user_name = user_name
        
        st.markdown("#### Financial Goal")
        goal_selection = st.selectbox(
            "Select Goal Type",
            ["Wedding Fund", "KPR Down Payment", "Child Education"],
            index=["Wedding Fund", "KPR Down Payment", "Child Education"].index(st.session_state.financial_goal)
        )
        if goal_selection != st.session_state.financial_goal:
            st.session_state.toast_shown = False
        st.session_state.financial_goal = goal_selection
        
        target_amount = st.number_input(
            "Target Amount (Rupiah)",
            min_value=10_000_000,
            max_value=5_000_000_000,
            value=st.session_state.target_nominal,
            step=50_000_000
        )
        if target_amount != st.session_state.target_nominal:
            st.session_state.toast_shown = False
        st.session_state.target_nominal = target_amount
        
        if goal_selection == "Wedding Fund":
            horizon = st.slider("Investment Horizon (Years)", 1.0, 50.0, st.session_state.investment_horizon, 0.5,
            help="Recommended:1-3 years. You can extend for testing scenarios.")
        elif goal_selection == "KPR Down Payment":
            horizon = st.slider("Investment Horizon (Years)", 1.0, 50.0, st.session_state.investment_horizon, 0.5,
            help="Recommended:3-5 years. You can extend for testing scenarios.")
        else:
            horizon = st.slider("Investment Horizon (Years)", 1.0, 50.0, st.session_state.investment_horizon, 1.0,
            help="Recommended:10+ years. You can extend for testing scenarios.")
        
        if horizon != st.session_state.investment_horizon:
            st.session_state.toast_shown = False
        st.session_state.investment_horizon = horizon
        
        monthly_save = st.number_input(
            "Monthly Saving (Rupiah)",
            min_value=500_000,
            max_value=50_000_000,
            value=st.session_state.monthly_contribution,
            step=500_000
        )
        if monthly_save != st.session_state.monthly_contribution:
            st.session_state.toast_shown = False
        st.session_state.monthly_contribution = monthly_save
        
        st.markdown("#### Risk Profile")
        risk_level = st.selectbox(
            "Select Risk Tolerance",
            ["Conservative", "Moderate", "Aggressive"],
            index=["Conservative", "Moderate", "Aggressive"].index(st.session_state.risk_tolerance)
        )
        st.session_state.risk_tolerance = risk_level
        
        st.markdown("#### Portfolio Construction")
        
        # ============================================================
        # RECOMMENDED PORTFOLIO BASED ON GOAL (FROM LQ45 ANALYSIS)
        # ============================================================
        
        recommended_portfolios = {
            "Wedding Fund": {
                "stocks": ["BBCA", "TLKM", "ASII"],
                "weights": [50, 30, 20],
                "risk_focus": "Low Risk (<1.5% volatility)",
                "note": "Capital preservation for 1-3 year horizon"
            },
            "KPR Down Payment": {
                "stocks": ["BBRI", "BMRI", "BBCA"],
                "weights": [40, 35, 25],
                "risk_focus": "Medium Risk (1.5-3% volatility)",
                "note": "Balanced growth with dividend synergy for 3-5 year horizon"
            },
            "Child Education": {
                "stocks": ["ADRO", "ITMG", "BBRI"],
                "weights": [40, 35, 25],
                "risk_focus": "High Risk (>3% volatility)",
                "note": "Maximum growth for 10+ year horizon, higher dividend potential"
            }
        }
        
        # Get recommendation for current goal
        current_goal = st.session_state.financial_goal
        rec = recommended_portfolios.get(current_goal, recommended_portfolios["Wedding Fund"])
        
        # Show recommendation with clear instruction
        st.info(f"💡 **Recommended for {current_goal}:** {rec['note']}")
        st.caption(f"Risk focus: {rec['risk_focus']}")
        st.caption(f"Suggested stocks: {', '.join(rec['stocks'])} with weights {rec['weights']}")
        
        # Apply button
        if st.button("Apply This Recommendation", key="apply_rec_btn"):
            st.session_state.portfolio_stocks = rec["stocks"].copy()
            st.session_state.stock_weights = {}
            for stock, weight in zip(rec["stocks"], rec["weights"]):
                st.session_state.stock_weights[stock] = weight
            st.session_state.calculation_complete = True
            st.success(f"Recommended portfolio applied! You can still adjust below.")
            st.rerun()
        
        st.markdown("---")
        st.caption("Or select your own stocks below:")
        
        # Continue with existing stock selection
        sector_groups = {}
        for stock in EMITEN:
            sector = SECTOR_MAP.get(stock, "Other")
            sector_groups.setdefault(sector, []).append(stock)
        
        portfolio_stocks = []
        for sector, stocks in sorted(sector_groups.items()):
            with st.expander(f"{sector} ({len(stocks)})"):
                for stock in stocks:
                    is_selected = stock in st.session_state.portfolio_stocks
                    if st.checkbox(stock, value=is_selected, key=f"port_{stock}"):
                        portfolio_stocks.append(stock)
        
        st.session_state.portfolio_stocks = list(set(portfolio_stocks))
        
        if st.session_state.portfolio_stocks:
            st.markdown("**Allocation Percentage**")
            total_weight = 0
            default_weight = 100 // len(st.session_state.portfolio_stocks) if st.session_state.portfolio_stocks else 0
            for stock in st.session_state.portfolio_stocks:
                current = st.session_state.stock_weights.get(stock, default_weight)
                weight = st.slider(stock, 0, 100, current, 5, key=f"weight_{stock}")
                st.session_state.stock_weights[stock] = weight
                total_weight += weight
            st.progress(min(total_weight / 100, 1.0), text=f"Total: {total_weight}%")
            
            if total_weight != 100:
                st.warning("Total allocation must equal 100 percent")
            else:
                st.success("Portfolio balanced")
    
    with summary_col:
        st.markdown("#### Plan Summary")
        
        if st.session_state.user_name:
            st.write(f"**Investor:** {st.session_state.user_name}")
        st.write(f"**Goal:** {st.session_state.financial_goal}")
        st.write(f"**Target:** {format_idr(st.session_state.target_nominal)}")
        st.write(f"**Horizon:** {st.session_state.investment_horizon:.0f} years")
        st.write(f"**Monthly:** {format_idr(st.session_state.monthly_contribution)}")
        st.write(f"**Risk Profile:** {st.session_state.risk_tolerance}")
        
        st.markdown("---")
        st.markdown("**Portfolio Allocation**")
        has_allocation = False
        for stock, weight in st.session_state.stock_weights.items():
            if weight > 0 and stock in st.session_state.portfolio_stocks:
                st.write(f"- {stock}: {weight}%")
                has_allocation = True
        if not has_allocation:
            st.write("(No stocks selected)")
        
        st.markdown("---")
        
        total_weight = sum(st.session_state.stock_weights.values())
        ready_to_calculate = (
            st.session_state.user_name and
            st.session_state.portfolio_stocks and
            total_weight == 100
        )
        
        if ready_to_calculate:
            if st.button("Calculate Plan", use_container_width=True):
                st.session_state.toast_shown = False
                st.session_state.calculation_complete = True
                st.rerun()
    
    if st.session_state.calculation_complete:
        if "toast_shown" not in st.session_state:
            st.toast("Calculation Complete! Your financial plan is ready.", icon="🎯")
            st.session_state.toast_shown = True
        st.divider()
        st.markdown("### Calculation Results")
        
        # ============================================================
        # NOTIFICATION BOX
        # ============================================================
        st.success("**Calculation Complete!** Your financial plan has been generated below.")
        st.info("**What you can do next:**")
        st.markdown("""
        - Adjust your monthly saving or horizon using the controls above
        - Modify your portfolio allocation in the Portfolio Construction section
        - Click **Calculate Plan** again to regenerate your projection
        - Review your success probability and dividend analysis below
        """)

        active_stocks = [s for s in st.session_state.portfolio_stocks if st.session_state.stock_weights.get(s, 0) > 0]
        
        if not active_stocks:
            st.error("No stocks selected for simulation")
        else:
            proxy_stock = active_stocks[0]
            source_data = STOCK_DATA[proxy_stock].copy()
            
            returns_data, sampling_weights = prepare_simulation_data(source_data, st.session_state.crisis_weight_enabled)
            
            if returns_data is not None and len(returns_data) > 0:
                simulator = BootstrapSimulator(returns=returns_data, weights=sampling_weights)
                
                expected_return_map = {"Conservative": 0.08, "Moderate": 0.10, "Aggressive": 0.12}
                annual_return = expected_return_map.get(st.session_state.risk_tolerance, 0.10)
                
                total_months = st.session_state.investment_horizon * 12
                
                # Handle zero months
                if total_months <= 0:
                    total_months = 12
                
                monthly_rate = annual_return / 12
                
                # Calculate required monthly saving
                if monthly_rate > 0 and total_months > 0:
                    try:
                        # Future value of annuity formula
                        required_monthly = st.session_state.target_nominal * monthly_rate / ((1 + monthly_rate) ** total_months - 1)
                        
                        # Jika required_monthly lebih kecil dari 1 rupiah atau lebih besar dari target, gunakan simple division
                        if required_monthly < 1 or required_monthly > st.session_state.target_nominal:
                            required_monthly = st.session_state.target_nominal / total_months
                    except (OverflowError, ZeroDivisionError):
                        required_monthly = st.session_state.target_nominal / total_months
                else:
                    required_monthly = st.session_state.target_nominal / total_months
                
                # Final validation
                if np.isnan(required_monthly) or np.isinf(required_monthly) or required_monthly <= 0:
                    required_monthly = st.session_state.target_nominal / total_months
                
                # Batasi required_monthly tidak lebih dari target
                if required_monthly > st.session_state.target_nominal:
                    required_monthly = st.session_state.target_nominal / total_months
                
                # Ensure required_monthly is finite
                if np.isnan(required_monthly) or np.isinf(required_monthly) or required_monthly <= 0:
                    required_monthly = st.session_state.target_nominal / total_months
                
                total_contributions = st.session_state.monthly_contribution * total_months
                
                try:
                    simulation_result = simulator.simulate_with_target(
                        total_contributions,
                        st.session_state.target_nominal,
                        st.session_state.investment_horizon
                    )
                    success_rate = simulation_result.probability_success
                    final_values = simulation_result.final_values
                    median_value = np.percentile(final_values, 50)
                    var_estimate = np.percentile(final_values, 5)
                    upper_estimate = np.percentile(final_values, 95)
                except Exception as sim_error:
                    success_rate = 50.0
                    median_value = total_contributions * (1 + annual_return) ** st.session_state.investment_horizon
                    var_estimate = median_value * 0.7
                    upper_estimate = median_value * 1.5
                
                result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                
                with result_col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{success_rate:.0f}%</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{format_idr(median_value)}</div>
                        <div class="stat-label">Median Outcome</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with result_col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{format_idr(var_estimate)}</div>
                        <div class="stat-label">VaR 95% (Worst Case)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with result_col4:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value">{format_idr(upper_estimate)}</div>
                        <div class="stat-label">Optimistic (95th Percentile)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### Savings & Monthly Analysis")
                
                col_s1, col_s2, col_s3 = st.columns(3)
                
                with col_s1:
                    st.metric("Investment Horizon", f"{st.session_state.investment_horizon:.1f} years")
                    st.metric("Target Amount", format_idr(st.session_state.target_nominal))
                    st.metric("Total Contributions", format_idr(total_contributions))
                
                with col_s2:
                    st.metric("Your Monthly Saving", format_idr(st.session_state.monthly_contribution))
                    st.metric("Required Monthly", format_idr(required_monthly))
                    
                    diff = st.session_state.monthly_contribution - required_monthly
                    if diff >= 0:
                        st.success(f"Surplus: +{format_idr(diff)}/month")
                        progress_val = min(1.0, st.session_state.monthly_contribution / required_monthly if required_monthly > 0 else 1.0)
                        st.progress(progress_val)
                    else:
                        st.error(f"Shortfall: {format_idr(abs(diff))}/month")
                        pct = (st.session_state.monthly_contribution / required_monthly) * 100 if required_monthly > 0 else 0
                        st.caption(f"at {pct:.0f}% of required")
                        progress_val = min(1.0, st.session_state.monthly_contribution / required_monthly if required_monthly > 0 else 1.0)
                        st.progress(progress_val)
                
                with col_s3:
                    gap = st.session_state.target_nominal - total_contributions
                    if gap > 0:
                        st.metric("Return Needed", format_idr(gap))
                    else:
                        st.metric("Status", "Target Achieved", delta="by contributions alone")
                    
                    # Simple projection
                    projected = total_contributions * (1 + annual_return) ** st.session_state.investment_horizon
                    st.caption(f"Projected (no dividend): {format_idr(projected)}")

                # SIMPLIFIED CALCULATION 
                st.markdown("#### Dividend Impact Analysis")
                
                # Average dividend yield from portfolio
                dividend_yield_map = {
                    "BBCA": 0.032, "BBRI": 0.0497, "BMRI": 0.045, "TLKM": 0.038,
                    "ASII": 0.035, "ADRO": 0.065, "ITMG": 0.095, "PTBA": 0.075
                }
                
                total_dividend_yield = 0
                for stock, weight in st.session_state.stock_weights.items():
                    if weight > 0 and stock in dividend_yield_map:
                        total_dividend_yield += dividend_yield_map.get(stock, 0.04) * (weight / 100)
                
                years = st.session_state.investment_horizon
                
                # Simple future value with and without dividends
                # Without dividends: only price appreciation
                future_without_div = total_contributions * (1 + annual_return) ** years
                
                # With dividends: add annual dividend income (not reinvested to avoid compounding errors)
                annual_dividend_income = total_contributions * total_dividend_yield
                future_with_div = future_without_div + (annual_dividend_income * years)
                
                # Ensure numbers are reasonable
                future_without_div = min(future_without_div, st.session_state.target_nominal * 3)
                future_with_div = min(future_with_div, st.session_state.target_nominal * 3)
                
                col_div1, col_div2, col_div3 = st.columns(3)
                
                with col_div1:
                    pct_target = (future_without_div / st.session_state.target_nominal) * 100
                    st.metric(
                        "Price Appreciation Only", 
                        format_idr(future_without_div),
                        delta=f"{pct_target:.0f}% of target"
                    )
                
                with col_div2:
                    pct_target_div = (future_with_div / st.session_state.target_nominal) * 100
                    st.metric(
                        "Price With Dividends",
                        format_idr(future_with_div),
                        delta=f"{pct_target_div:.0f}% of target"
                    )
                
                with col_div3:
                    st.metric(
                        "Dividend Yield",
                        f"{total_dividend_yield * 100:.2f}%",
                        delta=f"~{format_idr(annual_dividend_income)}/year"
                    )
                
                # Show which scenario meets target
                if future_without_div >= st.session_state.target_nominal:
                    st.success("Without dividends: Your target is achievable")
                elif future_with_div >= st.session_state.target_nominal:
                    st.success("With dividends: Your target becomes achievable")
                else:
                    needed = st.session_state.target_nominal - future_with_div
                    st.warning(f"Remaining shortfall: {format_idr(needed)}")
                
                # Capital breakdown
                st.markdown("#### Capital Breakdown")
                
                # Calculate future value of monthly contributions (annuity)
                monthly_rate = annual_return / 12
                total_months = int(st.session_state.investment_horizon * 12)
                
                if monthly_rate > 0:
                    # Future value of ordinary annuity
                    fv_annuity = st.session_state.monthly_contribution * ((1 + monthly_rate) ** total_months - 1) / monthly_rate
                else:
                    fv_annuity = st.session_state.monthly_contribution * total_months
                
                # With dividends (simplified: add average dividend yield to annual return)
                annual_return_with_div = annual_return + total_dividend_yield
                monthly_rate_with_div = annual_return_with_div / 12
                
                if monthly_rate_with_div > 0:
                    fv_with_div = st.session_state.monthly_contribution * ((1 + monthly_rate_with_div) ** total_months - 1) / monthly_rate_with_div
                else:
                    fv_with_div = fv_annuity
                
                # Capital Gains = Growth from price appreciation only
                capital_gains = fv_annuity - total_contributions
                
                # Dividend Income = Extra from dividends
                dividend_income = fv_with_div - fv_annuity
                
                col_cap1, col_cap2, col_cap3 = st.columns(3)
                
                with col_cap1:
                    st.metric("Your Contributions", format_idr(total_contributions))
                with col_cap2:
                    st.metric("Capital Gains (Price Appreciation)", format_idr(max(0, capital_gains)))
                with col_cap3:
                    st.metric("Dividend Income (Reinvested)", format_idr(max(0, dividend_income)))
                
                # Total projected value
                total_projected = fv_with_div
                pct_of_target = (total_projected / st.session_state.target_nominal) * 100
                st.caption(f"**Total Projected Value:** {format_idr(total_projected)} -({pct_of_target:.0f}% of target)")
                
                # ============================================================
                # SHOW DIVIDEND YIELD PER STOCK (TAMBAHKAN DI SINI)
                # ============================================================
                
                # Show dividend yield for selected stocks
                if st.session_state.portfolio_stocks:
                    st.markdown("#### Dividend Yield per Stock (Historical Average)")
                    
                    # Dividend yield data from Table 3.7
                    dividend_yield_map = {
                        "BBCA": 0.032, "BBRI": 0.0497, "BMRI": 0.045, "TLKM": 0.038,
                        "ASII": 0.035, "ADRO": 0.065, "ITMG": 0.095, "PTBA": 0.075,
                        "UNVR": 0.035, "INDF": 0.035, "ICBP": 0.025, "GGRM": 0.045,
                        "CPIN": 0.030, "PGAS": 0.040, "BBNI": 0.045, "MEDC": 0.020,
                        "AKRA": 0.025, "AMRT": 0.015, "ARTO": 0.000, "BRPT": 0.020,
                        "EXCL": 0.035, "HRUM": 0.050, "ISAT": 0.045, "JSMR": 0.030,
                        "KLBF": 0.025, "MAPA": 0.020, "MAPI": 0.015, "MDKA": 0.010,
                        "SIDO": 0.040, "SMGR": 0.035, "TOWR": 0.030, "UNTR": 0.080,
                        "BTN": 0.025
                    }   
                    
                    div_data = []
                    for stock in st.session_state.portfolio_stocks:
                        if stock in dividend_yield_map:
                            yield_pct = dividend_yield_map[stock]
                            # Risk tier berdasarkan yield
                            if yield_pct > 0.07:
                                risk_tier = "High Risk"
                            elif yield_pct > 0.04:
                                risk_tier = "Medium Risk"
                            else:
                                risk_tier = "Low Risk"
                            
                            div_data.append({
                                "Stock": stock,
                                "Dividend Yield": f"{yield_pct*100:.1f}%",
                                "Risk Tier": risk_tier
                            })
                    
                    if div_data:
                        st.dataframe(pd.DataFrame(div_data), use_container_width=True)
                        st.caption("Note: Historical average dividend yield (2019-2024). Past performance does not guarantee future dividends.")
                
                st.markdown("#### Success Probability Gauge")
                
                gauge_chart = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=success_rate,
                    title={"text": "Probability of Reaching Target"},
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": COLORS["primary"]},
                        "steps": [
                            {"range": [0, 33], "color": COLORS["danger"]},
                            {"range": [33, 66], "color": COLORS["warning"]},
                            {"range": [66, 100], "color": COLORS["success"]}
                        ],
                        "threshold": {"value": 70, "line": {"color": COLORS["dark"], "width": 2}}
                    }
                ))
                gauge_chart.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(gauge_chart, use_container_width=True)
                
                st.markdown("#### Recommendation")
                
            if success_rate >= 80:
                    st.success("Your plan is well-structured and on track. Maintain consistent monthly contributions and review semi-annually.")
            elif success_rate >= 60:
                    st.warning("Your plan would benefit from minor adjustments. Consider increasing monthly contributions or extending your investment horizon.")
            else:
                    st.error("Your plan requires significant revision. Consider adjusting your target amount, increasing contributions, or extending your timeline before proceeding.")
                
            if st.button("Start Over", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

# ============================================================================
# TAB 2: STOCK DATA EXPLORER
# ============================================================================
with tab_explorer:
    st.markdown("### Stock Data Explorer")
    
    explorer_col1, explorer_col2 = st.columns([1, 2])
    
    with explorer_col1:
        target_stock = st.selectbox("Select Stock", EMITEN, key="explorer_stock")
        date_start = st.date_input("Start Date", value=date(2020, 1, 1), key="exp_start")
        date_end = st.date_input("End Date", value=date(2024, 12, 31), key="exp_end")
        
        data_columns = ["open_price", "high", "low", "close", "volume", "value", "foreign_buy", "foreign_sell", "weight_for_index"]
        selected_columns = st.multiselect("Display Columns", data_columns, default=["open_price", "high", "low", "close"])
    
    with explorer_col2:
        stock_df = STOCK_DATA[target_stock].copy()
        stock_df = stock_df.reset_index()
        stock_df.rename(columns={"index": "date"}, inplace=True)
        stock_df["date"] = pd.to_datetime(stock_df["date"])
        
        date_mask = (stock_df["date"] >= pd.to_datetime(date_start)) & (stock_df["date"] <= pd.to_datetime(date_end))
        filtered_data = stock_df[date_mask]
        
        if len(filtered_data) > 0 and selected_columns:
            display_data = filtered_data[["date"] + selected_columns].tail(20).sort_values("date", ascending=False)
            
            for col in selected_columns:
                if col in ["open_price", "high", "low", "close"]:
                    display_data[col] = display_data[col].apply(lambda x: format_idr(x))
                elif col in ["volume", "value", "foreign_buy", "foreign_sell"]:
                    display_data[col] = display_data[col].apply(lambda x: format_number(x))
            
            st.dataframe(display_data, use_container_width=True)
    
    st.divider()
    
    st.markdown("#### Price Chart")
    
    candle_data = STOCK_DATA[target_stock].copy()
    candle_data = candle_data.reset_index()
    candle_data.rename(columns={"index": "date"}, inplace=True)
    candle_data["date"] = pd.to_datetime(candle_data["date"])
    candle_mask = (candle_data["date"] >= pd.to_datetime(date_start)) & (candle_data["date"] <= pd.to_datetime(date_end))
    candle_filtered = candle_data[candle_mask]
    
    if len(candle_filtered) > 0:
        candlestick = go.Figure(data=[go.Candlestick(
            x=candle_filtered["date"],
            open=candle_filtered["open_price"],
            high=candle_filtered["high"],
            low=candle_filtered["low"],
            close=candle_filtered["close"],
            name=target_stock,
            increasing_line_color=COLORS["success"],
            decreasing_line_color=COLORS["danger"]
        )])
        candlestick.update_layout(
            title=f"{target_stock} - Price History",
            title_x=0.5,
            yaxis_title="Price (Rupiah)",
            height=450
        )
        st.plotly_chart(candlestick, use_container_width=True)
 
 # Foreign Flow Analysis (replaces Price Checker)
    st.markdown("#### Foreign Flow Analysis")
    
    if 'foreign_buy' in STOCK_DATA[target_stock].columns:
        ff_stock = target_stock
        
        ff_data = STOCK_DATA[ff_stock].copy()
        ff_data = ff_data.reset_index()
        ff_data.rename(columns={"index": "date"}, inplace=True)
        ff_data["date"] = pd.to_datetime(ff_data["date"])
        ff_data["net_foreign"] = ff_data["foreign_buy"] - ff_data["foreign_sell"]
        ff_data["cumulative_foreign"] = ff_data["net_foreign"].cumsum()
        
        ff_mask = (ff_data["date"] >= pd.to_datetime(date_start)) & (ff_data["date"] <= pd.to_datetime(date_end))
        ff_filtered = ff_data[ff_mask]
        
        if len(ff_filtered) > 0:
            fig = go.Figure()
            
            colors = [COLORS["success"] if x > 0 else COLORS["danger"] for x in ff_filtered["net_foreign"]]
            fig.add_trace(go.Bar(
                x=ff_filtered["date"], 
                y=ff_filtered["net_foreign"], 
                name="Net Foreign Flow", 
                marker_color=colors,
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=ff_filtered["date"], 
                y=ff_filtered["cumulative_foreign"], 
                name="Cumulative Flow", 
                line=dict(color=COLORS["primary"], width=2),
                yaxis="y2"
            ))
            
            fig.update_layout(
                title=f"{ff_stock} - Foreign Investor Flow",
                xaxis_title="Date",
                height=400,
                yaxis=dict(title="Net Flow (Rp)", side="left"),
                yaxis2=dict(title="Cumulative (Rp)", overlaying="y", side="right"),
                template="simple_white"
            )
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
            
            total_net = ff_filtered["net_foreign"].sum()
            if total_net > 0:
                st.success(f"Net foreign inflow: {format_idr(total_net)} over period")
            else:
                st.warning(f"Net foreign outflow: {format_idr(abs(total_net))} over period")
        else:
            st.info("No foreign flow data available for selected date range")
    else:
        st.info("Foreign flow data not available for this stock")
# ============================================================================
# TAB 3: Statistic ANALYSIS
# ============================================================================
with tab_analysis:
    st.markdown("### Statistic Analysis")
    
    analysis_tabs = st.tabs(["Return Distribution", "Volatility", "Drawdown", "Correlation", "Value at Risk"])
    
    with analysis_tabs[0]:
        selected_stock = st.selectbox("Select Stock", EMITEN, key="return_stock")
        
        return_data = STOCK_DATA[selected_stock].copy()
        return_data = calculate_returns(return_data)
        daily_returns = return_data["daily_return"].dropna()
        
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric("Mean Daily Return", f"{daily_returns.mean():.4f}%")
            st.metric("Standard Deviation", f"{daily_returns.std():.4f}%")
        with stat_col2:
            st.metric("Skewness", f"{daily_returns.skew():.3f}")
            st.metric("Kurtosis", f"{daily_returns.kurtosis():.3f}")
        
        hist_chart = px.histogram(x=daily_returns, nbins=50, title=f"Daily Return Distribution: {selected_stock}",
                                  color_discrete_sequence=[COLORS["primary"]])
        hist_chart.update_layout(title_x=0.5, height=500, xaxis_title="Daily Return (%)", yaxis_title="Frequency")
        st.plotly_chart(hist_chart, use_container_width=True)
    
    with analysis_tabs[1]:
        selected_stock = st.selectbox("Select Stock", EMITEN, key="vol_stock")
        
        vol_data = STOCK_DATA[selected_stock].copy()
        vol_data = calculate_returns(vol_data)
        vol_data = calculate_volatility(vol_data, window=20)
        
        vol_chart = go.Figure()
        vol_chart.add_trace(go.Scatter(x=vol_data.index, y=vol_data["volatility_20d"],
                                      mode="lines", name="20-Day Volatility",
                                      line=dict(color=COLORS["primary"], width=2)))
        vol_chart.add_hline(y=1.5, line_dash="dash", line_color=COLORS["success"], annotation_text="Low Risk Threshold")
        vol_chart.add_hline(y=3.0, line_dash="dash", line_color=COLORS["danger"], annotation_text="High Risk Threshold")
        vol_chart.add_vrect(x0="2020-03-01", x1="2020-06-30", fillcolor=COLORS["danger"], opacity=0.08, layer="below")
        vol_chart.update_layout(title=f"Rolling Volatility: {selected_stock}", title_x=0.5, height=500)
        st.plotly_chart(vol_chart, use_container_width=True)
    
    with analysis_tabs[2]:
        selected_stock = st.selectbox("Select Stock", EMITEN, key="dd_stock")
        
        dd_data = STOCK_DATA[selected_stock].copy()
        dd_data = calculate_drawdown(dd_data)
        
        dd_chart = go.Figure()
        dd_chart.add_trace(go.Scatter(x=dd_data.index, y=dd_data["drawdown"], mode="lines",
                                     name="Drawdown", fill="tozeroy",
                                     line=dict(color=COLORS["danger"], width=1.5)))
        dd_chart.add_hline(y=-20, line_dash="dash", line_color=COLORS["warning"], annotation_text="Bear Market Threshold")
        dd_chart.add_vrect(x0="2020-03-01", x1="2020-06-30", fillcolor=COLORS["danger"], opacity=0.08, layer="below")
        dd_chart.update_layout(title=f"Drawdown: {selected_stock}", title_x=0.5,
                              yaxis_title="Drawdown (%)", height=500)
        st.plotly_chart(dd_chart, use_container_width=True)
        
        max_dd = dd_data["drawdown"].min()
        current_dd = dd_data["drawdown"].iloc[-1]
        
        dd_metric_col1, dd_metric_col2 = st.columns(2)
        with dd_metric_col1:
            st.metric("Maximum Drawdown", f"{max_dd:.1f}%")
        with dd_metric_col2:
            st.metric("Current Drawdown", f"{current_dd:.1f}%")
    
    with analysis_tabs[3]:
        default_correlation_stocks = ["BBCA", "BBRI", "TLKM"]
        valid_defaults = [s for s in default_correlation_stocks if s in EMITEN]
        correlation_stocks = st.multiselect("Select Stocks (minimum 2)", EMITEN, default=valid_defaults)
        
        if len(correlation_stocks) >= 2:
            correlation_matrix_data = pd.DataFrame()
            for stock in correlation_stocks:
                corr_data = STOCK_DATA[stock].copy()
                corr_data = calculate_returns(corr_data)
                correlation_matrix_data[stock] = corr_data["daily_return"]
            
            correlation_matrix = correlation_matrix_data.corr()
            corr_chart = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale="RdBu_r",
                                   zmin=-1, zmax=1, title="Stock Return Correlation Matrix")
            corr_chart.update_layout(title_x=0.5, height=600)
            st.plotly_chart(corr_chart, use_container_width=True)
        else:
            st.info("Please select at least 2 stocks to display correlation matrix")
    
    with analysis_tabs[4]:
        selected_stock = st.selectbox("Select Stock", EMITEN, key="var_stock")
        
        var_data = STOCK_DATA[selected_stock].copy()
        var_data = calculate_returns(var_data)
        daily_returns_var = var_data["daily_return"].dropna()
        
        var_95 = daily_returns_var.quantile(0.05)
        cvar_95 = daily_returns_var[daily_returns_var <= var_95].mean()
        
        var_col1, var_col2 = st.columns(2)
        with var_col1:
            st.metric("Value at Risk (VaR 95%)", f"{var_95:.2f}%",
                     help="5 percent probability of losing more than this on any trading day")
        with var_col2:
            st.metric("Conditional VaR (CVaR 95%)", f"{cvar_95:.2f}%",
                     help="Average loss when VaR threshold is exceeded")
        
        st.caption(f"VaR to CVaR Gap: {abs(cvar_95 - var_95):.2f} percentage points - indicates tail risk severity")
        
        var_chart = go.Figure()
        var_chart.add_trace(go.Histogram(x=daily_returns_var, nbinsx=50, name="Returns",
                                        marker_color=COLORS["primary"], opacity=0.7))
        var_chart.add_vline(x=var_95, line_dash="dash", line_color=COLORS["danger"],
                           annotation_text=f"VaR 95%: {var_95:.2f}%")
        var_chart.add_vline(x=cvar_95, line_dash="dash", line_color=COLORS["warning"],
                           annotation_text=f"CVaR: {cvar_95:.2f}%")
        var_chart.update_layout(title=f"Return Distribution with Risk Metrics: {selected_stock}",
                               title_x=0.5, xaxis_title="Daily Return (%)", yaxis_title="Frequency",
                               height=500)
        st.plotly_chart(var_chart, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown(f"""
<div class="footer">
    <strong>LQ45 Financial Planning Engine by Kinaya Rafa</strong><br>
    Data Period: July 2019 - February 2025 | 40 Constituent Stocks | 1.355 Trading Days<br>
    Methodology: Crisis-Weighted Bootstrap Simulation (10.000 Paths)<br>
    <span>Data compiled by wildangunawan/Dataset-Saham-IDX (GitHub) | IDX historical data</span><br>
    <span>Disclaimer: Educational purposes only.</span>
</div>
""", unsafe_allow_html=True)

print("Application initialized successfully")