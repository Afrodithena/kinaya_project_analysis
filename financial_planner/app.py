"""
Financial Planning Engine - Main Dashboard
Professional financial planning tool based on LQ45 historical data (2019-2025)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from src.data.loader import load_all_stocks
from src.features.returns import calculate_returns
from src.simulation.bootstrap import BootstrapSimulator, add_crisis_weights
from src.utils.constants import SECTOR_MAP
from src.recommendation.engine import (
    build_stock_features_dataframe,
    normalize_features,
    generate_recommendation
)
from src.advisory.portfolio_manager import PortfolioManager
from src.advisory.lq45_overview import LQ45_OVERVIEW

st.set_page_config(page_title="Financial Planning Engine", page_icon="", layout="wide")


def prepare_bootstrap_data(df, use_crisis_weight=True):
    """Prepare returns and weights for bootstrap simulation."""
    df = df.copy()
    df = calculate_returns(df)
    
    if use_crisis_weight:
        df = add_crisis_weights(df)
        df_clean = df[df["daily_return"].notna()]
        returns = df_clean["daily_return"].values
        weights = df_clean["bootstrap_weight"].values
        return returns, weights
    else:
        returns = df["daily_return"].dropna().values
        return returns, None


@st.cache_data
def load_data():
    return load_all_stocks()


emiten_clean, all_stocks_data = load_data()


@st.cache_data
def get_features_data():
    """Build features dataframe for recommendation engine (global scope)."""
    df_features = build_stock_features_dataframe(all_stocks_data)
    df_norm = normalize_features(df_features)
    return df_features, df_norm


# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_age" not in st.session_state:
    st.session_state.user_age = 25
if "has_lq45_experience" not in st.session_state:
    st.session_state.has_lq45_experience = None
if "user_positions" not in st.session_state:
    st.session_state.user_positions = []
if "selected_stocks" not in st.session_state:
    st.session_state.selected_stocks = []
if "stock_allocations" not in st.session_state:
    st.session_state.stock_allocations = {}
if "goal" not in st.session_state:
    st.session_state.goal = None
if "target_amount" not in st.session_state:
    st.session_state.target_amount = 100_000_000
if "years" not in st.session_state:
    st.session_state.years = 2.0
if "monthly_saving" not in st.session_state:
    st.session_state.monthly_saving = 2_000_000
if "risk_profile" not in st.session_state:
    st.session_state.risk_profile = "Moderate"
if "crisis_weight" not in st.session_state:
    st.session_state.crisis_weight = True


# ============================================
# STEP 1: WELCOME SCREEN + LQ45 EXPERIENCE
# ============================================
if st.session_state.step == 1:
    st.markdown("### Welcome to Your Financial Planning Journey")
    st.markdown("This engine analyzes LQ45 stock market data from July 2019 to February 2025 to help you make informed financial decisions for your life goals. Unlike traditional calculators that assume fixed returns, this tool uses historical market behavior, including the 2020 COVID-19 crisis, to provide realistic projections.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        user_name = st.text_input("Your Name / Initial", value=st.session_state.user_name, key="name_input")
        st.session_state.user_name = user_name
    with col2:
        user_age = st.number_input("Your Age", min_value=18, max_value=60, value=st.session_state.user_age, key="age_input")
        st.session_state.user_age = user_age
    with col3:
        st.date_input("Today's Date", value=date.today(), key="today_date")
    
    st.markdown("---")
    st.markdown("### Before We Begin")
    st.markdown(LQ45_OVERVIEW)
    
    experience = st.radio(
        "Have you ever owned LQ45 stocks before?",
        ["No, this is my first time investing in LQ45", 
         "Yes, I currently own LQ45 stocks", 
         "Yes, but I have sold them previously"],
        key="experience_radio"
    )
    
    st.markdown(f"*Based on {len(emiten_clean)} LQ45 stocks | Period: 2019-2025 | 10,000 Monte Carlo simulation paths*")
    
    if "today_date" in st.session_state and st.session_state.today_date:
        try:
            st.caption(f"Today: {st.session_state.today_date.strftime('%B %d, %Y')}")
        except AttributeError:
            st.caption(f"Today: {st.session_state.today_date}")
    
    if st.button("Start Planning", key="start_btn"):
        if not st.session_state.user_name:
            st.warning("Please enter your name to continue.")
        else:
            st.session_state.has_lq45_experience = experience
            
            if experience == "Yes, I currently own LQ45 stocks":
                st.session_state.step = 5
            else:
                st.session_state.step = 2
            st.rerun()


# ============================================
# STEP 2: SELECT STOCKS (for first-time investors)
# ============================================
elif st.session_state.step == 2:
    st.markdown(f"### Hello {st.session_state.user_name}. Select Your Portfolio Stocks")
    st.markdown("Choose 2 to 8 stocks for your portfolio. You can adjust the allocation percentage for each stock. The total allocation must sum to 100%.")
    
    if "stock_lots" not in st.session_state:
        st.session_state.stock_lots = {}
    if "stock_purchase_dates" not in st.session_state:
        st.session_state.stock_purchase_dates = {}

    stocks_by_sector = {}
    for stock in emiten_clean:
        sector = SECTOR_MAP.get(stock, "Other")
        if sector not in stocks_by_sector:
            stocks_by_sector[sector] = []
        stocks_by_sector[sector].append(stock)
    
    cols = st.columns(3)
    sector_list = list(stocks_by_sector.keys())
    
    for idx, sector in enumerate(sector_list):
        col_idx = idx % 3
        with cols[col_idx]:
            st.markdown(f"**{sector}**")
            for stock in stocks_by_sector[sector]:
                is_checked = stock in st.session_state.selected_stocks
                if st.checkbox(stock, value=is_checked, key=f"stock_{stock}"):
                    if stock not in st.session_state.selected_stocks:
                        st.session_state.selected_stocks.append(stock)
                else:
                    if stock in st.session_state.selected_stocks:
                        st.session_state.selected_stocks.remove(stock)

                        if stock in st.session_state.stock_lots:
                            del st.session_state.stock_lots[stocks]
                        if stock in st.session_state.stock_purchase_dates:
                            del st.session_state.stock_purchase_dates[stocks]
    
    if len(st.session_state.selected_stocks) > 0:
        st.markdown("---")
        st.markdown("### Set Your Investment Details")
        st.markdown("For each selected stock, specify the number of lots and purchase date")
        st.caption("Note: 1 lot = 100 shares")

        
        for i, stock in enumerate(st.session_state.selected_stocks):
            with st.container():
                st.markdown(f"**{stock}**")

                col_lot, col_date, col_alloc = st.columns([2,3,2])

                with col_lot:
                    current_lot = st.session_state.stock_lots.get(stock, 1)
                    lot_size = st.number_input(
                        "Lot Size",
                        min_value=1,
                        max_value=1000,
                        value=current_lot,
                        step=1,
                        key=f"lot_{stock}"
                    )
                    st.session_state.stock_lots[stock] = lot_size
                    shares = lot_size * 100
                    st.caption(f"Shares: {shares:,}")
                
                with col_date:
                    current_date = st.session_state.stock_purchase_dates.get(stock, date.today())
                    purchase_date = st.date_input(
                        "Purchase Date",
                        value=current_date,
                        key=f"date_{stock}"
                    )
                    st.session_state.stock_purchase_dates[stock] = purchase_date

                    try:
                        df_stock = all_stocks_data[stock]
                        closest_price = df_stock[df_stock.index <= pd.to_datetime(purchase_date)]['close'].iloc[-1] if len(df_stock[df_stock.index <= pd.to_datetime(purchase_date)]) > 0 else df_stock['close'].iloc[0]
                        st.caption(f"Est. Price: Rp {closest_price:,.0f}")
                        st.session_state.stock_estimated_prices = st.session_state.get("stock_estimated_prices", {})
                        st.session_state.stock_estimated_prices[stock] = closest_price
                    except:
                        st.caption("Price not available")
                
                with col_alloc:
                    current_alloc = st.session_state.stock_allocations.get(stock, 0)
                    allocation = st.number_input(
                        "Allocation (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(current_alloc),
                        step=5.0,
                        key=f"alloc_{stock}"
                    )
                    st.session_state.stock_allocations[stock] = allocation
                
                st.markdown("---")
        
        # Hitung total alokasi
        total_alloc = sum(st.session_state.stock_allocations.get(s, 0) for s in st.session_state.selected_stocks)
        
        # Hitung total investasi berdasarkan lot dan harga estimasi
        total_investment = 0
        for stock in st.session_state.selected_stocks:
            lot = st.session_state.stock_lots.get(stock, 0)
            estimated_price = st.session_state.stock_estimated_prices.get(stock, 0)
            total_investment += lot * 100 * estimated_price
        
        st.markdown(f"**Total Allocation: {total_alloc:.1f}%**")
        if total_investment > 0:
            st.markdown(f"**Estimated Total Investment:** Rp {total_investment:,.0f}")
        
        if total_alloc != 100:
            st.warning(f"Total allocation must be 100%. Current total: {total_alloc:.1f}%")
        else:
            st.success("Allocation balanced!")
    
    else:
        st.info("Please select at least one stock to continue.")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("← Back to Welcome", key="back_to_welcome_step2"):
            st.session_state.step = 1
            st.rerun()
    with col_btn2:
        if st.button("Clear Selection", key="clear_selection"):
            st.session_state.selected_stocks = []
            st.session_state.stock_allocations = {}
            st.session_state.stock_lots = {}
            st.session_state.stock_purchase_dates = {}
            st.session_state.stock_estimated_prices = {}
            st.rerun()
    with col_btn3:
        if st.button("Next: Define Goal", key="next_goal_btn_fixed"):
            if total_alloc == 100:
                st.session_state.step = 3
                st.rerun()
            else:
                st.error(f"Total allocation must be 100%. Current total: {total_alloc:.1f}%")

# ============================================
# STEP 3: SET GOAL & RISK PROFILE
# ============================================
elif st.session_state.step == 3:
    st.markdown(f"### Define Your Financial Goal, {st.session_state.user_name}")
    
    df_features, df_norm = get_features_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        goal = st.selectbox(
            "Select your financial objective",
            ["Wedding Fund (1-3 years)", "KPR Down Payment (3-5 years)", "Child Education (10-18 years)"],
            key="goal_select"
        )
        st.session_state.goal = goal
        
        target = st.number_input(
            "Target amount (Indonesian Rupiah)",
            min_value=10_000_000,
            value=100_000_000,
            step=10_000_000,
            key="target_input"
        )
        st.session_state.target_amount = target
        
        if "Wedding" in goal:
            years = st.slider("Investment time horizon (years)", 1.0, 3.0, 2.0, 0.5, key="years_wedding")
        elif "KPR" in goal:
            years = st.slider("Years to save for down payment", 3.0, 5.0, 4.0, 0.5, key="years_kpr")
        else:
            years = st.slider("Years until college enrollment", 10, 18, 12, 1, key="years_edu")
        st.session_state.years = years
        
        monthly = st.number_input(
            "Your monthly saving capacity (Rupiah)",
            min_value=500_000,
            value=2_000_000,
            step=500_000,
            key="monthly_input"
        )
        st.session_state.monthly_saving = monthly
    
    with col2:
        st.markdown("### Investment Profile")
        risk_profile = st.selectbox(
            "Select your risk tolerance",
            ["Conservative", "Moderate", "Aggressive"],
            key="risk_profile_select"
        )
        st.session_state.risk_profile = risk_profile
        
        st.markdown("---")
        st.markdown("### Simulation Parameters")
        crisis_weight = st.checkbox("Include COVID-19 crisis weighting", value=True, key="crisis_setting")
        st.session_state.crisis_weight = crisis_weight
        st.caption("When enabled, the simulation gives higher weight to the March-June 2020 period, providing a more conservative estimate.")
        
        st.markdown("---")
        
        if st.button("Generate Portfolio Recommendation", key="suggest_btn"):
            with st.spinner("Analyzing 40 LQ45 stocks based on your profile..."):
                recommendation = generate_recommendation(
                    goal=st.session_state.goal,
                    risk_profile=st.session_state.risk_profile,
                    df_features=df_features,
                    df_norm=df_norm,
                    top_n=5
                )
                
                st.session_state.selected_stocks = recommendation["recommended_stocks"]
                st.session_state.stock_allocations = recommendation["allocations"]
                
                st.success(f"Portfolio generated with {len(recommendation['recommended_stocks'])} stocks")
                
                with st.expander("View Recommendation Rationale", expanded=True):
                    st.markdown(recommendation["strategy"])
                
                st.info(recommendation["explanation"])
                
                st.markdown("### Recommended Stocks")
                for stock, score in recommendation["scores"].items():
                    risk_label = recommendation["risk_levels"].get(stock, "Medium Risk")
                    st.markdown(f"- **{stock}** (Score: {score:.3f}, {risk_label})")
                
                st.markdown("### Suggested Allocation")
                allocation_text = ", ".join([f"{stock}: {pct}%" for stock, pct in recommendation["allocations"].items()])
                st.info(f"Allocation: {allocation_text}")
                
                if st.button("Apply This Portfolio", key="use_suggested_btn"):
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### Current Portfolio Selection")
        for stock in st.session_state.selected_stocks:
            pct = st.session_state.stock_allocations.get(stock, 0)
            if pct > 0:
                st.markdown(f"- {stock}: {pct}%")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.session_state.get("is_existing_investor",False):
            if st.button("← Back to Portfolio Review", key="back_to_portfolio_from_goal"):
                st.session_state.step = 5
                st.rerun()
        else:
            if st.button("← Back to Stock Selection", key="back_to_stocks_btn"):
                st.session_state.step = 2
                st.rerun()
    with col_btn2:
        if st.button("Reset to Welcome", key="reset_to_welcome"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with col_btn3:
        if st.button("Calculate My Plan", key="calculate_btn_fixed"):
            st.session_state.risk_profile = risk_profile
            st.session_state.crisis_weight = crisis_weight
            st.session_state.step = 4
            st.rerun()

# ============================================
# STEP 4: RESULT
# ============================================
elif st.session_state.step == 4:
    st.markdown(f"### Your Financial Plan, {st.session_state.user_name}")
    
    if st.session_state.get("is_existing_investor", False):
        selected_stocks = st.session_state.get("existing_portfolio_stocks", [])
        weights = [1/len(selected_stocks)] * len(selected_stocks) if selected_stocks else []
    else:
        selected_stocks = [s for s in st.session_state.selected_stocks 
                          if st.session_state.stock_allocations.get(s, 0) > 0]
        weights = [st.session_state.stock_allocations.get(s, 0) / 100 for s in selected_stocks]
    
    if not selected_stocks:
        st.error("No stocks selected in your portfolio. Please go back and selected at least one stock.")
        if st.button("← Back to Stock Selection"):
            st.session_state.step = 2
            st.rerun()
        st.stop()

    proxy_stock = selected_stocks[0]
    
    df_proxy = all_stocks_data[proxy_stock].copy()
    returns, weights_bs = prepare_bootstrap_data(df_proxy, st.session_state.crisis_weight)
    bootstrap = BootstrapSimulator(returns=returns, weights=weights_bs)
    
    risk_return = {"Conservative": 0.08, "Moderate": 0.10, "Aggressive": 0.12}
    expected_return = risk_return.get(st.session_state.risk_profile, 0.10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Your Input Summary")
        st.markdown(f"- **Financial Goal:** {st.session_state.goal}")
        st.markdown(f"- **Target Amount:** Rp {st.session_state.target_amount:,.0f}")
        st.markdown(f"- **Investment Period:** {st.session_state.years} years")
        st.markdown(f"- **Monthly Contribution:** Rp {st.session_state.monthly_saving:,.0f}")
        st.markdown(f"- **Risk Profile:** {st.session_state.risk_profile}")
        st.markdown(f"- **Crisis Scenario:** {'Included' if st.session_state.crisis_weight else 'Excluded'}")
        
        st.markdown("### Your Portfolio Composition")
        for stock, pct in zip(selected_stocks, weights):
            if pct > 0:
                st.markdown(f"- {stock}: {pct*100:.0f}%")
    
    with col2:
        months = st.session_state.years * 12
        monthly_rate = expected_return / 12
        if monthly_rate > 0:
            required_saving = st.session_state.target_amount * monthly_rate / ((1 + monthly_rate) ** months - 1)
        else:
            required_saving = st.session_state.target_amount / months
        
        initial_investment = st.session_state.monthly_saving * months
        bootstrap_result = bootstrap.simulate_with_target(
            initial_investment, st.session_state.target_amount, st.session_state.years
        )
        
        st.markdown("### Quantitative Outlook")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Required Monthly", f"Rp {required_saving:,.0f}")
        with col_m2:
            st.metric("Success Probability", f"{bootstrap_result.probability_success:.1f}%")
        
        st.progress(bootstrap_result.probability_success / 100)
        
        if st.session_state.monthly_saving >= required_saving:
            st.success(f"Your monthly contribution of Rp {st.session_state.monthly_saving:,.0f} is adequate.")
            surplus = st.session_state.monthly_saving - required_saving
            if surplus > 0:
                st.markdown(f"Monthly surplus: Rp {surplus:,.0f}")
        else:
            shortfall = required_saving - st.session_state.monthly_saving
            st.error(f"Shortfall: Rp {shortfall:,.0f}/month")
            st.markdown("**Recommended Adjustments:**")
            st.markdown(f"1. Increase to Rp {required_saving:,.0f}/month")
            st.markdown(f"2. Extend horizon by 1-2 years")
            st.markdown(f"3. Consider more aggressive risk profile")
        
        st.markdown("### Risk Assessment")
        
        goal_type = st.session_state.goal
        time_horizon = st.session_state.years
        prob = bootstrap_result.probability_success
        monthly_contrib = st.session_state.monthly_saving
        required = required_saving
        target = st.session_state.target_amount
        
        if st.session_state.risk_profile == "Conservative":
            max_drawdown = 15
            risk_desc = "low volatility, prioritizing capital preservation over high returns"
            suitable_horizon = "1-3 years"
        elif st.session_state.risk_profile == "Moderate":
            max_drawdown = 25
            risk_desc = "balanced between growth and stability, accepting moderate fluctuations"
            suitable_horizon = "3-7 years"
        else:
            max_drawdown = 35
            risk_desc = "high growth potential with elevated volatility, suitable for long-term horizons"
            suitable_horizon = "7+ years"
        
        st.markdown(f"""
        **Portfolio Risk Profile: {st.session_state.risk_profile}**
        
        Your selected {st.session_state.risk_profile.lower()} profile means your portfolio is designed with {risk_desc}. 
        Based on LQ45 historical data from July 2019 to February 2025, which includes the COVID-19 crisis where the index dropped approximately 35% from peak to trough, here is how your plan measures up:
        
        | Risk Metric | Your Exposure | Market Context |
        |-------------|---------------|----------------|
        | Expected annual return | {expected_return*100:.0f}% | Based on historical average returns for {st.session_state.risk_profile.lower()} portfolios |
        | Historical maximum drawdown | -{max_drawdown}% | During severe downturns like COVID-19 (March 2020) |
        | Typical recovery period | 6 to 12 months | Based on post-crisis recovery patterns in LQ45 |
        | Volatility expectation | {'Low (5-15% annual)' if max_drawdown <= 15 else 'Moderate (15-25% annual)' if max_drawdown <= 25 else 'High (25-40% annual)'} | Daily price fluctuation expectation |
        
        **Time Horizon Alignment Analysis**
        
        Your investment horizon is {time_horizon:.0f} years. This is {'well-aligned with' if (st.session_state.risk_profile == 'Conservative' and time_horizon <= 3) or (st.session_state.risk_profile == 'Moderate' and 3 <= time_horizon <= 7) or (st.session_state.risk_profile == 'Aggressive' and time_horizon >= 7) else 'longer than typically recommended for' if time_horizon > 7 else 'shorter than typically recommended for'} a {st.session_state.risk_profile.lower()} profile.
        
        {'For a {:.0f}-year horizon, a {:.0f}% allocation to equities is reasonable, but you may want to consider a gradual shift to lower-risk assets as your target date approaches.'.format(time_horizon, 60 if max_drawdown <= 15 else 70 if max_drawdown <= 25 else 85) if time_horizon > 3 else 'For short-term horizons under 3 years, capital preservation should be your priority. Consider keeping 30-50% of your portfolio in cash equivalents 12-18 months before your target date.'}
        """)
        
        # Goal-specific advisory (PANJANG)
        st.markdown("#### Goal-Specific Advisory")
        
        if "Wedding" in goal_type:
            if prob >= 80:
                st.markdown(f"""
                **Wedding Fund Assessment: High Confidence**
                
                Your plan shows a {prob:.0f}% probability of reaching Rp {target:,.0f} within {time_horizon:.0f} years. This is considered a well-structured plan.
                
                **Why this works:** Your monthly contribution of Rp {monthly_contrib:,.0f} is {monthly_contrib - required:,.0f} above the required amount of Rp {required:,.0f}. This surplus, combined with the expected {expected_return*100:.0f}% annual return from your selected portfolio, puts you ahead of schedule.
                
                **What to watch:** Despite the healthy probability, market downturns can temporarily reduce your portfolio value. The COVID-19 crisis in March 2020 caused a 35% drawdown in LQ45, which would have reduced your portfolio value by approximately Rp {initial_investment * 0.35:,.0f} at the lowest point. However, the market fully recovered within 6-12 months.
                
                **Recommendation:** Six months before your wedding date, consider moving 3-6 months of wedding expenses (approximately Rp {target * 0.25:,.0f} to Rp {target * 0.5:,.0f}) to cash equivalents or money market instruments. This protects your essential wedding budget from last-minute market volatility.
                """)
            elif prob >= 60:
                st.markdown(f"""
                **Wedding Fund Assessment: Moderate Confidence**
                
                Your plan has a {prob:.0f}% probability of success. While more than half of historical scenarios succeeded, approximately {100-prob:.0f}% of scenarios fell short of your target.
                
                **Why the gap exists:** Your current monthly contribution of Rp {monthly_contrib:,.0f} is below the required Rp {required:,.0f}. This monthly shortfall of Rp {required - monthly_contrib:,.0f} compounds to approximately Rp {(required - monthly_contrib) * months:,.0f} over {time_horizon:.0f} years before accounting for investment returns.
                
                **What this means for you:** In 4 out of 10 historical market scenarios, you would need to either:
                - Delay your wedding by 6-12 months
                - Reduce your wedding budget by approximately {int((1 - monthly_contrib/required) * 100)}%
                - Or supplement your savings with additional income
                
                **Recommended Actions (prioritized):**
                1. Increase monthly contribution by Rp {required - monthly_contrib:,.0f} (that is {int((required/monthly_contrib - 1)*100)}% more)
                2. Extend your wedding timeline by 6-12 months, which would reduce required monthly saving to approximately Rp {required * (time_horizon/(time_horizon+1)):.0f}
                3. Consider a slightly more aggressive risk profile to target higher returns (note: this increases potential drawdown risk)
                4. Reduce your wedding budget by {int((1 - monthly_contrib/required) * 100)}% to Rp {target * (monthly_contrib/required):,.0f}
                """)
            else:
                st.markdown(f"""
                **Wedding Fund Assessment: Needs Immediate Attention**
                
                Your plan has a {prob:.0f}% probability of success, meaning in more than half of historical market scenarios, you would not reach your target. This requires action before you commit to wedding vendors.
                
                **Primary issue:** The gap between your monthly saving (Rp {monthly_contrib:,.0f}) and the required amount (Rp {required:,.0f}) is substantial. This {required - monthly_contrib:,.0f} monthly shortfall represents a {int((required/monthly_contrib - 1)*100)}% increase needed.
                
                **Impact analysis:** At your current saving rate, you would accumulate Rp {monthly_contrib * months:,.0f} over {time_horizon:.0f} years. With expected returns, this would grow to approximately Rp {monthly_contrib * months * (1 + expected_return/2):,.0f}, leaving a gap of Rp {target - monthly_contrib * months * (1 + expected_return/2):,.0f}.
                
                **Action Items (must implement at least two):**
                1. **Increase monthly contribution** to Rp {required:,.0f} (requires additional Rp {required - monthly_contrib:,.0f}/month)
                2. **Extend wedding timeline** by 1-2 years, which would reduce required monthly saving to approximately Rp {required * (time_horizon/(time_horizon+2)):.0f}
                3. **Consider a more aggressive risk profile** to target {expected_return*100 + 4:.0f}% annual return (current: {expected_return*100:.0f}%)
                4. **Reduce wedding budget** to Rp {int(target * 0.8):,.0f} while keeping current savings rate
                
                **Important:** Do not proceed with current numbers. Adjust your plan using the recommendations above before committing to wedding vendors.
                """)
        
        elif "KPR" in goal_type:
            st.markdown(f"""
            **KPR Down Payment Assessment**
            
            Your plan targets a down payment of {target/1000000:.0f} million Rupiah within {time_horizon:.0f} years. Based on {prob:.0f}% historical probability, here is your position:
            
            **Hidden Costs Reminder (Often Overlooked)**
            
            When budgeting for a house, remember that total funds needed include not just the DP but also:
            - BPHTB (5% of property value): Approximately Rp {target * 0.25:,.0f} (assuming 20% DP)
            - Notary fees (1%): Approximately Rp {target * 0.05:,.0f}
            - Bank provisions (0.5-1%): Additional Rp {target * 0.025:,.0f} to Rp {target * 0.05:,.0f}
            
            **Dividend Synergy Opportunity**
            
            Bank stocks (BBCA, BBRI, BMRI) in your portfolio typically yield 3-4% annually. At your current investment level of approximately Rp {monthly_contrib * 12 * time_horizon:,.0f} total contributions, dividends could generate an additional Rp {monthly_contrib * 12 * time_horizon * 0.035:,.0f} over {time_horizon:.0f} years. This effectively reduces your required monthly saving by approximately Rp {monthly_contrib * 12 * time_horizon * 0.035 / (time_horizon * 12):,.0f} per month.
            
            **Probability Outlook:** {'Your plan is on solid ground. Maintain your current discipline and review semi-annually.' if prob >= 70 else 'Your plan needs strengthening. Focus on the recommended adjustments above before proceeding with property search.'}
            """)
        
        else:  # Education
            st.markdown(f"""
            **Education Fund Assessment (10-18 year horizon)**
            
            Your plan targets university funding of Rp {target/1000000:.0f} million (4-year total) for enrollment in {time_horizon:.0f} years. This long horizon is your greatest asset.
            
            **Why Education Funds Differ from Other Goals**
            
            Education planning requires special consideration of inflation, which averages 8-12% annually for university costs in Indonesia. The {time_horizon:.0f}-year horizon means that what costs Rp {target/4:,.0f} per year today could cost Rp {target/4 * (1.1**time_horizon):,.0f} per year by the time your child enrolls.
            
            **Historical Perspective** 
            
            Long-term investments (10+ years) have historically recovered from all major downturns, including:
            - 2008 Global Financial Crisis: Recovery within 2-3 years
            - 2020 COVID-19 Crisis: Recovery within 6-12 months
            - 2022 Inflation Correction: Recovery within 1 year
            
            **Consumer Stocks as Inflation Hedge**
            
            Your portfolio includes consumer stocks (ICBP, INDF, UNVR). These companies can pass inflation to consumers through price increases, historically preserving purchasing power during periods of high inflation (8-12% annually for education costs).
            
            **Probability Analysis:** With {prob:.0f}% success probability, your plan is {'well-positioned' if prob >= 70 else 'needing adjustment'}. 
            
            **What to expect:** Even with {prob:.0f}% probability, the long horizon means you have flexibility. If markets underperform in early years, you can:
            - Increase contributions gradually as your income grows over {time_horizon:.0f} years
            - Extend the horizon slightly (education timing is flexible by 1-2 years)
            - Rebalance toward slightly higher growth 5-7 years before college
            - Leverage compound growth: Your monthly contributions of Rp {monthly_contrib:,.0f} will grow significantly over {time_horizon:.0f} years.
            """)
        
        # Probability Interpretation (PANJANG)
        st.markdown("#### Understanding Your Probability of Success")
        
        if prob >= 80:
            st.success(f"""
            **{prob:.0f}% - High Confidence**
            
            This percentage comes from analyzing 10,000 historical scenarios spanning July 2019 to February 2025, including the COVID-19 crash and subsequent recovery.
            
            **What this means for you:** Your plan succeeded in {int(prob/10)} out of 10 historical market scenarios. The scenarios that failed typically involved severe market downturns occurring within 12 months of your target date.
            
            **Confidence level:** You can proceed with confidence. Your monthly contribution adequately funds your goal, and your risk profile matches your time horizon.
            
            **Recommendation:** Continue your current plan, review annually, and consider investing any surplus for additional buffer. Six months before your target date, begin shifting to lower-risk assets.
            """)
        elif prob >= 60:
            st.info(f"""
            **{prob:.0f}% - Moderate Confidence**
            
            This percentage comes from analyzing 10,000 historical scenarios spanning July 2019 to February 2025.
            
            **What this means for you:** Your plan succeeded in approximately {int(prob/10)} out of 10 historical market scenarios. The primary risk factor is the timing of potential market downturns relative to your goal deadline.
            
            **Risk factor:** If a market correction similar to COVID-19 occurs within 18 months of your target date, you may need to either delay your goal or accept a smaller outcome (reduced wedding budget, lower down payment, or more affordable university).
            
            **Mitigation strategy:** As you approach your target date (12-18 months out), systematically shift 20-30% of your portfolio to lower volatility assets. This "glide path" approach reduces the impact of last-minute market declines.
            
            **Next Steps:** Implement at least one of the recommended actions above, then re-run the simulation to see improved probability.
            """)
        else:
            st.warning(f"""
            **{prob:.0f}% - Low Confidence (Requires Action)**
            
            This percentage comes from analyzing 10,000 historical scenarios spanning July 2019 to February 2025, including the COVID-19 crisis.
            
            **What this means for you:** In more than half of historical market scenarios, your current plan would not reach your target. This does not mean failure is guaranteed, but it does mean your plan needs adjustment.
            
            **Primary drivers of low probability:**
            1. Monthly saving rate (Rp {monthly_contrib:,.0f}) is below required (Rp {required:,.0f}) - this is the most significant factor
            2. Time horizon ({time_horizon:.0f} years) may be too short for your risk profile
            3. Target amount (Rp {target/1000000:.0f} million) may be ambitious relative to current capacity
            
            **Immediate next steps (prioritized):**
            - **First:** Adjust your monthly contribution. This has the most direct impact on probability.
            - **Second:** Consider extending your timeline. Adding 1-2 years significantly improves probability.
            - **Third:** Review your target amount. Is it realistic given your current income and saving capacity?
            
            **Do not proceed with current numbers** without implementing at least two of the recommended changes above.
            """)
        
        st.caption("Risk analysis is based on LQ45 historical returns (July 2019 - February 2025). Past performance does not guarantee future results. This analysis is for informational purposes and does not constitute financial advice. Consider consulting a licensed financial advisor for personalized guidance.")

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.session_state.get("is_existing_investor", False):
            if st.button("← Back to Portfolio Review", key="back_to_portfolio_btn"):
                st.session_state.step = 5
                st.rerun()
        else:
                if st.button("← Back to Modify Plan", key="back_to_modify_btn"):
                    st.session_state.step = 3
                    st.rerun()
    with col_btn2:
        if st.button("Save My Plan", key="save_plan_btn"):
            # Create plan summary
            import json
            from datetime import datetime
            
            plan_summary = {
                "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_name": st.session_state.user_name,
                "user_age": st.session_state.user_age,
                "goal": st.session_state.goal,
                "target_amount": st.session_state.target_amount,
                "years": st.session_state.years,
                "monthly_saving": st.session_state.monthly_saving,
                "risk_profile": st.session_state.risk_profile,
                "crisis_weight": st.session_state.crisis_weight,
                "selected_stocks": st.session_state.selected_stocks,
                "stock_allocations": st.session_state.stock_allocations,
                "success_probability": bootstrap_result.probability_success,
                "required_monthly": required_saving
            }
            
            # Save as JSON
            with open(f"plan_{st.session_state.user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
                json.dump(plan_summary, f, indent=4, default=str)
            
            st.success(f"Plan saved! File: plan_{st.session_state.user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            st.balloons()
    with col_btn3:
        if st.button("Start Over", key="start_over_btn"):
            # Reset all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
# ============================================
# STEP 5: PORTFOLIO MANAGEMENT (for existing investors)
# ============================================
elif st.session_state.step == 5:
    st.markdown(f"### Portfolio Review, {st.session_state.user_name}")
    
    if "user_positions" not in st.session_state:
        st.session_state.user_positions = []
    
    portfolio_manager = PortfolioManager(all_stocks_data, st.session_state.user_positions)
    
    with st.expander("About LQ45 Index", expanded=False):
        st.markdown(LQ45_OVERVIEW)
    
    st.markdown("### Your Current Stock Positions")
    st.markdown("Enter the stocks you currently own. This information will be used to calculate unrealized gains.")
    
    with st.form(key="add_position_form"):
        col1, col2 = st.columns(2)
        with col1:
            stock = st.selectbox("Stock Ticker", emiten_clean, key="shift_stock")
            purchase_date = st.date_input("Purchase Date", value=date(2022, 1, 1), key="shift_date")
        with col2:
            purchase_price = st.number_input("Purchase Price (Rp)", min_value=0, value=0, step=500, key="shift_price")
            lot_size = st.number_input("Lot Size (1 lot = 100 shares)", min_value=1, value=1, step=1, key="shift_lot")
        
        st.caption("If purchase price is 0, the system will auto-fill from historical data.")
        
        submitted = st.form_submit_button("Add Position")
        if submitted:
            if purchase_price == 0:
                historical_price = portfolio_manager.get_price_on_date(stock, purchase_date)
                if historical_price:
                    purchase_price = historical_price
                    st.info(f"Auto-filled price: Rp {historical_price:,.0f}")
                else:
                    st.error(f"Could not find price for {stock} on {purchase_date}")
                    st.stop()
            
            success = portfolio_manager.add_position(stock, purchase_date, purchase_price, lot_size)
            if success:
                st.success(f"Added {stock}: {lot_size} lot(s)")
                st.rerun()
            else:
                st.error(f"Could not add {stock}.")
    
    if len(portfolio_manager.user_positions) > 0:
        df, total_cost, total_current = portfolio_manager.get_unrealized_gain_loss()
        
        st.markdown("### Current Portfolio Summary")
        st.dataframe(df[['Stock', 'Purchase Date', 'Purchase Price', 'Current Price', 'Lot Size', 'Return %']])
        
        st.markdown(f"**Total Invested:** Rp {total_cost:,.0f}")
        st.markdown(f"**Current Value:** Rp {total_current:,.0f}")
        st.markdown(f"**Unrealized Return:** {((total_current - total_cost) / total_cost * 100):.1f}%")
        
        st.session_state.existing_portfolio_stocks = df['Stock'].tolist()
        st.session_state.is_existing_investor = True
        st.session_state.user_portfolio_value = total_current
        st.session_state.user_portfolio_cost = total_cost

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Clear All Positions", key="clear_positions_btn"):
                st.session_state.user_positions = []
                st.rerun()
        with col_btn2:
            if st.button("Calculate My Plan", key="calculate_from_portfolio"):
                st.session_state.step = 3
                st.rerun()
    else:
        st.info("No positions added yet. Add your stack positions above, or skip to build a new portfolio.")

        col_skip1, col_skip2 = st.columns(2)
        with col_skip1:
            if st.button("Skip - Build New Portfolio", key = "skip_positions"):
                st.session_state.step = 2
                st.rerun()
        with col_skip2:
            if st.button("Back to Welcome", key="back_to_welcom_btn"):
                st.session_state.step = 1
                st.rerun()

# ============================================
# DARK THEME CSS
# ============================================
st.markdown("""
<style>
.stApp { background-color: #0f172a; font-family: 'Inter', sans-serif; }
.main-header { font-size: 2rem; font-weight: 600; color: #ffffff; margin-bottom: 0.5rem; }
.sub-header { font-size: 1rem; color: #94a3b8; margin-bottom: 2rem; }
.stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #ffffff !important; }
.metric-card { background: #1e293b; border-radius: 14px; padding: 1.2rem; box-shadow: 0 2px 6px rgba(0,0,0,0.2); }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #e67e22; }
.metric-label { font-size: 0.8rem; color: #94a3b8; }
.stButton button { background-color: #e67e22; color: white; font-weight: 600; border: none; padding: 0.5rem 2rem; border-radius: 10px; transition: 0.2s; }
.stButton button:hover { background-color: #d35400; transform: translateY(-1px); }
div[data-testid="stSelectbox"] label { color: #ffffff !important; font-weight: 600; }
div[data-baseweb="select"] > div { background-color: #1e293b !important; color: #ffffff !important; border-radius: 10px; border: 1px solid #334155; }
div[data-baseweb="select"] div[role="button"] { color: #ffffff !important; }
div[data-baseweb="popover"] { background-color: #1e293b !important; }
div[data-baseweb="popover"] li { background-color: #1e293b; color: #ffffff !important; }
div[data-baseweb="popover"] li:hover { background-color: #334155 !important; }
div[data-testid="stCheckbox"] label { color: #ffffff !important; }
div[data-testid="stCheckbox"] label span { color: #ffffff !important; }
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input { color: #ffffff !important; background-color: #1e293b !important; border: 1px solid #334155 !important; border-radius: 8px; }
div[data-testid="stSlider"] label { color: #ffffff !important; }
div[data-testid="stRadio"] label { color: #ffffff !important; }
.stTabs [data-baseweb="tab-list"] button p { color: #ffffff !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #e67e22; }
section[data-testid="stSidebar"] { background-color: #0f172a; }
section[data-testid="stSidebar"] p { color: #ffffff !important; }
.stProgress > div > div { background-color: #e67e22; }
.stExpander details summary p { color: #ffffff !important; }
.stExpander { border: 1px solid #334155 !important; border-radius: 8px !important; }
.stAlert { background-color: #1e293b !important; }
.stAlert p { color: #ffffff !important; }
.stInfo { background-color: #1e3a5f !important; }
.stInfo p { color: #ffffff !important; }
div[data-testid="column"] p { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

