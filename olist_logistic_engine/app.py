"""
Olist Logistics Network Intelligence Engine
Main Dashboard - Professional Enterprise Grade
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_all, check_data_availability

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Olist Logistics Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD CUSTOM CSS
# ============================================
def load_css(css_file: str):
    """Load custom CSS file"""
    css_path = Path(__file__).parent / "assets" / css_file
    if css_path.exists():
        with open(css_path, "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def load_js(js_file: str):
    """Load custom JavaScript file"""
    js_path = Path(__file__).parent / "assets" / js_file
    if js_path.exists():
        with open(js_path, "r") as f:
            js = f.read()
        st.markdown(f"<script>{js}</script>", unsafe_allow_html=True)

load_css("style.css")
load_js("custom.js")

# ============================================
# HEADER
# ============================================
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown('<div class="logo-placeholder"></div>', unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="main-header">Olist Logistics Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enterprise logistics network analytics | Route optimization | Warehouse planning</div>', unsafe_allow_html=True)

# ============================================
# DATA LOADING
# ============================================
@st.cache_resource
def load_all_data():
    """Load all data with caching"""
    try:
        data = load_all()
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

with st.spinner("Loading logistics data..."):
    data = load_all_data()

if data is None or data.get('network') is None or data['network'].empty:
    st.error("Failed to load data. Please check that all parquet files exist in the data/ directory.")
    st.info("""
    **Required files in /data folder:**
    - network_data.parquet
    - state_centroids.parquet
    - warehouse_candidates.parquet (optional)
    - route_features.parquet (optional)
    """)
    st.stop()

# ============================================
# SIDEBAR NAVIGATION
# ============================================
with st.sidebar:
    st.markdown('<div class="sidebar-header">NAVIGATION</div>', unsafe_allow_html=True)
    
    # Page selection
    page = st.radio(
        "",
        ["Map View", "Route Analytics", "Warehouse Optimization", "Performance Reports"],
        label_visibility="collapsed",
        format_func=lambda x: f" {x}"
    )
    
    st.markdown("---")
    st.markdown('<div class="sidebar-header">QUICK STATS</div>', unsafe_allow_html=True)
    
    # Quick stats from data
    df_network = data['network']
    total_routes = len(df_network)
    total_orders = df_network['order_count'].sum()
    avg_delivery = df_network['avg_delivery_days'].mean() if 'avg_delivery_days' in df_network.columns else 0
    
    st.metric("Total Routes", f"{total_routes:,}")
    st.metric("Total Orders", f"{total_orders:,}")
    st.metric("Avg Delivery", f"{avg_delivery:.1f} days")
    
    st.markdown("---")
    st.caption("© 2026 Olist Logistics Intelligence")
    st.caption("Data: Brazilian E-commerce (2016-2018)")

# ============================================
# MAIN CONTENT - ROUTING
# ============================================
if page == "Map View":
    from pages.map_view import render_map_view
    render_map_view()
elif page == "Route Analytics":
    from pages.analytics import render_analytics
    render_analytics()
elif page == "Warehouse Optimization":
    from pages.warehouse_optimization import render_warehouse_optimization
    render_warehouse_optimization()
elif page == "Performance Reports":
    from pages.reports import render_reports
    render_reports()