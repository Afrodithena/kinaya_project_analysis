"""
Olist Logistics Network Intelligence Engine
Full Feature Dashboard - Enterprise Grade with Map View
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from pathlib import Path
import json

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Olist Logistics Intelligence",
    page_icon=":truck:",
    layout="wide"
)

# ============================================
# CUSTOM CSS (Dengan Box Styles)
# ============================================
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1a1a2e;
        font-weight: 600;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .stButton button {
        background-color: #667eea;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #5a67d8;
    }
    
    /* Performance Metrics Boxes */
    .metric-box {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-box-green {
        border-left: 4px solid #10b981;
    }
    .metric-box-red {
        border-left: 4px solid #ef4444;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: white;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Quick Insight Boxes */
    .insight-box {
        background-color: #0f172a;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        text-align: center;
        border: 1px solid #334155;
    }
    .insight-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f97316;
    }
    .insight-label {
        font-size: 0.65rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING WITH FALLBACK
# ============================================
@st.cache_data
def load_all_data():
    """Load all data from parquet files"""
    
    data = {
        'network': None,
        'route_features': None,
        'seller_features': None,
        'category_features': None,
        'warehouse_candidates': None,
        'route_priority': None,
        'state_centroids': None,
        'cost_benefit': None,
        'simulation': None
    }
    
    data_dir = Path(__file__).parent / "data"
    
    if not data_dir.exists():
        st.error("Data directory not found. Please ensure data files are available.")
        return data
    
    # Load network data
    network_path = data_dir / "network_data.parquet"
    if network_path.exists():
        try:
            data['network'] = pd.read_parquet(network_path)
        except Exception as e:
            st.warning(f"Could not load network_data.parquet: {e}")
    
    # Load route features
    route_path = data_dir / "route_features.parquet"
    if route_path.exists():
        try:
            data['route_features'] = pd.read_parquet(route_path)
        except Exception as e:
            st.warning(f"Could not load route_features.parquet: {e}")
    
    # Load warehouse candidates
    warehouse_path = data_dir / "warehouse_candidates.parquet"
    if warehouse_path.exists():
        try:
            data['warehouse_candidates'] = pd.read_parquet(warehouse_path)
        except Exception as e:
            st.warning(f"Could not load warehouse_candidates.parquet: {e}")
    
    # Load state centroids
    centroids_path = data_dir / "state_centroids.parquet"
    if centroids_path.exists():
        try:
            data['state_centroids'] = pd.read_parquet(centroids_path)
        except Exception as e:
            st.warning(f"Could not load state_centroids.parquet: {e}")
    
    # Load cost benefit summary
    costbenefit_path = data_dir / "cost_benefit_summary.parquet"
    if costbenefit_path.exists():
        try:
            data['cost_benefit'] = pd.read_parquet(costbenefit_path)
        except Exception as e:
            st.warning(f"Could not load cost_benefit_summary.parquet: {e}")
    
    return data

def create_sample_network_data():
    """Create sample data for demonstration when parquet files are missing"""
    np.random.seed(42)
    
    states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'PE', 'CE', 'AM', 'PA', 'GO', 'DF', 'ES']
    performances = ['Fast', 'Normal', 'Slow', 'Critical']
    
    routes = []
    for i in range(200):
        seller = np.random.choice(states)
        customer = np.random.choice(states)
        if seller == customer:
            customer = np.random.choice([s for s in states if s != seller])
        
        perf = np.random.choice(performances, p=[0.25, 0.40, 0.20, 0.15])
        
        if perf == 'Fast':
            days = np.random.uniform(5, 10)
            volume = np.random.randint(200, 1000)
        elif perf == 'Normal':
            days = np.random.uniform(10, 15)
            volume = np.random.randint(100, 600)
        elif perf == 'Slow':
            days = np.random.uniform(15, 20)
            volume = np.random.randint(50, 400)
        else:
            days = np.random.uniform(20, 35)
            volume = np.random.randint(20, 200)
        
        routes.append({
            'seller_state': seller,
            'customer_state': customer,
            'order_count': volume,
            'avg_delivery_days': days,
            'performance': perf,
            'distance_km': np.random.uniform(200, 3500),
            'seller_lat': np.random.uniform(-30, -5),
            'seller_lng': np.random.uniform(-70, -35),
            'customer_lat': np.random.uniform(-30, -5),
            'customer_lng': np.random.uniform(-70, -35)
        })
    
    return pd.DataFrame(routes)

# ============================================
# LOAD DATA
# ============================================
with st.spinner("Loading logistics network data..."):
    data = load_all_data()

# Use sample data if real data not available
if data['network'] is None:
    st.info("Using demonstration data. Upload parquet files to data directory for full functionality.")
    df_network = create_sample_network_data()
    df_route_features = df_network.copy()
else:
    df_network = data['network']
    df_route_features = data['route_features'] if data['route_features'] is not None else df_network

warehouses = data['warehouse_candidates']
state_centroids = data['state_centroids']
cost_benefit = data['cost_benefit']

# Calculate metrics for sidebar
total_routes = len(df_network)
total_orders = df_network['order_count'].sum() if 'order_count' in df_network.columns else 0
avg_delivery = df_network['avg_delivery_days'].mean() if 'avg_delivery_days' in df_network.columns else 0
critical_count = len(df_network[df_network['performance'] == 'Critical']) if 'performance' in df_network.columns else 0
fast_count = len(df_network[df_network['performance'] == 'Fast']) if 'performance' in df_network.columns else 0
unique_sellers = df_network['seller_state'].nunique() if 'seller_state' in df_network.columns else 0
unique_buyers = df_network['customer_state'].nunique() if 'customer_state' in df_network.columns else 0
on_time = len(df_network[df_network['avg_delivery_days'] <= 15]) if 'avg_delivery_days' in df_network.columns else 0
on_time_pct = (on_time / total_routes * 100) if total_routes > 0 else 0

# ============================================
# SIDEBAR NAVIGATION WITH METRICS BOXES
# ============================================
with st.sidebar:
    st.markdown("## Navigation")
    
    page = st.radio(
        "Select Dashboard",
        ["Map View", "Route Analytics", "Warehouse Optimization", "Performance Reports"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("## Performance Metrics")
    
    # Active Routes
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{total_routes:,}</div>
        <div class="metric-label">Active Routes</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Total Orders
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{total_orders:,}</div>
        <div class="metric-label">Total Orders</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Average Delivery
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{avg_delivery:.1f} days</div>
        <div class="metric-label">Avg Delivery</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Critical Routes (Red border)
    st.markdown(f"""
    <div class="metric-box metric-box-red">
        <div class="metric-value">{critical_count}</div>
        <div class="metric-label">Critical Routes</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Fast Routes (Green border)
    st.markdown(f"""
    <div class="metric-box metric-box-green">
        <div class="metric-value">{fast_count}</div>
        <div class="metric-label">Fast Routes</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## Quick Insights")
    
    # Active Seller States
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-value">{unique_sellers}</div>
        <div class="insight-label">Active Seller States</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Active Buyer States
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-value">{unique_buyers}</div>
        <div class="insight-label">Active Buyer States</div>
    </div>
    """, unsafe_allow_html=True)
    
    # On-Time Delivery Rate
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-value">{on_time_pct:.0f}%</div>
        <div class="insight-label">On-Time Delivery Rate</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Data period: 2016-2018")
    st.caption("Powered by Olist")

# ============================================
# MAP VIEW FUNCTION (Built-in PyDeck)
# ============================================
def render_map_view():
    """Render interactive map with PyDeck arc layers"""
    st.markdown("# Network Map")
    st.markdown("Interactive visualization of delivery routes across Brazil")
    
    # Prepare data for map
    map_data = df_network.copy()
    
    # Check if coordinate columns exist
    has_coords = all(col in map_data.columns for col in ['seller_lat', 'seller_lng', 'customer_lat', 'customer_lng'])
    
    if not has_coords:
        st.warning("Coordinate data not available. Showing route summary instead.")
        st.dataframe(map_data[['seller_state', 'customer_state', 'order_count', 'avg_delivery_days', 'performance']].head(20), use_container_width=True)
        return
    
    # Drop rows with missing coordinates
    map_data = map_data.dropna(subset=['seller_lat', 'seller_lng', 'customer_lat', 'customer_lng'])
    
    if map_data.empty:
        st.warning("No routes with valid coordinates found.")
        return
    
    # Limit for performance
    if len(map_data) > 500:
        map_data = map_data.nlargest(500, 'order_count')
        st.caption(f"Showing top 500 routes by order volume (total {len(df_network)} routes available)")
    
    # ============================================
    # PREPARE TOOLTIP HTML (Pre-formatted untuk ROUTES)
    # ============================================
    map_data['tooltip_html'] = map_data.apply(
        lambda row: f"""
        <div style="background: #1e293b; padding: 10px 14px; border-radius: 8px; 
                    border-left: 3px solid #f97316; font-family: monospace;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
            <b style="color: #f97316; font-size: 14px;">{row.get('seller_state', '?')} → {row.get('customer_state', '?')}</b><br>
            <span style="color: #94a3b8;">Orders:</span> <b style="color: white;">{row.get('order_count', 0):,}</b><br>
            <span style="color: #94a3b8;">Delivery:</span> <b style="color: white;">{row.get('avg_delivery_days', 0):.1f} days</b><br>
            <span style="color: #94a3b8;">Performance:</span> <b style="color: #f97316;">{row.get('performance', 'Unknown')}</b>
        </div>
        """,
        axis=1
    )
    
    # Color mapping for performance
    color_map = {
        'Fast': [16, 185, 129, 200],
        'Normal': [245, 158, 11, 210],
        'Slow': [249, 115, 22, 220],
        'Critical': [239, 68, 68, 230]
    }
    
    if 'performance' in map_data.columns:
        map_data['color'] = map_data['performance'].map(lambda x: color_map.get(x, [100, 116, 139, 180]))
    else:
        map_data['color'] = [100, 116, 139, 180]
    
    # Calculate line width based on order volume
    if 'order_count' in map_data.columns and map_data['order_count'].max() > 0:
        max_orders = map_data['order_count'].max()
        map_data['line_width'] = (map_data['order_count'] / max_orders * 8).clip(2, 8)
    else:
        map_data['line_width'] = 3
    
    # Create Arc Layer (for routes)
    arc_layer = pdk.Layer(
        'ArcLayer',
        data=map_data,
        get_source_position=['seller_lng', 'seller_lat'],
        get_target_position=['customer_lng', 'customer_lat'],
        get_source_color='color',
        get_target_color='color',
        get_width='line_width',
        pickable=True,
        auto_highlight=True
    )
    
    # Start layers with arc layer
    layers = [arc_layer]
    
    # Add state centroid layer if available (not pickable)
    if state_centroids is not None and not state_centroids.empty:
        if 'lat' in state_centroids.columns and 'lng' in state_centroids.columns:
            scatter_layer = pdk.Layer(
                'ScatterplotLayer',
                data=state_centroids,
                get_position=['lng', 'lat'],
                get_radius=20000,
                get_fill_color=[100, 116, 139, 80],
                get_line_color=[71, 85, 105, 180],
                pickable=False
            )
            layers.append(scatter_layer)
    
    # ============================================
    # ADD WAREHOUSE LAYER (pickable=False - NO TOOLTIP)
    # ============================================
    if warehouses is not None and not warehouses.empty:
        lat_col = None
        lng_col = None
        for col in warehouses.columns:
            col_lower = col.lower()
            if col_lower in ['lat', 'latitude']:
                lat_col = col
            if col_lower in ['lng', 'lon', 'long', 'longitude']:
                lng_col = col
        
        if lat_col and lng_col:
            warehouse_layer = pdk.Layer(
                'ScatterplotLayer',
                data=warehouses,
                get_position=[lng_col, lat_col],
                get_radius=15000,
                get_fill_color=[249, 115, 22, 200],
                get_line_color=[234, 88, 12, 255],
                pickable=False  # Tidak bisa di-hover, jadi tidak error tooltip
            )
            layers.append(warehouse_layer)
    
    # View state centered on Brazil
    view_state = pdk.ViewState(
        latitude=-14.2350,
        longitude=-51.9253,
        zoom=3.5,
        pitch=40,
        bearing=0
    )
    
    # ============================================
    # TOOLTIP - Hanya untuk Route Layer
    # ============================================
    tooltip = {
        "html": "{tooltip_html}",
        "style": {"backgroundColor": "transparent"}
    }
    
    # Map style (free CartoDB style)
    map_style = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'
    
    # Create deck
    deck = pdk.Deck(
        map_style=map_style,
        initial_view_state=view_state,
        layers=layers,
        tooltip=tooltip
    )
    
    st.pydeck_chart(deck, use_container_width=True)
    
    # Route statistics
    with st.expander("Route Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Routes in Map", f"{len(map_data):,}")
        with col2:
            total_orders_val = map_data['order_count'].sum() if 'order_count' in map_data.columns else 0
            st.metric("Total Orders", f"{total_orders_val:,}")
        with col3:
            avg_delivery_val = map_data['avg_delivery_days'].mean() if 'avg_delivery_days' in map_data.columns else 0
            st.metric("Avg Delivery", f"{avg_delivery_val:.1f} days")

# ============================================
# PAGE: MAP VIEW
# ============================================
if page == "Map View":
    render_map_view()

# ============================================
# PAGE: ROUTE ANALYTICS
# ============================================
elif page == "Route Analytics":
    st.markdown("# Route Analytics")
    st.markdown("Comprehensive analysis of delivery routes, performance metrics, and network trends")
    
    # Key Metrics Row
    st.markdown("### Key Performance Indicators")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    fast_pct = (len(df_network[df_network['performance'] == 'Fast']) / total_routes * 100) if 'performance' in df_network.columns else 0
    critical_pct = (len(df_network[df_network['performance'] == 'Critical']) / total_routes * 100) if 'performance' in df_network.columns else 0
    
    with metric_col1:
        st.metric("Average Delivery Time", f"{avg_delivery:.1f} days", delta=None)
    with metric_col2:
        st.metric("On-Time Performance", f"{100 - critical_pct:.1f}%", delta="Good")
    with metric_col3:
        st.metric("Fast Routes", f"{fast_pct:.1f}%", delta="Target: 30%")
    with metric_col4:
        st.metric("Critical Routes", f"{critical_pct:.1f}%", delta="Needs attention", delta_color="inverse")
    
    # Performance Distribution
    st.markdown("### Route Performance Distribution")
    
    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        if 'performance' in df_network.columns:
            perf_counts = df_network['performance'].value_counts().reset_index()
            perf_counts.columns = ['Performance', 'Count']
            
            color_map = {'Fast': '#2ecc71', 'Normal': '#3498db', 'Slow': '#f39c12', 'Critical': '#e74c3c'}
            
            fig = px.bar(perf_counts, x='Performance', y='Count', title='Route Performance Distribution',
                         color='Performance', color_discrete_map=color_map, text='Count', height=450)
            fig.update_traces(textposition='outside', textfont=dict(size=14, weight='bold'))
            fig.update_layout(showlegend=False, plot_bgcolor='#f8f9fa')
            st.plotly_chart(fig, use_container_width=True)
    
    with col_stats:
        st.markdown("#### Performance Summary")
        for perf in ['Fast', 'Normal', 'Slow', 'Critical']:
            count = len(df_network[df_network['performance'] == perf]) if 'performance' in df_network.columns else 0
            pct = (count / total_routes * 100) if total_routes > 0 else 0
            st.markdown(f"- **{perf}**: {count} routes ({pct:.1f}%)")
        
        st.markdown("---")
        st.markdown("#### Problematic Routes")
        problematic = len(df_network[df_network['performance'].isin(['Slow', 'Critical'])]) if 'performance' in df_network.columns else 0
        st.markdown(f"**{problematic} routes** require intervention")
        st.markdown(f"Affecting approximately **{(problematic / total_routes * 100):.1f}%** of network")
    
    # Highest Volume Routes
    st.markdown("### Highest Volume Routes")
    
    try:
        if 'order_count' in df_network.columns and not df_network.empty:
            top_routes = df_network.nlargest(15, 'order_count')
            
            display_data = {}
            if 'seller_state' in top_routes.columns:
                display_data['Origin'] = top_routes['seller_state']
            if 'customer_state' in top_routes.columns:
                display_data['Destination'] = top_routes['customer_state']
            if 'order_count' in top_routes.columns:
                display_data['Order Count'] = top_routes['order_count']
            if 'avg_delivery_days' in top_routes.columns:
                display_data['Avg Delivery (Days)'] = top_routes['avg_delivery_days'].round(1)
            if 'performance' in top_routes.columns:
                display_data['Performance'] = top_routes['performance']
            
            if display_data:
                top_routes_display = pd.DataFrame(display_data)
                if 'Performance' in top_routes_display.columns:
                    def color_performance(val):
                        colors = {'Fast': '#2ecc71', 'Normal': '#3498db', 'Slow': '#f39c12', 'Critical': '#e74c3c'}
                        return f'background-color: {colors.get(val, "white")}; color: white'
                    st.dataframe(top_routes_display.style.applymap(color_performance, subset=['Performance']), use_container_width=True)
                else:
                    st.dataframe(top_routes_display, use_container_width=True)
            else:
                st.info("Route data available but missing required columns.")
        else:
            st.info("Order count data not available to display top routes.")
    except Exception as e:
        st.warning(f"Could not display top routes: {e}")
        if 'order_count' in df_network.columns:
            simple_top = df_network.nlargest(10, 'order_count')[['seller_state', 'customer_state', 'order_count']].copy()
            simple_top.columns = ['Origin', 'Destination', 'Order Count']
            st.dataframe(simple_top, use_container_width=True)
    
    # Delivery by State
    st.markdown("### Delivery Performance by Origin State")
    
    col_state1, col_state2 = st.columns(2)
    
    with col_state1:
        if 'seller_state' in df_network.columns and 'avg_delivery_days' in df_network.columns:
            state_perf = df_network.groupby('seller_state').agg({'avg_delivery_days': 'mean', 'order_count': 'sum'}).reset_index()
            state_perf = state_perf.sort_values('avg_delivery_days')
            
            fig = px.bar(state_perf.head(10), x='seller_state', y='avg_delivery_days', title='Fastest Origins by Delivery Time',
                         color='avg_delivery_days', color_continuous_scale='Greens', text=state_perf['avg_delivery_days'].head(10).round(1))
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    with col_state2:
        if 'seller_state' in df_network.columns and 'avg_delivery_days' in df_network.columns:
            slowest = df_network.groupby('seller_state')['avg_delivery_days'].mean().reset_index()
            slowest = slowest.sort_values('avg_delivery_days', ascending=False)
            
            fig = px.bar(slowest.head(10), x='seller_state', y='avg_delivery_days', title='Slowest Origins by Delivery Time',
                         color='avg_delivery_days', color_continuous_scale='Reds', text=slowest['avg_delivery_days'].head(10).round(1))
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    # Distance vs Delivery Time
    st.markdown("### Distance vs Delivery Time Analysis")
    
    if 'distance_km' in df_network.columns and 'avg_delivery_days' in df_network.columns:
        fig = px.scatter(df_network, x='distance_km', y='avg_delivery_days', color='performance' if 'performance' in df_network.columns else None,
                         color_discrete_map={'Fast': '#2ecc71', 'Normal': '#3498db', 'Slow': '#f39c12', 'Critical': '#e74c3c'},
                         size='order_count', size_max=15, hover_data=['seller_state', 'customer_state'],
                         title='Relationship Between Distance and Delivery Time',
                         labels={'distance_km': 'Distance (km)', 'avg_delivery_days': 'Average Delivery Time (days)'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE: WAREHOUSE OPTIMIZATION
# ============================================
elif page == "Warehouse Optimization":
    st.markdown("# Warehouse Optimization")
    st.markdown("Strategic analysis for optimal warehouse placement to reduce delivery times")
    
    # Current Network Status
    st.markdown("### Current Network Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Routes", f"{total_routes:,}")
    with col2:
        st.metric("Current Avg Delivery", f"{avg_delivery:.1f} days")
    with col3:
        n_warehouses = len(warehouses) if warehouses is not None and len(warehouses) > 0 else 5
        st.metric("Proposed Warehouses", f"{n_warehouses}")
    with col4:
        st.metric("Estimated Improvement", "-30%", delta="Target reduction")
    
    # Warehouse Candidate Locations
    if warehouses is not None and len(warehouses) > 0:
        st.markdown("### Recommended Warehouse Locations")
        st.dataframe(warehouses, use_container_width=True)
        
        lat_col = None
        lng_col = None
        for col in warehouses.columns:
            col_lower = col.lower()
            if col_lower in ['lat', 'latitude']:
                lat_col = col
            if col_lower in ['lng', 'lon', 'long', 'longitude']:
                lng_col = col
        
        if lat_col and lng_col:
            st.caption(f"Location coordinates: using columns '{lat_col}' and '{lng_col}'")
    else:
        st.info("No warehouse candidate data available. Run DBSCAN clustering first.")
    
    # Priority Routes for Intervention
    st.markdown("### Priority Routes for Warehouse Intervention")
    
    if 'performance' in df_network.columns:
        priority_routes = df_network[df_network['performance'].isin(['Slow', 'Critical'])].nlargest(15, 'order_count')
        
        if len(priority_routes) > 0:
            display_routes = priority_routes[['seller_state', 'customer_state', 'order_count', 'avg_delivery_days', 'performance']]
            display_routes.columns = ['Origin', 'Destination', 'Orders', 'Avg Days', 'Performance']
            st.dataframe(display_routes, use_container_width=True)
            st.markdown(f"**Total problematic orders:** {priority_routes['order_count'].sum():,} orders per year")
        else:
            st.info("No problematic routes detected in current data.")
    
    # Impact Analysis
    st.markdown("### Impact Analysis")
    
    impact_col1, impact_col2 = st.columns(2)
    
    with impact_col1:
        st.markdown("#### Estimated Improvements")
        improvement_data = {
            'Metric': ['Delivery Time Reduction', 'Distance Reduction', 'Problematic Route Reduction', 'Northeast Coverage'],
            'Estimated Improvement': ['30-40%', '50-60%', '40-50%', '85%'],
            'Priority': ['High', 'High', 'Medium', 'High']
        }
        st.dataframe(pd.DataFrame(improvement_data), use_container_width=True)
    
    with impact_col2:
        st.markdown("#### Cost-Benefit Summary")
        if cost_benefit is not None:
            st.dataframe(cost_benefit, use_container_width=True)
        else:
            st.warning("""
            **Cost-Benefit Analysis Summary**
            - 5-Year ROI: -247.6 percent
            - Payback Period: Greater than 5 years
            - Break-even Volume: 294 times current volume
            **Recommendation:** Defer warehouse investment until order volume scales significantly.
            """)
    
    # Route Impact Simulation
    st.markdown("### Route Impact Simulation")
    
    if 'distance_km' in df_network.columns:
        simulation_data = df_network[df_network['performance'].isin(['Slow', 'Critical'])].head(10) if 'performance' in df_network.columns else df_network.head(10)
        
        if len(simulation_data) > 0:
            simulation_data = simulation_data.copy()
            simulation_data['estimated_new_distance'] = simulation_data['distance_km'] * 0.4
            simulation_data['distance_saved'] = simulation_data['distance_km'] - simulation_data['estimated_new_distance']
            
            sim_display = simulation_data[['seller_state', 'customer_state', 'distance_km', 'estimated_new_distance', 'distance_saved']]
            sim_display.columns = ['Origin', 'Destination', 'Current Distance (km)', 'New Distance (km)', 'Distance Saved (km)']
            sim_display = sim_display.round(0)
            
            st.dataframe(sim_display, use_container_width=True)
            total_saved = sim_display['Distance Saved (km)'].sum()
            st.markdown(f"**Total distance saved for top problematic routes:** {total_saved:,.0f} km")

# ============================================
# PAGE: PERFORMANCE REPORTS
# ============================================
elif page == "Performance Reports":
    st.markdown("# Performance Reports")
    st.markdown("Downloadable analytics, performance summaries, and executive reports")
    
    # Summary Statistics
    st.markdown("### Executive Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Network Overview")
        st.markdown(f"- **Total Routes Analyzed:** {total_routes:,}")
        st.markdown(f"- **Total Orders:** {total_orders:,}")
        st.markdown(f"- **Average Delivery Time:** {avg_delivery:.1f} days")
        st.markdown(f"- **States Covered:** {df_network['seller_state'].nunique() if 'seller_state' in df_network.columns else 27}")
    
    with col2:
        st.markdown("#### Performance Distribution")
        if 'performance' in df_network.columns:
            for perf in ['Fast', 'Normal', 'Slow', 'Critical']:
                count = len(df_network[df_network['performance'] == perf])
                pct = (count / total_routes * 100) if total_routes > 0 else 0
                st.markdown(f"- **{perf}:** {count} routes ({pct:.1f}%)")
    
    with col3:
        st.markdown("#### Key Insights")
        st.markdown("- Distance is the primary predictor of delivery delay")
        st.markdown("- Peak season months perform better than average")
        st.markdown("- Northeast region has structural supply gap")
        st.markdown("- 89 percent of orders arrive earlier than estimated")
    
    # Download Reports Section
    st.markdown("### Export Reports")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        csv_data = df_network.to_csv(index=False)
        st.download_button(label="Download Route Analysis (CSV)", data=csv_data, file_name="route_analysis.csv", mime="text/csv")
    
    with export_col2:
        if 'performance' in df_network.columns:
            problematic_routes = df_network[df_network['performance'].isin(['Slow', 'Critical'])]
            csv_problematic = problematic_routes.to_csv(index=False)
            st.download_button(label="Download Problematic Routes (CSV)", data=csv_problematic, file_name="problematic_routes.csv", mime="text/csv")
    
    with export_col3:
        summary_data = {
            'total_routes': int(total_routes),
            'total_orders': int(total_orders),
            'avg_delivery_days': float(avg_delivery),
            'fast_routes': int(len(df_network[df_network['performance'] == 'Fast'])) if 'performance' in df_network.columns else 0,
            'critical_routes': int(len(df_network[df_network['performance'] == 'Critical'])) if 'performance' in df_network.columns else 0,
            'problematic_routes': int(len(df_network[df_network['performance'].isin(['Slow', 'Critical'])]) if 'performance' in df_network.columns else 0)
        }
        json_data = json.dumps(summary_data, indent=2)
        st.download_button(label="Download Summary Report (JSON)", data=json_data, file_name="performance_summary.json", mime="application/json")
    
    # Performance Charts for Report
    st.markdown("### Visual Analytics for Report")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        if 'performance' in df_network.columns:
            perf_counts = df_network['performance'].value_counts()
            fig = px.pie(values=perf_counts.values, names=perf_counts.index, title='Route Performance Distribution',
                         color=perf_counts.index, color_discrete_map={'Fast': '#2ecc71', 'Normal': '#3498db', 'Slow': '#f39c12', 'Critical': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        if 'avg_delivery_days' in df_network.columns:
            fig = px.box(df_network, y='avg_delivery_days', title='Delivery Time Distribution',
                         color='performance' if 'performance' in df_network.columns else None,
                         color_discrete_map={'Fast': '#2ecc71', 'Normal': '#3498db', 'Slow': '#f39c12', 'Critical': '#e74c3c'},
                         labels={'avg_delivery_days': 'Delivery Time (days)'})
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 12px;'>"
    "Olist Logistics Intelligence | Data source: Brazilian E-commerce Dataset (2016-2018) | Powered by Streamlit"
    "</p>",
    unsafe_allow_html=True
)