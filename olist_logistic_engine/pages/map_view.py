"""
Network Map View - Interactive Route Visualization
Using CartoDB Base Map (FREE, no credit card required)
"""

import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import time
from src.data.loader import load_all

def render_map_view():
    """Render the interactive network map page with FREE base map"""
    
    st.markdown('<div class="page-title">Network Map</div>', unsafe_allow_html=True)
    st.markdown("Interactive visualization of delivery routes across Brazil")
    
    # Load data
    @st.cache_data
    def get_data():
        data = load_all()
        return data
    
    data = get_data()
    
    if data is None or data.get('network') is None:
        st.error("Failed to load network data. Please check data files.")
        return
    
    df_network = data['network']
    state_centroids = data.get('state_centroids', pd.DataFrame())
    warehouse_candidates = data.get('warehouse_candidates', None)
    
    # ============================================
    # CHECK REQUIRED COLUMNS
    # ============================================
    required_cols = ['seller_lat', 'seller_lng', 'customer_lat', 'customer_lng']
    missing_cols = [col for col in required_cols if col not in df_network.columns]
    
    if missing_cols:
        st.warning(f"Missing coordinate columns. Attempting to add from state centroids...")
        
        if not state_centroids.empty:
            # Add seller coordinates
            if 'seller_state' in df_network.columns:
                df_network = df_network.merge(
                    state_centroids[['state', 'lat', 'lng']].rename(
                        columns={'state': 'seller_state', 'lat': 'seller_lat', 'lng': 'seller_lng'}
                    ),
                    on='seller_state',
                    how='left'
                )
            
            # Add customer coordinates
            if 'customer_state' in df_network.columns:
                df_network = df_network.merge(
                    state_centroids[['state', 'lat', 'lng']].rename(
                        columns={'state': 'customer_state', 'lat': 'customer_lat', 'lng': 'customer_lng'}
                    ),
                    on='customer_state',
                    how='left'
                )
    
    # Drop rows with missing coordinates
    df_network = df_network.dropna(subset=['seller_lat', 'seller_lng', 'customer_lat', 'customer_lng'])
    
    if df_network.empty:
        st.error("No routes with complete coordinate data available.")
        st.info("Please ensure network_data.parquet has seller_lat, seller_lng, customer_lat, customer_lng columns.")
        return
    
    # ============================================
    # SIDEBAR FILTERS
    # ============================================
    with st.sidebar:
        st.markdown('<div class="sidebar-header">MAP CONTROLS</div>', unsafe_allow_html=True)
        
        # Camera
        st.markdown("**Camera View**")
        pitch = st.slider("Pitch", 20.0, 80.0, 45.0, 5.0)
        zoom = st.slider("Zoom", 2.0, 6.0, 3.5, 0.5)
        
        # Data Filters
        st.markdown("**Route Filters**")
        min_orders = st.slider("Minimum Orders", 50, 2000, 100, 50)
        
        performance_filter = st.multiselect(
            "Performance",
            ['Fast', 'Normal', 'Slow', 'Critical'],
            default=['Fast', 'Normal', 'Slow', 'Critical']
        )
        
        # Warehouse
        st.markdown("**Warehouse Settings**")
        show_warehouses = st.checkbox("Show Recommended Warehouses", value=True)
        
        # Map Style Selection
        st.markdown("**Map Style**")
        map_style_option = st.selectbox(
            "",
            ["Light (Positron)", "Dark (Dark Matter)"],
            index=1
        )
    
    # ============================================
    # MAP STYLE - FREE, NO TOKEN REQUIRED
    # ============================================
    CARTO_LIGHT = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'
    CARTO_DARK = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'
    
    SELECTED_MAP_STYLE = CARTO_DARK
    
    # ============================================
    # DATA FILTERING
    # ============================================
    df_filtered = df_network[df_network['order_count'] >= min_orders].copy()
    df_filtered = df_filtered[df_filtered['performance'].isin(performance_filter)]
    
    # Limit for performance
    if len(df_filtered) > 800:
        df_filtered = df_filtered.nlargest(800, 'order_count')
    
    # Calculate line width based on order volume
    if len(df_filtered) > 0:
        max_orders = df_filtered['order_count'].max()
        df_filtered['line_width'] = (df_filtered['order_count'] / max_orders * 8).clip(2, 8)
    else:
        df_filtered['line_width'] = 6
    
    # Color mapping
    color_map = {
        'Fast': [16, 185, 129, 200],
        'Normal': [245, 158, 11, 210],
        'Slow': [249, 115, 22, 220],
        'Critical': [239, 68, 68, 230]
    }
    df_filtered['color'] = df_filtered['performance'].map(
        lambda x: color_map.get(x, [100, 116, 139, 180])
    )
    
    df_filtered['delivery_days_formatted'] = df_filtered['avg_delivery_days'].round(1)
    df_filtered['orders_formatted'] = df_filtered['order_count'].apply(lambda x: f"{x:,}")
    
    # ============================================
    # PREPARE NODE DATA (State Centroids)
    # ============================================
    node_data = state_centroids.copy() if not state_centroids.empty else pd.DataFrame()
    
    if not node_data.empty:
        if 'outgoing_volume' in node_data.columns:
            max_vol = node_data['outgoing_volume'].max()
            node_data['radius'] = (node_data['outgoing_volume'] / max_vol * 40000).clip(5000, 40000)
        else:
            node_data['radius'] = 15000
    
    # ============================================
    # PREPARE WAREHOUSE DATA
    # ============================================
    wh_data = None
    if show_warehouses and warehouse_candidates is not None and not warehouse_candidates.empty:
        wh_data = warehouse_candidates.copy()
        if 'warehouse_id' not in wh_data.columns:
            wh_data['warehouse_id'] = [f"WH-{i+1}" for i in range(len(wh_data))]
    
    # ============================================
    # CREATE LAYERS
    # ============================================
    layers = []
    
    # 1. Arc Layer (Routes)
    if not df_filtered.empty:        
        arc_layer = pdk.Layer(
            'ArcLayer',
            data=df_filtered,
            get_source_position=['seller_lng', 'seller_lat'],
            get_target_position=['customer_lng', 'customer_lat'],
            get_width='line_width',
            get_source_color='color',
            get_target_color='color',
            pickable=True,
            auto_highlight=True,
            highlight_color=[255, 255, 255, 100]
        )
        layers.append(arc_layer)
    
    # 2. Node Layer (State Centroids)
    if not node_data.empty:
        node_layer = pdk.Layer(
            'ScatterplotLayer',
            data=node_data,
            get_position=['lng', 'lat'],
            get_radius='radius',
            get_fill_color=[100, 116, 139, 40],
            get_line_color=[71, 85, 105, 180],
            line_width_min_pixels=1.5,
            pickable=False,
            auto_highlight=True
        )
        layers.append(node_layer)
    
    # 3. Warehouse Layer (if available)
    if wh_data is not None:
        warehouse_layer = pdk.Layer(
            'ScatterplotLayer',
            data=wh_data,
            get_position=['lng', 'lat'],
            get_radius=12000,
            get_fill_color=[249, 115, 22, 200],
            get_line_color=[234, 88, 12, 255],
            line_width_min_pixels=2,
            pickable=False
        )
        layers.append(warehouse_layer)
        
        # Add text labels for warehouses
        text_layer = pdk.Layer(
            'TextLayer',
            data=wh_data,
            get_position=['lng', 'lat'],
            get_text='warehouse_id',
            get_size=12,
            get_color=[30, 41, 59, 255],
            get_angle=0,
            get_text_anchor='"middle"',
            get_alignment_baseline='"center"'
        )
        layers.append(text_layer)
    
    # ============================================
    # BRAZIL COORDINATES
    # ============================================
    BRAZIL_CENTER = {
        'lat': -14.2350,
        'lng': -51.9253
    }
    
    # ============================================
    # RENDER MAP
    # ============================================
    if layers:
        # Create view state
        view_state = pdk.ViewState(
            latitude=BRAZIL_CENTER['lat'],
            longitude=BRAZIL_CENTER['lng'],
            zoom=zoom,
            pitch=pitch,
            bearing=0
        )
        
        # Tooltip configuration
        tooltip = {
            "html": """
            <div style="background: #1e293b; padding: 8px 12px; border-radius: 8px; 
                        border-left: 3px solid #f97316; font-family: 'Inter', sans-serif;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <b style="color: #f97316;">{seller_state} → {customer_state}</b><br>
                <span style="color: #94a3b8;">Orders:</span> <b style="color: white;">{orders_formatted}</b><br>
                <span style="color: #94a3b8;">Delivery:</span> <b style="color: white;">{delivery_days_formatted} days</b><br>
                <span style="color: #94a3b8;">Performance:</span> <b style="color: #f97316;">{performance}</b>
            </div>
            """,
            "style": {"backgroundColor": "transparent"}
        }
        
        # Create deck with FREE map style
        deck = pdk.Deck(
            map_style=SELECTED_MAP_STYLE,
            initial_view_state=view_state,
            layers=layers,
            tooltip=tooltip
        )
        
        # Render
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.info("No routes match the selected filters. Please adjust your criteria.")
    
    # ============================================
    # METRICS DASHBOARD
    # ============================================
    st.markdown("---")
    st.markdown("### Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Active Routes", f"{len(df_filtered):,}")
    with col2:
        total_orders_val = df_filtered['order_count'].sum()
        st.metric("Total Orders", f"{total_orders_val:,.0f}")
    with col3:
        avg_delivery_val = df_filtered['avg_delivery_days'].mean() if 'avg_delivery_days' in df_filtered.columns else 0
        st.metric("Avg Delivery", f"{avg_delivery_val:.1f} days")
    with col4:
        critical_count_val = len(df_filtered[df_filtered['performance'] == 'Critical'])
        st.metric("Critical Routes", f"{critical_count_val}")
    with col5:
        fast_count_val = len(df_filtered[df_filtered['performance'] == 'Fast'])
        st.metric("Fast Routes", f"{fast_count_val}")
          
    # ============================================
    # INSIGHTS ROW
    # ============================================
    st.markdown("---")
    st.markdown("### Quick Insights")
    
    col_i1, col_i2, col_i3 = st.columns(3)
    
    with col_i1:
        unique_sellers = df_filtered['seller_state'].nunique() if 'seller_state' in df_filtered.columns else 0
        st.metric("Active Seller States", f"{unique_sellers}")
    
    with col_i2:
        unique_buyers = df_filtered['customer_state'].nunique() if 'customer_state' in df_filtered.columns else 0
        st.metric("Active Buyer States", f"{unique_buyers}")
    
    with col_i3:
        on_time = len(df_filtered[df_filtered['avg_delivery_days'] <= 15]) if 'avg_delivery_days' in df_filtered.columns else 0
        on_time_pct = (on_time / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        st.metric("On-Time Delivery Rate", f"{on_time_pct:.0f}%")