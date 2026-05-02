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
    # CartoDB Positron (Light) - for bright theme
    CARTO_LIGHT = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'
    
    # CartoDB Dark Matter (Dark) - for dark theme
    CARTO_DARK = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'
    
    # Select style based on user preference
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

    df_filtered['warehouse_id'] = ''
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
        st.markdown(f"""
        <div style="background: #ffffff; border-radius: 12px; padding: 1rem 1.25rem; 
                    border: 1px solid #e2e8f0; height: 100%;">
              <div style="font-size: 1.75rem; font-weight: 700; color: #0f172a;">{len(df_filtered):,}</div>
              <div style="font-size: 0.7rem; color: #64748b; font-weight: 500; 
                        text-transform: uppercase; letter-spacing: 0.03em; margin-top: 0.5rem;">Active Routes</div>
    </div>
    """, unsafe_allow_html=True)

    with col2:
        total_orders = df_filtered['order_count'].sum()
        st.markdown(f"""
        <div style="background: #ffffff; border-radius: 12px; padding: 1rem 1.25rem; 
                    border: 1px solid #e2e8f0; height: 100%;">
            <div style="font-size: 1.75rem; font-weight: 700; color: #0f172a;">{total_orders:,.0f}</div>
            <div style="font-size: 0.7rem; color: #64748b; font-weight: 500; 
                      text-transform: uppercase; letter-spacing: 0.03em; margin-top: 0.5rem;">Total Orders</div>
    </div>
    """, unsafe_allow_html=True)

    with col3:
          avg_delivery = df_filtered['avg_delivery_days'].mean() if 'avg_delivery_days' in df_filtered.columns else 0
          st.markdown(f"""
          <div style="background: #ffffff; border-radius: 12px; padding: 1rem 1.25rem; 
                      border: 1px solid #e2e8f0; height: 100%;">
              <div style="font-size: 1.75rem; font-weight: 700; color: #0f172a;">{avg_delivery:.1f}</div>
              <div style="font-size: 0.7rem; color: #64748b; font-weight: 500; 
                          text-transform: uppercase; letter-spacing: 0.03em; margin-top: 0.5rem;">Avg Delivery (days)</div>
    </div>
    """, unsafe_allow_html=True)

    with col4:
          critical_count = len(df_filtered[df_filtered['performance'] == 'Critical'])
          st.markdown(f"""
          <div style="background: #ffffff; border-radius: 12px; padding: 1rem 1.25rem; 
                      border: 1px solid #e2e8f0; height: 100%;">
              <div style="font-size: 1.75rem; font-weight: 700; color: #ef4444;">{critical_count}</div>
              <div style="font-size: 0.7rem; color: #64748b; font-weight: 500; 
                          text-transform: uppercase; letter-spacing: 0.03em; margin-top: 0.5rem;">Critical Routes</div>
    </div>
    """, unsafe_allow_html=True)

    with col5:
          fast_count = len(df_filtered[df_filtered['performance'] == 'Fast'])
          st.markdown(f"""
          <div style="background: #ffffff; border-radius: 12px; padding: 1rem 1.25rem; 
                      border: 1px solid #e2e8f0; height: 100%;">
              <div style="font-size: 1.75rem; font-weight: 700; color: #10b981;">{fast_count}</div>
              <div style="font-size: 0.7rem; color: #64748b; font-weight: 500; 
                          text-transform: uppercase; letter-spacing: 0.03em; margin-top: 0.5rem;">Fast Routes</div>
    </div>
    """, unsafe_allow_html=True)
          
    # ============================================
    # INSIGHTS ROW
    # ============================================
    st.markdown("---")
    st.markdown("### Quick Insights")
    
    col_i1, col_i2, col_i3 = st.columns(3)
    
    with col_i1:
        unique_sellers = df_filtered['seller_state'].nunique() if 'seller_state' in df_filtered.columns else 0
        st.markdown(f"""
        <div class="card">
            <div class="insight-title">Active Seller States</div>
            <div class="insight-value">{unique_sellers}</div>
            <div class="insight-trend">states with outbound shipments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_i2:
        unique_buyers = df_filtered['customer_state'].nunique() if 'customer_state' in df_filtered.columns else 0
        st.markdown(f"""
        <div class="card">
            <div class="insight-title">Active Buyer States</div>
            <div class="insight-value">{unique_buyers}</div>
            <div class="insight-trend">states receiving deliveries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_i3:
        on_time = len(df_filtered[df_filtered['avg_delivery_days'] <= 15]) if 'avg_delivery_days' in df_filtered.columns else 0
        on_time_pct = (on_time / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        st.markdown(f"""
        <div class="card">
            <div class="insight-title">On-Time Delivery Rate</div>
            <div class="insight-value">{on_time_pct:.0f}%</div>
            <div class="insight-trend">delivered within 15 days</div>
        </div>
        """, unsafe_allow_html=True)
    

def add_insight_styles():
    """Add CSS styles for insight cards"""
    st.markdown("""
    <style>
    .insight-title {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }
    .insight-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }
    .insight-trend {
        font-size: 0.7rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)


# Call CSS function
add_insight_styles()