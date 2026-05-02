"""
Warehouse Optimization - Strategic Warehouse Placement Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data.loader import load_all

def render_warehouse_optimization():
    """Render warehouse optimization dashboard"""
    
    st.markdown('<div class="page-title">Warehouse Optimization</div>', unsafe_allow_html=True)
    st.markdown("Strategic analysis for optimal warehouse placement to reduce delivery times")
    
    # Load data
    @st.cache_data
    def get_data():
        data = load_all()
        return data
    
    data = get_data()
    
    warehouse_candidates = data.get('warehouse_candidates', None)
    df_network = data.get('network', pd.DataFrame())
    
    st.markdown("### Current Network Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df_network):,}</div>
            <div class="metric-label">Current Routes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_delivery = df_network['avg_delivery_days'].mean() if 'avg_delivery_days' in df_network.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_delivery:.1f} days</div>
            <div class="metric-label">Current Avg Delivery</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if warehouse_candidates is not None:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(warehouse_candidates)}</div>
                <div class="metric-label">Proposed Warehouses</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Warehouse candidates display
    st.markdown('<div class="card-title">Recommended Warehouse Locations</div>', unsafe_allow_html=True)
    
    if warehouse_candidates is not None and not warehouse_candidates.empty:
        col_map, col_table = st.columns([2, 1])
        
        with col_map:
            # Create a simple map visualization
            import pydeck as pdk
            from src.visualization.pydeck_layers import get_warehouse_layer, get_view_state
            
            warehouse_layer = get_warehouse_layer(
                data=warehouse_candidates,
                radius=15000,
                fill_color=[249, 115, 22, 200],
                line_color=[234, 88, 12, 255]
            )
            
            deck = pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v11',
                initial_view_state=get_view_state(zoom=3.5, pitch=30),
                layers=[warehouse_layer],
                tooltip={"html": "<b>{warehouse_id}</b><br>Cluster Size: {cluster_size}"}
            )
            st.pydeck_chart(deck, use_container_width=True)
        
        with col_table:
            st.dataframe(
                warehouse_candidates[['warehouse_id', 'lat', 'lng', 'cluster_size']],
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("No warehouse candidates available. Run warehouse optimization algorithm first.")
        
        # Demo data for visualization
        demo_warehouses = pd.DataFrame({
            'warehouse_id': ['WH_SP', 'WH_RJ', 'WH_MG', 'WH_RS', 'WH_BA'],
            'lat': [-23.55, -22.91, -19.92, -30.03, -12.97],
            'lng': [-46.63, -43.20, -43.94, -51.23, -38.51],
            'cluster_size': [2450, 1830, 1250, 980, 760]
        })
        
        st.markdown("#### Preview: Example Warehouse Locations")
        st.dataframe(demo_warehouses, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Impact Analysis
    st.markdown('<div class="card-title">Potential Impact Analysis</div>', unsafe_allow_html=True)
    
    col_imp1, col_imp2 = st.columns(2)
    
    with col_imp1:
        st.markdown("#### Estimated Benefits")
        
        benefits = {
            "Metric": [
                "Delivery Time Reduction",
                "Freight Cost Reduction",
                "Route Optimization",
                "Carbon Footprint Reduction"
            ],
            "Estimated Improvement": [
                "25-35%",
                "15-20%",
                "30-40%",
                "20-25%"
            ]
        }
        st.dataframe(pd.DataFrame(benefits), use_container_width=True, hide_index=True)
    
    with col_imp2:
        st.markdown("#### ROI Analysis")
        
        # Create a simple gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 85,
            title = {'text': "Projected ROI", 'font': {'size': 14}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "#f97316"},
                'steps': [
                    {'range': [0, 50], 'color': "#fee2e2"},
                    {'range': [50, 80], 'color': "#fed7aa"},
                    {'range': [80, 100], 'color': "#d1fae5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # Implementation Plan
    st.markdown("---")
    st.markdown('<div class="card-title">Implementation Roadmap</div>', unsafe_allow_html=True)
    
    roadmap = pd.DataFrame({
        "Phase": ["Phase 1", "Phase 2", "Phase 3", "Phase 4"],
        "Timeline": ["Month 1-2", "Month 3-4", "Month 5-6", "Month 7-8"],
        "Activities": [
            "Feasibility Study & Site Selection",
            "Warehouse Construction/Lease",
            "Logistics Integration & Testing",
            "Full Deployment & Monitoring"
        ],
        "Status": ["Planning", "Not Started", "Not Started", "Not Started"]
    })
    st.dataframe(roadmap, use_container_width=True, hide_index=True)