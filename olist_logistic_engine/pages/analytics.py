"""
Route Analytics Page - Performance Metrics and Charts
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils.helpers import format_number, format_currency


def render_analytics():
    """Render the route analytics page"""
    
    st.markdown('<div class="page-header">Route Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-description">Comprehensive analysis of delivery routes, performance metrics, and trends</div>', unsafe_allow_html=True)
    data = load_all()
    warehouse_candidates = data.get('warehouse_candidates', None)
    df_network = data['network'].copy()
    
    # ============================================
    # FILTERS
    # ============================================
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        min_orders_filter = st.slider("Min Orders", 50, 2000, 50, step=50, key="analytics_min_orders")
    
    with col_f2:
        performance_filter = st.multiselect(
            "Performance",
            options=['Fast', 'Normal', 'Slow', 'Critical'],
            default=['Fast', 'Normal', 'Slow', 'Critical'],
            key="analytics_performance"
        )
    
    with col_f3:
        top_n = st.selectbox("Show Top N Routes", [10, 20, 50, 100], index=1, key="analytics_top_n")
    
    # Apply filters
    df_filtered = df_network[df_network['order_count'] >= min_orders_filter].copy()
    df_filtered = df_filtered[df_filtered['performance'].isin(performance_filter)]
    
    # ============================================
    # KPI ROW
    # ============================================
    st.markdown("### Key Performance Indicators")
    
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    
    total_orders = df_filtered['order_count'].sum()
    avg_delivery = df_filtered['avg_delivery_days'].mean() if 'avg_delivery_days' in df_filtered.columns else 0
    avg_freight = df_filtered['avg_freight'].mean() if 'avg_freight' in df_filtered.columns else 0
    total_revenue = df_filtered['total_revenue'].sum() if 'total_revenue' in df_filtered.columns else 0
    unique_routes = df_filtered['seller_state'].nunique() * df_filtered['customer_state'].nunique()
    
    with col_k1:
        st.metric("Total Orders", format_number(total_orders))
    with col_k2:
        st.metric("Avg Delivery", f"{avg_delivery:.1f} days")
    with col_k3:
        st.metric("Avg Freight", format_currency(avg_freight))
    with col_k4:
        st.metric("Total Revenue", format_currency(total_revenue))
    with col_k5:
        st.metric("Active Routes", len(df_filtered))
    
    # ============================================
    # CHARTS SECTION
    # ============================================
    st.markdown("---")
    st.markdown("### Performance Analysis")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        # Performance Distribution
        perf_counts = df_filtered['performance'].value_counts().reset_index()
        perf_counts.columns = ['Performance', 'Count']
        
        fig1 = px.bar(
            perf_counts,
            x='Performance',
            y='Count',
            color='Performance',
            color_discrete_map={
                'Fast': '#10b981',
                'Normal': '#f59e0b',
                'Slow': '#f97316',
                'Critical': '#ef4444'
            },
            title="Route Performance Distribution",
            text_auto=True
        )
        fig1.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_c2:
        # Top Routes by Order Volume
        top_routes = df_filtered.nlargest(top_n, 'order_count')[['seller_state', 'customer_state', 'order_count']]
        top_routes['route'] = top_routes['seller_state'] + " → " + top_routes['customer_state']
        
        fig2 = px.bar(
            top_routes,
            x='order_count',
            y='route',
            orientation='h',
            title=f"Top {top_n} Routes by Order Volume",
            color='order_count',
            color_continuous_scale='oranges'
        )
        fig2.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            xaxis_title="Order Count",
            yaxis_title="Route"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # ============================================
    # DELIVERY TIME ANALYSIS
    # ============================================
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        # Delivery Time by Route
        if 'avg_delivery_days' in df_filtered.columns:
            delivery_data = df_filtered.groupby('seller_state')['avg_delivery_days'].mean().reset_index()
            delivery_data.columns = ['State', 'Avg Delivery Days']
            
            fig3 = px.bar(
                delivery_data.sort_values('Avg Delivery Days', ascending=False).head(15),
                x='State',
                y='Avg Delivery Days',
                title="Average Delivery Days by Seller State",
                color='Avg Delivery Days',
                color_continuous_scale='reds'
            )
            fig3.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    with col_d2:
        # Distance vs Delivery Time
        if 'distance_km' in df_filtered.columns and 'avg_delivery_days' in df_filtered.columns:
            fig4 = px.scatter(
                df_filtered,
                x='distance_km',
                y='avg_delivery_days',
                size='order_count',
                color='performance',
                color_discrete_map={
                    'Fast': '#10b981',
                    'Normal': '#f59e0b',
                    'Slow': '#f97316',
                    'Critical': '#ef4444'
                },
                title="Distance vs Delivery Time",
                labels={'distance_km': 'Distance (km)', 'avg_delivery_days': 'Delivery Days'},
                hover_data=['seller_state', 'customer_state']
            )
            fig4.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=400
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    # ============================================
    # DATA TABLE
    # ============================================
    st.markdown("---")
    st.markdown("### Route Details")
    
    display_cols = ['seller_state', 'customer_state', 'order_count', 'avg_delivery_days', 'performance']
    if 'distance_km' in df_filtered.columns:
        display_cols.insert(2, 'distance_km')
    if 'avg_freight' in df_filtered.columns:
        display_cols.append('avg_freight')
    
    st.dataframe(
        df_filtered[display_cols].sort_values('order_count', ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    # ============================================
    # INSIGHTS SECTION
    # ============================================
    st.markdown("---")
    st.markdown("### Key Insights")
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Performance Summary</div>
        """, unsafe_allow_html=True)
        
        fast_pct = (len(df_filtered[df_filtered['performance'] == 'Fast']) / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        slow_pct = (len(df_filtered[df_filtered['performance'].isin(['Slow', 'Critical'])]) / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        
        st.markdown(f"""
        <ul style="color: #0f172a;">
            <li><span class="badge-success">Fast Routes</span> = {fast_pct:.1f}% of all routes</li>
            <li><span class="badge-danger">Slow/Critical Routes</span> = {slow_pct:.1f}% of all routes</li>
            <li>Average delivery time across all routes = {avg_delivery:.1f} days</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_i2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Recommendations</div>
        """, unsafe_allow_html=True)
        
        recommendations = []
        if avg_delivery > 15:
            recommendations.append("Consider optimizing long-haul routes")
        if len(df_filtered[df_filtered['performance'] == 'Critical']) > 10:
            recommendations.append("Critical routes need immediate attention")
        if not recommendations:
            recommendations.append("Network performance is good. Maintain current operations.")
        
        for i, rec in enumerate(recommendations[:3]):
            st.markdown(f"- {rec}")
        st.markdown("</div>", unsafe_allow_html=True)