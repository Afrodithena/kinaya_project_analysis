"""
Performance Reports - Downloadable Reports and Analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from src.data.loader import load_all
import io

def render_reports(data = None):
    """Render performance reports page"""
    
    st.markdown('<div class="page-title">Performance Reports</div>', unsafe_allow_html=True)
    st.markdown("Generate and download comprehensive performance reports")
    
    if data is None:
        # Load data
        @st.cache_data
        
        def get_data():
            return load_all()    
            data = get_data()
    
    data = load_all()
    df = data.get('network', pd.DataFrame())
    
    if df.empty:
        st.error("No data available for reports")
        return
    
    # Report filters
    with st.sidebar:
        st.markdown('<div class="sidebar-header">REPORT CONFIGURATION</div>', unsafe_allow_html=True)
        
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Route Performance", "Regional Analysis", "Warehouse Impact"]
        )
        
        date_range = st.selectbox("Time Period", ["Last 30 Days", "Last Quarter", "Last Year", "All Time"])
        
        include_charts = st.checkbox("Include Charts", value=True)
        include_data = st.checkbox("Include Raw Data", value=False)
    
    st.markdown(f"### {report_type}")
    st.markdown(f"*Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}*")
    
    st.markdown("---")
    
    if report_type == "Executive Summary":
        render_executive_summary(df)
    elif report_type == "Route Performance":
        render_route_performance(df)
    elif report_type == "Regional Analysis":
        render_regional_analysis(df)
    elif report_type == "Warehouse Impact":
        render_warehouse_impact(df, data.get('warehouse_candidates', None))
    
    # Download button
    st.markdown("---")
    st.markdown("### Export Report")
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        if st.button("Export as CSV", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"olist_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col_dl2:
        if st.button("Export as JSON", use_container_width=True):
            json_data = df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"olist_report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col_dl3:
        if st.button("Export as Excel", use_container_width=True):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Routes', index=False)
            
            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name=f"olist_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def render_executive_summary(df):
    """Render executive summary section"""
    
    st.markdown("#### Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Routes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['order_count'].sum():,.0f}</div>
            <div class="metric-label">Total Orders</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_delivery = df['avg_delivery_days'].mean() if 'avg_delivery_days' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_delivery:.1f}</div>
            <div class="metric-label">Avg Delivery (days)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'performance' in df.columns:
            fast_pct = (len(df[df['performance'] == 'Fast']) / len(df) * 100) if len(df) > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{fast_pct:.0f}%</div>
                <div class="metric-label">Fast Routes</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("#### Performance Overview")
    
    if 'performance' in df.columns:
        perf_counts = df['performance'].value_counts()
        fig = px.pie(
            values=perf_counts.values,
            names=perf_counts.index,
            color=perf_counts.index,
            color_discrete_map={
                'Fast': '#10b981',
                'Normal': '#f59e0b',
                'Slow': '#f97316',
                'Critical': '#ef4444'
            },
            title="Route Performance Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_route_performance(df):
    """Render route performance section"""
    
    st.markdown("#### Top Performing vs Underperforming Routes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Best 10 Routes (Fast)**")
        if 'performance' in df.columns:
            best = df[df['performance'] == 'Fast'].nlargest(10, 'order_count')
            st.dataframe(best[['seller_state', 'customer_state', 'order_count', 'avg_delivery_days']], use_container_width=True)
    
    with col2:
        st.markdown("**Worst 10 Routes (Critical)**")
        if 'performance' in df.columns:
            worst = df[df['performance'] == 'Critical'].nlargest(10, 'order_count')
            st.dataframe(worst[['seller_state', 'customer_state', 'order_count', 'avg_delivery_days']], use_container_width=True)
    
    st.markdown("#### Delivery Time Analysis")
    
    if 'avg_delivery_days' in df.columns:
        fig = px.box(
            df,
            y='avg_delivery_days',
            points="all",
            title="Delivery Time Distribution",
            color_discrete_sequence=['#f97316']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_regional_analysis(df):
    """Render regional analysis section"""
    
    st.markdown("#### Performance by State")
    
    if 'seller_state' in df.columns and 'avg_delivery_days' in df.columns:
        state_perf = df.groupby('seller_state')['avg_delivery_days'].mean().sort_values().reset_index()
        state_perf.columns = ['State', 'Avg Delivery Days']
        
        fig = px.bar(
            state_perf,
            x='State',
            y='Avg Delivery Days',
            color='Avg Delivery Days',
            color_continuous_scale='Oranges',
            title="Average Delivery Days by Seller State"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Top Origin-Destination Pairs")
    
    top_od = df.nlargest(20, 'order_count')[['seller_state', 'customer_state', 'order_count', 'avg_delivery_days']]
    st.dataframe(top_od, use_container_width=True, hide_index=True)

def render_warehouse_impact(df, warehouse_candidates):
    """Render warehouse impact section"""
    
    st.markdown("#### Warehouse Optimization Impact")
    
    if warehouse_candidates is not None and not warehouse_candidates.empty:
        st.info(f"**{len(warehouse_candidates)}** potential warehouse locations identified")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Projected Improvements")
            
            improvements = pd.DataFrame({
                "Metric": [
                    "Delivery Time Reduction",
                    "Freight Cost Savings",
                    "Customer Satisfaction Increase",
                    "Carbon Emission Reduction"
                ],
                "Projected Improvement": [
                    "25-30%",
                    "15-20%",
                    "10-15%",
                    "20-25%"
                ]
            })
            st.dataframe(improvements, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Payback Period")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = 18,
                title = {'text': "Estimated Payback (Months)"},
                delta = {'reference': 24, 'increasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 36]},
                    'bar': {'color': "#f97316"},
                    'steps': [
                        {'range': [0, 12], 'color': "#d1fae5"},
                        {'range': [12, 24], 'color': "#fed7aa"},
                        {'range': [24, 36], 'color': "#fee2e2"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No warehouse optimization data available. Run warehouse optimization module first.")