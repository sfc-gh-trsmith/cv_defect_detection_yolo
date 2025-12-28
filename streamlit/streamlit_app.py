"""
PCB Defect Detection Dashboard - Executive Overview

Main entry point for the Streamlit application.
Displays KPIs, defect distribution, and trends.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from snowflake.snowpark.context import get_active_session

from utils.data_loader import (
    load_defect_summary,
    load_daily_trends,
    load_factory_line_data
)
from utils.query_registry import (
    execute_query,
    TOTAL_DEFECTS_SQL,
    PCB_COUNT_SQL
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #64D2FF;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image("streamlit/images/logo.svg", width=150)
    st.title("PCB Defect Detection")
    st.markdown("---")
    
    st.markdown("---")
    st.markdown("### Quick Stats")

# =============================================================================
# DATA LOADING
# =============================================================================

session = get_active_session()

try:
    # Load data
    defect_summary = load_defect_summary(session)
    daily_trends = load_daily_trends(session)
    factory_data = load_factory_line_data(session)
    
    # Get total counts
    total_df = execute_query(session, TOTAL_DEFECTS_SQL, "total_defects")
    pcb_df = execute_query(session, PCB_COUNT_SQL, "pcb_count")
    
    total_defects = int(total_df['TOTAL_DEFECTS'].iloc[0]) if not total_df.empty else 0
    total_pcbs = int(pcb_df['TOTAL_PCBS'].iloc[0]) if not pcb_df.empty else 0
    
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False
    total_defects = 0
    total_pcbs = 0

# =============================================================================
# HEADER
# =============================================================================

st.title("🔍 PCB Defect Detection Dashboard")
st.markdown("*Real-time defect analytics powered by YOLOv12 on Snowflake*")

# =============================================================================
# KPI CARDS
# =============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_defects:,}</div>
        <div class="metric-label">Total Defects</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_pcbs:,}</div>
        <div class="metric-label">PCBs Inspected</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    defect_rate = (total_defects / max(total_pcbs, 1)) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{defect_rate:.1f}%</div>
        <div class="metric-label">Defect Rate</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    num_classes = len(defect_summary) if data_loaded and not defect_summary.empty else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{num_classes}</div>
        <div class="metric-label">Defect Types</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# CHARTS
# =============================================================================

if data_loaded and not defect_summary.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Defect Distribution (Pareto)")
        
        # Sort by count for Pareto
        df_sorted = defect_summary.sort_values('DEFECT_COUNT', ascending=False)
        
        fig = go.Figure()
        
        # Bar chart
        fig.add_trace(go.Bar(
            x=df_sorted['DETECTED_CLASS'],
            y=df_sorted['DEFECT_COUNT'],
            name='Count',
            marker_color='#64D2FF'
        ))
        
        # Cumulative line
        df_sorted['CUMULATIVE_PCT'] = df_sorted['DEFECT_COUNT'].cumsum() / df_sorted['DEFECT_COUNT'].sum() * 100
        fig.add_trace(go.Scatter(
            x=df_sorted['DETECTED_CLASS'],
            y=df_sorted['CUMULATIVE_PCT'],
            name='Cumulative %',
            yaxis='y2',
            line=dict(color='#FF9F0A', width=2),
            mode='lines+markers'
        ))
        
        fig.update_layout(
            paper_bgcolor='#0f172a',
            plot_bgcolor='#0f172a',
            font=dict(color='#e2e8f0'),
            yaxis=dict(title='Count', gridcolor='#334155'),
            yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 105]),
            xaxis=dict(title='Defect Class', gridcolor='#334155'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏭 Factory Line Performance")
        
        if not factory_data.empty:
            # Pivot for heatmap
            pivot_df = factory_data.pivot_table(
                index='FACTORY_LINE_ID',
                columns='DETECTED_CLASS',
                values='DEFECT_COUNT',
                fill_value=0
            )
            
            fig = px.imshow(
                pivot_df.values,
                labels=dict(x="Defect Type", y="Factory Line", color="Count"),
                x=pivot_df.columns.tolist(),
                y=pivot_df.index.tolist(),
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                paper_bgcolor='#0f172a',
                plot_bgcolor='#0f172a',
                font=dict(color='#e2e8f0'),
                margin=dict(l=40, r=40, t=40, b=40),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No factory line data available")

    # Trends chart
    st.subheader("📈 Defect Trends Over Time")
    
    if not daily_trends.empty:
        fig = px.line(
            daily_trends,
            x='DETECTION_DATE',
            y='DEFECT_COUNT',
            color='DETECTED_CLASS',
            markers=True
        )
        
        fig.update_layout(
            paper_bgcolor='#0f172a',
            plot_bgcolor='#0f172a',
            font=dict(color='#e2e8f0'),
            xaxis=dict(title='Date', gridcolor='#334155'),
            yaxis=dict(title='Defect Count', gridcolor='#334155'),
            legend=dict(title='Defect Class'),
            margin=dict(l=40, r=40, t=40, b=40),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trend data available yet. Run the notebook to generate defect logs.")

else:
    st.info("📭 No defect data available. Run the YOLOv12 training notebook to generate inference results.")
    
    st.markdown("""
    ### Getting Started
    
    1. **Deploy the infrastructure**: Run `./deploy.sh` to set up Snowflake resources
    2. **Execute the notebook**: Run `./run.sh main` to train YOLOv12 and generate defect logs
    3. **Refresh this dashboard**: Data will appear automatically after inference runs
    """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption("Powered by Snowflake Notebooks (Container Runtime) with GPU • YOLOv12 Object Detection")

