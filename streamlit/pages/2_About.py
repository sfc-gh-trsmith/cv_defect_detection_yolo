"""
PCB Defect Detection - About Page

Comprehensive documentation for dual audiences:
- Executive Overview: Business context, outcomes, value
- Technical Deep-Dive: Architecture, algorithms, implementation
"""

import streamlit as st

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="About | PCB Defect Detection",
    page_icon="ℹ️",
    layout="wide"
)

# Dark theme styling
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
    }
    .tech-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        background-color: #1e40af;
        color: white;
    }
    .tech-badge-external {
        background-color: #b45309;
    }
    .tech-badge-model {
        background-color: #166534;
    }
    .tech-badge-stage {
        background-color: #7c3aed;
    }
    .arch-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .value-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .problem-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image("https://www.snowflake.com/wp-content/themes/flavor/flavortheme/assets/images/logo.svg", width=150)
    st.title("PCB Defect Detection")
    st.markdown("---")
    st.markdown("""
    **Quick Links**
    - [Executive Overview](#executive-overview)
    - [Technical Deep-Dive](#technical-deep-dive)
    - [Data Architecture](#data-architecture)
    - [Technology Stack](#technology-stack)
    """)

# =============================================================================
# HEADER
# =============================================================================

st.title("ℹ️ About This Application")
st.markdown("*AI-powered PCB defect detection using YOLOv12 on Snowflake Container Runtime*")

st.divider()

# =============================================================================
# OVERVIEW SECTION (Problem + Solution)
# =============================================================================

st.header("Overview")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔴 The Problem")
    st.markdown("""
    Electronics manufacturers face a critical quality control challenge: **legacy AOI (Automated Optical 
    Inspection) systems generate excessive false positives**, flagging 15-25% of boards as defective when 
    they're actually fine.
    
    This creates a cascade of problems:
    - **Expensive manual re-inspection** slows production and inflates labor costs
    - **Data silos** prevent ML teams from accessing factory floor images to train better models
    - **Security policies** block modern open-source AI from running inside the corporate firewall
    - **Reactive quality control** catches defects too late, after boards have moved through expensive downstream operations
    
    **The cost?** Poor quality costs electronics manufacturers 2-4% of annual revenue—that's $200-400M 
    for a $10B operation (McKinsey).
    """)

with col2:
    st.subheader("🟢 The Solution")
    st.markdown("""
    **Train custom YOLOv12 models directly inside Snowflake** using GPU compute—keeping sensitive 
    factory images secure while leveraging state-of-the-art computer vision.
    
    **Key Capabilities:**
    - ✅ GPU-accelerated training
    - ✅ 6 defect class detection
    - ✅ Real-time inference (<2s)
    - ✅ AI remediation guidance
    - ✅ Live analytics dashboard
    """)

st.divider()

# =============================================================================
# DATA ARCHITECTURE
# =============================================================================

st.header("Data Architecture")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="arch-card">
        <h4 style="color: #e2e8f0;">📊 Internal Data</h4>
        <span class="tech-badge">PCB_METADATA</span><br/>
        <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 0.9rem;">
            Board identification, manufacturing date, factory line, product type.<br/>
            <strong>~1M records</strong> | <strong>Refresh:</strong> Real-time
        </p>
        <span class="tech-badge">DEFECT_LOGS</span><br/>
        <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 0.9rem;">
            Inference results with class, confidence, bounding box coordinates.<br/>
            <strong>~30K records</strong> | <strong>Refresh:</strong> Per-inference
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="arch-card">
        <h4 style="color: #e2e8f0;">🌐 External Data</h4>
        <span class="tech-badge tech-badge-external">Deep PCB Dataset</span><br/>
        <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 0.9rem;">
            Open-source PCB defect dataset with 1,500 image pairs and 6 defect classes.<br/>
            <strong>MIT License</strong> | <strong>Source:</strong> Tang et al. 2019
        </p>
        <span class="tech-badge tech-badge-external">IPC Standards</span><br/>
        <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 0.9rem;">
            IPC-A-610 Acceptability standards for electronics assemblies (PDFs for RAG).<br/>
            <strong>Reference:</strong> Cortex Search
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="arch-card">
        <h4 style="color: #e2e8f0;">🤖 Model Outputs</h4>
        <span class="tech-badge tech-badge-model">YOLOv12 Weights</span><br/>
        <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 0.9rem;">
            Trained model weights (best.pt) stored in MODEL_STAGE.<br/>
            <strong>~15MB</strong> | <strong>Format:</strong> PyTorch .pt
        </p>
        <span class="tech-badge tech-badge-stage">MODEL_STAGE</span><br/>
        <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 0.9rem;">
            Internal stage for images, models, configs, and training artifacts.<br/>
            <strong>~500MB</strong> | <strong>Location:</strong> @MODEL_STAGE/
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =============================================================================
# HOW IT WORKS (Tabbed for dual audience)
# =============================================================================

st.header("How It Works")

exec_tab, tech_tab = st.tabs(["📊 Executive Overview", "🔧 Technical Deep-Dive"])

with exec_tab:
    st.markdown("### Why Traditional Approaches Fall Short")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="problem-card">
            <h4 style="color: #f87171;">Legacy AOI Systems</h4>
            <p style="color: #94a3b8;">
            Rigid rule-based inspection that cannot adapt to new products or learn from production reality.
            Results in 15-25% false positive rates.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="problem-card">
            <h4 style="color: #f87171;">Data Silos</h4>
            <p style="color: #94a3b8;">
            Factory images trapped in legacy systems. Data scientists cannot access the data needed to 
            train modern AI models.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="problem-card">
            <h4 style="color: #f87171;">Security Barriers</h4>
            <p style="color: #94a3b8;">
            Corporate policies block GPU infrastructure and open-source ML frameworks from running 
            inside the firewall.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### How This Solution Works")
    st.markdown("""
    Think of this solution as **bringing the AI to the data, not the data to the AI**.
    
    Instead of moving sensitive factory images to external cloud ML platforms, we train custom 
    detection models directly inside Snowflake—where your data already lives, governed by your 
    existing security policies.
    
    **The AI learns what defects really look like** on your production boards, not generic samples. 
    When it detects a defect, it can instantly retrieve the correct IPC repair procedure—so 
    technicians know exactly what to do.
    
    **What you get:**
    - A model trained specifically on YOUR production data
    - Detection accuracy that improves as you add more examples
    - Real-time visibility into quality across all production lines
    - AI-powered guidance for defect remediation
    """)
    
    st.markdown("### Business Value")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="value-card">
            <h2 style="color: #4ade80; margin: 0;">25%</h2>
            <p style="color: #f8fafc; margin: 0.5rem 0 0 0; font-weight: 600;">Fewer False Positives</p>
            <p style="color: #94a3b8; margin: 0; font-size: 0.8rem;">Custom models vs. legacy AOI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="value-card">
            <h2 style="color: #4ade80; margin: 0;">15%</h2>
            <p style="color: #f8fafc; margin: 0.5rem 0 0 0; font-weight: 600;">Scrap Reduction</p>
            <p style="color: #94a3b8; margin: 0; font-size: 0.8rem;">Earlier, more accurate detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="value-card">
            <h2 style="color: #4ade80; margin: 0;">&lt;2s</h2>
            <p style="color: #f8fafc; margin: 0.5rem 0 0 0; font-weight: 600;">Inference Time</p>
            <p style="color: #94a3b8; margin: 0; font-size: 0.8rem;">Real-time defect detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="value-card">
            <h2 style="color: #4ade80; margin: 0;">$1.1M</h2>
            <p style="color: #f8fafc; margin: 0.5rem 0 0 0; font-weight: 600;">Annual Value</p>
            <p style="color: #94a3b8; margin: 0; font-size: 0.8rem;">For 1M boards/year operation</p>
        </div>
        """, unsafe_allow_html=True)

with tech_tab:
    st.markdown("### Architecture Overview")
    
    # Display architecture diagram
    st.image("images/architecture.svg", caption="Solution Architecture", use_container_width=True)
    
    st.markdown("### Model Architecture: YOLOv12")
    
    st.markdown("""
    We use **YOLOv12** (You Only Look Once v12), an attention-centric real-time object detector 
    that achieves state-of-the-art accuracy while maintaining fast inference.
    
    **Key architectural innovations:**
    - **Area Attention Module**: Efficient attention mechanism for capturing global context
    - **R-ELAN Block**: Reparameterized ELAN for improved gradient flow
    - **Anchor-free Detection**: No predefined anchor boxes, enabling better generalization
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Training Configuration:**
        | Parameter | Value |
        |-----------|-------|
        | Model | YOLOv12n (nano) |
        | Input Size | 640 × 640 |
        | Epochs | 50 |
        | Batch Size | 16 |
        | Optimizer | SGD |
        | Learning Rate | 0.01 |
        | Compute | GPU_NV_M |
        """)
    
    with col2:
        st.markdown("""
        **Defect Classes:**
        | Class ID | Defect Type |
        |----------|-------------|
        | 0 | Open (broken trace) |
        | 1 | Short (unintended connection) |
        | 2 | Mousebite (irregular edge) |
        | 3 | Spur (unwanted protrusion) |
        | 4 | Copper (excess copper) |
        | 5 | Pin-hole (small void) |
        """)
    
    st.markdown("### Training Pipeline")
    
    st.markdown("""
    ```
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   1. INGEST     │ →  │   2. PREPARE    │ →  │    3. TRAIN     │ →  │   4. PERSIST    │
    │                 │    │                 │    │                 │    │                 │
    │ Download from   │    │ Convert to      │    │ YOLOv12 on      │    │ Upload best.pt  │
    │ @MODEL_STAGE    │    │ YOLO format     │    │ GPU compute     │    │ to stage        │
    │ to /tmp         │    │ (normalized)    │    │ (50 epochs)     │    │                 │
    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
    ```
    
    **Data Format Conversion:**
    - Deep PCB format: `x1 y1 x2 y2 class_id` (pixel coordinates)
    - YOLO format: `class_id x_center y_center width height` (normalized 0-1)
    
    **Normalization Formula:**
    ```python
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    ```
    """)
    
    st.markdown("### Snowflake Components")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Container Runtime:**
        - Python 3.11+ with pip flexibility
        - GPU_NV_M compute pool (Tesla T4 or equivalent)
        - External Access Integration for PyPI, GitHub, HuggingFace
        
        **Network Configuration:**
        ```sql
        CREATE NETWORK RULE CV_DEFECT_DETECTION_YOLO_EGRESS_RULE
          MODE = EGRESS
          TYPE = HOST_PORT
          VALUE_LIST = (
            'pypi.org',
            'files.pythonhosted.org',
            'github.com',
            'raw.githubusercontent.com',
            'huggingface.co'
          );
        ```
        """)
    
    with col2:
        st.markdown("""
        **Stage Structure:**
        ```
        @MODEL_STAGE/
        ├── raw/
        │   └── deeppcb/       # Training data
        ├── models/
        │   └── yolov12_pcb/   # Weights
        │       └── best.pt
        ├── config/
        │   └── data.yaml      # YOLO config
        └── runs/              # Training logs
        ```
        
        **Table Schema (DEFECT_LOGS):**
        ```sql
        inference_id   VARCHAR(50)  PK
        board_id       VARCHAR(50)  FK
        timestamp      TIMESTAMP
        detected_class VARCHAR(20)
        confidence     FLOAT
        bbox_x, bbox_y FLOAT
        bbox_w, bbox_h FLOAT
        ```
        """)
    
    st.markdown("### Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Time", "~30 min", "50 epochs on GPU_NV_M")
    
    with col2:
        st.metric("Model Size", "~15 MB", "YOLOv12n weights")
    
    with col3:
        st.metric("Inference Latency", "<2 sec", "Per image including I/O")
    
    st.markdown("""
    **Known Limitations:**
    - YOLOv12 requires CUDA-compatible GPU (handled by GPU compute pool)
    - `flash-attn` compilation may take 15-20 minutes on first run
    - Model trained on Deep PCB dataset; production deployment requires retraining on customer data
    """)

st.divider()

# =============================================================================
# APPLICATION PAGES
# =============================================================================

st.header("Application Pages")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="arch-card">
        <h4 style="color: #e2e8f0;">📊 Executive Overview</h4>
        <p style="color: #94a3b8;">
        High-level dashboard with yield rates, defect Pareto analysis, and trend visualization.
        Designed for VP of Operations and quality leadership.
        </p>
        <p style="color: #64748b; font-size: 0.85rem;">
        <strong>Key metrics:</strong> Yield rate, defect rate, false positive rate, boards processed
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="arch-card">
        <h4 style="color: #e2e8f0;">🔍 Vision Lab</h4>
        <p style="color: #94a3b8;">
        Interactive inference tool for uploading PCB images and viewing YOLOv12 detection results
        with bounding box overlays and confidence scores.
        </p>
        <p style="color: #64748b; font-size: 0.85rem;">
        <strong>Features:</strong> Image upload, real-time inference, defect visualization
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =============================================================================
# TECHNOLOGY STACK
# =============================================================================

st.header("Technology Stack")

st.markdown("""
<span class="tech-badge">Snowflake Notebooks</span>
<span class="tech-badge">Container Runtime</span>
<span class="tech-badge">GPU Compute Pool</span>
<span class="tech-badge">Streamlit in Snowflake</span>
<span class="tech-badge tech-badge-model">YOLOv12</span>
<span class="tech-badge tech-badge-model">PyTorch</span>
<span class="tech-badge tech-badge-model">Ultralytics</span>
<span class="tech-badge">Cortex Search</span>
<span class="tech-badge">Cortex Analyst</span>
<span class="tech-badge tech-badge-external">External Access Integration</span>
""", unsafe_allow_html=True)

st.divider()

# =============================================================================
# GETTING STARTED
# =============================================================================

st.header("Getting Started")

st.markdown("""
Deploy the complete pipeline in your Snowflake account:

```bash
# 1. Deploy infrastructure (compute pools, network rules, stages)
./deploy.sh

# 2. Run the training notebook on GPU
./run.sh main

# 3. Launch the Streamlit dashboard
./run.sh streamlit

# 4. Clean up when done
./clean.sh --force
```

**Prerequisites:**
- Snowflake account with SPCS and Container Runtime enabled
- ACCOUNTADMIN or equivalent role for initial setup
- Quota for GPU_NV_M compute pool
""")

st.divider()

# =============================================================================
# LICENSES & ATTRIBUTIONS
# =============================================================================

st.header("Licenses & Attributions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="arch-card">
        <h4 style="color: #e2e8f0;">YOLOv12</h4>
        <span class="tech-badge tech-badge-model">AGPL-3.0</span>
        <p style="color: #94a3b8; margin-top: 0.5rem;">
            Attention-centric real-time object detector by University at Buffalo 
            and University of Chinese Academy of Sciences.
        </p>
        <p style="color: #94a3b8; font-size: 0.75rem;">
            <strong>Authors:</strong> Tian, Yunjie; Ye, Qixiang; Doermann, David<br/>
            <strong>Paper:</strong> <a href="https://arxiv.org/abs/2502.12524" style="color: #64D2FF;">arXiv:2502.12524</a><br/>
            <strong>Repository:</strong> <a href="https://github.com/sunsmarterjie/yolov12" style="color: #64D2FF;">github.com/sunsmarterjie/yolov12</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="arch-card">
        <h4 style="color: #e2e8f0;">Deep PCB Dataset</h4>
        <span class="tech-badge tech-badge-external">MIT License</span>
        <p style="color: #94a3b8; margin-top: 0.5rem;">
            PCB defect detection dataset with 1,500 image pairs and 6 defect classes.
        </p>
        <p style="color: #94a3b8; font-size: 0.75rem;">
            <strong>Authors:</strong> Tang, Sanli; He, Fan; Huang, Xiaolin; Yang, Jie<br/>
            <strong>Paper:</strong> <a href="https://arxiv.org/abs/1902.06197" style="color: #64D2FF;">arXiv:1902.06197</a><br/>
            <strong>Repository:</strong> <a href="https://github.com/tangsanli5201/DeepPCB" style="color: #64D2FF;">github.com/tangsanli5201/DeepPCB</a>
        </p>
    </div>
""", unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **PCB Defect Detection Demo**  
    AI-powered quality control using YOLOv12 on Snowflake Container Runtime with GPU compute.
    """)

with col2:
    st.markdown("""
    **Resources**  
    - [Snowflake Notebooks](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks)
    - [Container Runtime](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview)
    - [YOLOv12 Docs](https://docs.ultralytics.com/models/yolo12/)
    """)
