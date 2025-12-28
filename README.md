# PCB Defect Detection with YOLOv12

A demonstration of training YOLOv12 object detection models on Snowflake Container Runtime with GPU compute for PCB (Printed Circuit Board) defect detection.

## Overview

Electronics manufacturers face high scrap costs and yield losses due to rigid, legacy Automated Optical Inspection (AOI) systems that generate excessive false positives. Data silos prevent Data Science teams from accessing factory floor image data to train modern, adaptive Computer Vision models, while security policies restrict deploying open-source innovations like YOLO inside the corporate firewall.

This project demonstrates training a state-of-the-art **YOLOv12** object detection model directly inside Snowflake using GPU compute, enabling adaptive defect detection without moving sensitive image data off-platform.

### Business Value

| Metric | Target Improvement |
|--------|-------------------|
| False Positive Rate | 25% reduction |
| Scrap Cost | 15% reduction |
| Time-to-Insight | Real-time vs. weekly |

### User Personas

| Persona | Role | Use Case |
|---------|------|----------|
| Strategic | VP of Operations | Ask plain English questions about global defect density per product line |
| Operational | QA Manager | Visually verify flagged boards and retrieve rework protocols |
| Technical | Data Scientist | Train YOLOv12 models using GPU compute without moving data off-platform |

## Features

- **GPU-Accelerated Training**: Train YOLOv12 models on Snowflake's Container Runtime with NVIDIA GPU compute pools
- **6 Defect Classes**: Detect open, short, mousebite, spur, copper, and pin-hole defects
- **Secure Data Processing**: Images never leave the Snowflake platform
- **Interactive Dashboard**: 3-page Streamlit application with executive analytics, vision lab, and documentation
- **RAG Integration**: Cortex Search for IPC standard lookups and defect remediation guidance

## Prerequisites

- **Snowflake Account** with:
  - Snowpark Container Services (SPCS) enabled
  - Container Runtime enabled
  - GPU compute pool availability (`GPU_NV_M`)
- **Snowflake CLI** (`snow`): Install with `pip install snowflake-cli`
- **Configured Connection**: A named connection in your Snowflake CLI config

## Quick Start

```bash
# 1. Deploy infrastructure (creates database, roles, compute pool, notebook, Streamlit app)
./deploy.sh

# 2. Run the training notebook (trains YOLOv12 on GPU, ~10-30 minutes)
./run.sh main

# 3. Open the dashboard
./run.sh streamlit

# 4. Clean up resources when done
./clean.sh --force
```

## Deployment Scripts

### deploy.sh

Deploys all infrastructure and applications to Snowflake.

```bash
./deploy.sh [OPTIONS]

Options:
  -c, --connection NAME    Snowflake CLI connection name (default: demo)
  -p, --prefix PREFIX      Environment prefix for resources (e.g., DEV, PROD)
  --only-sql               Deploy only SQL infrastructure
  --only-data              Upload only dataset to stage
  --only-notebook          Deploy only the notebook
  --only-streamlit         Deploy only the Streamlit app
  -h, --help               Show help message

Examples:
  ./deploy.sh                       # Full deployment
  ./deploy.sh -c prod               # Use 'prod' connection
  ./deploy.sh -p DEV                # Prefix all resources with DEV_
  ./deploy.sh --only-streamlit      # Redeploy Streamlit only
```

### run.sh

Execute operations and manage resources.

```bash
./run.sh [OPTIONS] COMMAND

Commands:
  main       Execute the YOLOv12 training notebook
  test       Run query test suite (validates all Streamlit queries)
  status     Check status of all resources
  streamlit  Get Streamlit app URL
  suspend    Suspend the compute pool (stops billing)
  resume     Resume the compute pool

Options:
  -c, --connection NAME    Snowflake CLI connection name (default: demo)
  -p, --prefix PREFIX      Environment prefix for resources
  -h, --help               Show help message

Examples:
  ./run.sh main              # Execute notebook
  ./run.sh test              # Run query tests
  ./run.sh status            # Check resources
  ./run.sh suspend           # Stop compute pool billing
  ./run.sh -c prod streamlit # Get URL using 'prod' connection
```

### clean.sh

Remove all project resources from Snowflake.

```bash
./clean.sh [OPTIONS]

Options:
  -c, --connection NAME    Snowflake CLI connection name (default: demo)
  -p, --prefix PREFIX      Environment prefix for resources
  --force, --yes, -y       Skip confirmation prompt
  -h, --help               Show help message

Examples:
  ./clean.sh                 # Interactive cleanup
  ./clean.sh --force         # Non-interactive cleanup
  ./clean.sh -c prod --force # Use 'prod' connection
```

## Streamlit Dashboard

The dashboard provides three pages for different user personas:

### Executive Overview (Main Page)

- **KPI Cards**: Total defects, PCBs inspected, defect rate, defect types
- **Pareto Chart**: Defect distribution by class with cumulative percentage
- **Factory Heatmap**: Defect counts by factory line and defect type
- **Trend Analysis**: Defect trends over time by class

### Vision Lab

- **Image Analysis**: Browse images from Snowflake stage or upload new PCB images
- **Defect Detection**: Run YOLOv12 inference on selected images
- **AI Guidance**: RAG-powered chatbot for defect remediation
  - Query Manuals (Cortex Search): IPC standards and repair procedures
  - Query Data (Cortex Analyst): Defect analytics and trends
- **Defect Reference**: Quick reference for all 6 defect classes with severity levels

### About

- **Business Value**: Executive overview of ROI and expected outcomes
- **Technical Architecture**: Data flow diagram and component details
- **Data Sources**: Internal tables, external datasets, and model outputs
- **Licenses & Attributions**: YOLOv12 and Deep PCB dataset citations

## Snowflake Resources

The deployment creates the following resources:

| Resource | Name |
|----------|------|
| Database | `CV_DEFECT_DETECTION_YOLO` |
| Schema | `CV_DEFECT_DETECTION_YOLO` |
| Role | `CV_DEFECT_DETECTION_YOLO_ROLE` |
| Warehouse | `CV_DEFECT_DETECTION_YOLO_WH` |
| Compute Pool | `CV_DEFECT_DETECTION_YOLO_COMPUTE_POOL` |
| External Access | `CV_DEFECT_DETECTION_YOLO_EXTERNAL_ACCESS` |
| Notebook | `CV_DEFECT_DETECTION_YOLO_NOTEBOOK` |
| Streamlit App | `CV_DEFECT_DETECTION_YOLO_DASHBOARD` |

### Database Objects

| Object | Description |
|--------|-------------|
| `MODEL_STAGE` | Stage for Deep PCB dataset and trained models |
| `NOTEBOOKS` | Stage for notebook files |
| `PCB_METADATA` | Table tracking PCB boards and factory lines |
| `DEFECT_LOGS` | Table storing inference results |
| `DEFECT_SUMMARY` | View aggregating defects by class |
| `DAILY_DEFECT_TRENDS` | View for time-series analysis |
| `FACTORY_LINE_DEFECTS` | View for factory performance |

## Technology Stack

- **Compute**: Snowflake Notebooks on Container Runtime with GPU (`GPU_NV_M`)
- **Model**: YOLOv12 (Ultralytics) - Attention-centric real-time object detector
- **Framework**: PyTorch
- **Dashboard**: Streamlit in Snowflake
- **Visualization**: Plotly, Altair
- **AI Features**: Cortex Search (RAG), Cortex Analyst (structured data)

## Project Structure

```
├── deploy.sh                    # Deployment script
├── run.sh                       # Execution and operations script
├── clean.sh                     # Cleanup script
├── notebooks/
│   ├── pcb_defect_detection.ipynb   # YOLOv12 training notebook
│   └── environment.yml              # Notebook conda environment
├── streamlit/
│   ├── streamlit_app.py         # Main dashboard (Executive Overview)
│   ├── snowflake.yml            # Streamlit deployment config
│   ├── environment.yml          # Streamlit conda environment
│   ├── pages/
│   │   ├── 1_Vision_Lab.py      # Interactive inference page
│   │   └── 2_About.py           # Documentation page
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py       # Data loading utilities
│   │   └── query_registry.py    # SQL query definitions
│   └── images/
│       ├── architecture.svg     # Architecture diagram
│       └── logo.svg             # App logo
├── sql/
│   ├── 01_account_setup.sql     # Account-level setup (roles, compute pool, EAI)
│   └── 02_schema_setup.sql      # Schema-level setup (tables, stages, views)
├── scripts/
│   └── update_notebook.py       # Notebook update utility
├── weights/
│   └── yolo12n.pt               # Pretrained YOLOv12 weights
├── data/
│   └── DeepPCB/                 # Deep PCB dataset (git sparse clone)
└── DRD.md                       # Design Requirements Document
```

## Licenses & Attributions

### YOLOv12

This project uses **YOLOv12** for object detection, which is licensed under the **[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html)** license.

- **Original Repository**: [github.com/sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)
- **Ultralytics Implementation**: [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Documentation**: [docs.ultralytics.com/models/yolo12](https://docs.ultralytics.com/models/yolo12/)

If you use YOLOv12 in your research, please cite the original work by University at Buffalo and the University of Chinese Academy of Sciences:

```bibtex
@article{tian2025yolo12,
  title={YOLO12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}

@software{yolo12,
  author = {Tian, Yunjie and Ye, Qixiang and Doermann, David},
  title = {YOLO12: Attention-Centric Real-Time Object Detectors},
  year = {2025},
  url = {https://github.com/sunsmarterjie/yolov12},
  license = {AGPL-3.0}
}
```

### Deep PCB Dataset

Training data is from the **Deep PCB** dataset, licensed under the **[MIT](https://opensource.org/licenses/MIT)** license:

- **Repository**: [github.com/tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)
- **Paper**: [arXiv:1902.06197](https://arxiv.org/abs/1902.06197)

```bibtex
@article{tang2019online,
  title={Online PCB Defect Detector On A New PCB Defect Dataset},
  author={Tang, Sanli and He, Fan and Huang, Xiaolin and Yang, Jie},
  journal={arXiv preprint arXiv:1902.06197},
  year={2019}
}
```

## Resources

- [Snowflake Notebooks Documentation](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks)
- [Container Runtime Overview](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview)
- [GPU Compute Pools](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/compute-pool)
- [Ultralytics YOLO12 Documentation](https://docs.ultralytics.com/models/yolo12/)
- [Deep PCB Dataset](https://github.com/tangsanli5201/DeepPCB)
