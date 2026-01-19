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
- **Distributed Training**: Uses `PyTorchDistributor` from Snowflake ML for scalable GPU training
- **6 Defect Classes**: Detect open, short, mousebite, spur, copper, and pin-hole defects
- **Secure Data Processing**: Images never leave the Snowflake platform
- **Automatic Data Acquisition**: Notebook clones Deep PCB dataset from GitHub and caches to stage
- **Interactive Dashboard**: 3-page Streamlit application with executive analytics, vision lab, and documentation
- **AI Guidance (Demo)**: Placeholder UI for Cortex Search/Analyst integration (hardcoded responses for demonstration)

> **Current Limitations**: The Vision Lab inference and Cortex RAG features display demo/placeholder responses. Production deployment requires training the model first and deploying an SPCS inference service.

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

- **Image Browsing**: Browse PCB images from `@MODEL_STAGE/raw/deeppcb/` or upload new images
- **Image Upload**: Upload custom PCB images for analysis
- **Defect Detection (Demo Mode)**: Displays simulated detection results
- **AI Guidance (Demo Mode)**: Hardcoded responses demonstrating Cortex Search/Analyst integration patterns
  - Query Manuals: Shows example IPC standard guidance
  - Query Data: Shows example analytics response
- **Defect Reference**: Quick reference for all 6 defect classes with severity levels

> **Note**: The Vision Lab currently operates in demo mode. To enable real-time inference:
> 1. Train the model by running `./run.sh main`
> 2. Deploy an SPCS inference service with the trained model
> 3. Update `1_Vision_Lab.py` to call the inference service endpoint

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
| `PCB_METADATA` | Table tracking PCB boards and factory lines (sample data) |
| `DEFECT_LOGS` | Table storing inference results |
| `PCB_LABELED_DATA` | Table created by notebook for distributed training (images + labels) |
| `DEFECT_SUMMARY` | View aggregating defects by class |
| `DAILY_DEFECT_TRENDS` | View for time-series analysis |
| `FACTORY_LINE_DEFECTS` | View for factory performance |

## Data Acquisition

The training data is acquired automatically when the notebook runs:

1. **First Run**: The notebook uses `git sparse-checkout` to clone only the `PCBData/` folder from the [Deep PCB repository](https://github.com/tangsanli5201/DeepPCB)
2. **Caching**: Downloaded images are uploaded to `@MODEL_STAGE/raw/deeppcb/` for subsequent runs
3. **Subsequent Runs**: Data is downloaded from the Snowflake stage (faster than GitHub)

**Pretrained Weights**: The notebook expects YOLOv12 pretrained weights at `@MODEL_STAGE/weights/yolo12n.pt`. If deploying with `--only-data`, place `yolo12n.pt` in a local `weights/` folder before running `deploy.sh`.

### Stage Structure

```
@MODEL_STAGE/
├── raw/
│   └── deeppcb/           # Training images (auto-downloaded)
│       └── group*/        # Image groups from Deep PCB dataset
├── models/
│   └── yolov12_pcb/       # Trained model outputs
│       └── yolo_best.pt   # Best model weights
└── weights/
    └── yolo12n.pt         # Pretrained weights (manual upload)
```

## Training Architecture

The notebook uses Snowflake's `PyTorchDistributor` for distributed GPU training:

### Training Pipeline

1. **Data Preparation**: Images and labels are converted to YOLO format and loaded into the `PCB_LABELED_DATA` table
2. **Distributed Execution**: `PyTorchDistributor` launches training across GPU workers with data sharding
3. **Model Training**: Each worker receives a data shard via `ShardedDataConnector` and trains YOLOv12
4. **Model Persistence**: The best model weights are saved to `@MODEL_STAGE/models/yolov12_pcb/`

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_nodes` | 1 | Number of compute nodes |
| `num_workers_per_node` | 1 | GPU workers per node |
| `num_gpus` | 1 | GPUs per worker |
| `epochs` | 10 | Training epochs (increase for production) |
| `batch_size` | 16 | Images per batch |
| `imgsz` | 640 | Input image size |

To scale training, increase `num_nodes` or `num_workers_per_node` in the notebook's `PyTorchScalingConfig`.

## Technology Stack

- **Compute**: Snowflake Notebooks on Container Runtime with GPU (`GPU_NV_M`)
- **Distributed Training**: `PyTorchDistributor` from `snowflake-ml-python` for GPU-accelerated training
- **Model**: YOLOv12 (Ultralytics) - Attention-centric real-time object detector
- **Framework**: PyTorch with Ultralytics
- **Dashboard**: Streamlit in Snowflake
- **Visualization**: Plotly, Matplotlib
- **AI Features (Demo)**: Cortex Search and Cortex Analyst UI placeholders for future integration

## Project Structure

```
├── deploy.sh                    # Deployment script
├── run.sh                       # Execution and operations script
├── clean.sh                     # Cleanup script
├── create_user.sh               # User creation utility
├── notebooks/
│   ├── pcb_defect_detection.ipynb   # YOLOv12 training notebook
│   ├── environment.yml              # Notebook conda environment
│   └── snowflake.yml                # Snowflake notebook config
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
│   ├── 02_schema_setup.sql      # Schema-level setup (tables, stages, views)
│   └── 03_network_egress.sql    # Network egress configuration
├── solution_presentation/
│   ├── PCB_Defect_Detection_Overview.md  # Presentation slides content
│   └── images/                  # Presentation diagrams (SVG)
├── data/
│   └── .gitkeep                 # Placeholder (data downloaded at runtime)
└── DRD.md                       # Design Requirements Document
```

> **Note**: The Deep PCB dataset is automatically downloaded via git sparse clone when the notebook runs for the first time. Pretrained YOLOv12 weights (`yolo12n.pt`) must be downloaded separately and placed in a `weights/` folder before deployment if you want to skip downloading from HuggingFace at runtime.

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
