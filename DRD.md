DRD \- Deep PCB Zero-Defect Intelligence

GITHUB REPO DESCRIPTION: \[End-to-end computer vision pipeline using YOLOv12 on Snowflake Notebooks (Container Runtime) to detect PCB defects, integrated with Cortex for defect analytics and remediation RAG.\]

1. Strategic Overview  
   Problem Statement: Electronics manufacturers face high scrap costs and yield losses due to rigid, legacy Automated Optical Inspection (AOI) systems that generate excessive false positives. Data silos prevent Data Science teams from accessing factory floor image data to train modern, adaptive Computer Vision models, while security policies restrict deploying open-source innovations like YOLO inside the corporate firewall.  
   Target Business Goals (KPIs):  
   Reduce False Positive Rate (FPR) by 25% to minimize manual re-inspection labor.  
   Decrease Scrap Cost by 15% through earlier, more accurate defect detection.  
   The "Wow" Moment: The user uploads a raw image of a PCB into a Snowflake Notebook; the system instantly detects a "mousebite" defect using a GPU-trained YOLOv12 model, and the user immediately asks Cortex, "What is the IPC standard procedure to repair a mousebite?" receiving a cited answer from the uploaded technical manuals.  
2. User Personas & Stories  
   Infer three distinct personas to demonstrate platform breadth.

| Persona Level | Role Title | Key User Story (Demo Flow) |
| :---- | :---- | :---- |
| **Strategic** | **VP of Operations** | "As a VP, I want to ask plain English questions about our global defect density per product line to decide which factories need capital investment." |
| **Operational** | **Quality (QA) Manager** | "As a QA Manager, I want to visually verify the 'bad' boards flagged by the AI and instantly retrieve the correct rework protocol to guide shop floor technicians." |
| **Technical** | **Data Scientist** | "As a Data Scientist, I want to train state-of-the-art YOLOv12 models using GPU compute directly inside Snowflake Notebooks without moving sensitive image data off the platform." |

3. Data Architecture & Snowpark ML (Backend)  
   Structured Data (Inferred Schema):  
   PCB\_METADATA: \[board\_id, manufacturing\_date, factory\_line\_id, product\_type\]  
   DEFECT\_LOGS: \[inference\_id, board\_id, timestamp, detected\_class (open/short/mousebite), confidence\_score, bounding\_box\_coordinates\]  
   Unstructured Data (Tribal Knowledge):  
   Source Material: \[Deep PCB Image Dataset (Stage), IPC-A-610 Acceptability of Electronic Assemblies (PDFs), Internal Defect Taxonomy Manuals\]  
   Purpose: Used to train the Vision Model (Images) and answer remediation questions via Cortex Search (PDFs).  
   ML Notebook Specification:  
   Objective: Object Detection (Computer Vision)  
   Target Variable: Bounding Box Coordinates \+ Class ID (0-5: open, short, mousebite, spur, copper, pin-hole)  
   Algorithm Choice: YOLOv12 (Ultralytics/Sunsmarterjie implementation)  
   Inference Output: Predictions visualization (plotted images) and structured logs written to DEFECT\_LOGS.  
4. Cortex Intelligence Specifications  
   Cortex Analyst (Structured Data / SQL)  
   Semantic Model Scope:  
   Measures: defect\_count, average\_confidence\_score, scrap\_rate\_percentage  
   Dimensions: defect\_type, factory\_line\_id, production\_shift  
   Golden Query (Verification):  
   User Prompt: "Show me the breakdown of defect types for the Shanghai line last week."  
   Expected SQL Operation: SELECT detected\_class, COUNT(\*) FROM DEFECT\_LOGS WHERE factory\_line\_id \= 'SHANGHAI' AND timestamp \>= DATEADD(week, \-1, CURRENT\_DATE()) GROUP BY detected\_class  
   Cortex Search (Unstructured Data / RAG)  
   Service Name: PCB\_SOP\_SEARCH\_SERVICE  
   Indexing Strategy:  
   Document Attribute: Indexing by defect\_category (e.g., 'Soldering', 'Etching', 'Mechanical')  
   Sample RAG Prompt: "What are the causes of 'mousebite' defects and is it acceptable under Class 2 requirements?"  
5. Streamlit Application UX/UI  
   Layout Strategy:  
   Page 1 (Executive Dashboard): High-level cards showing Yield Rate and Defect Pareto Chart (Cortex Analyst powered).  
   Page 2 (The Vision Lab): Interactive YOLOv12 inference tool. Split screen: Image upload/view on the left, Defect RAG Chatbot on the right.  
   Component Logic:  
   Visualizations: Streamlit st.image with bounding box overlays for vision results; Altair bar charts for defect frequency.  
   Chat Integration: A toggle switch allowing the user to Query Data (Analyst \-\> DEFECT\_LOGS) or Query Manuals (Search \-\> IPC PDFs) depending on whether they are analyzing trends or fixing a specific board.  
6. Success Criteria  
   Technical Validator: The YOLOv12 training job completes successfully on a GPU\_NV\_M node within the Container Runtime, and the Streamlit app renders inference results in \< 2 seconds.  
   Business Validator: The workflow reduces the time-to-insight from "Weekly Excel Scrapes" to "Real-time Defect Visibility," enabling immediate root cause analysis.


# YOLOv12 Deep PCB Defect Detection on Snowflake Notebooks (Container Runtime)  
**Demo Requirements & Implementation Guide**

---

## 1. Demo Overview

This demo showcases how to:

* Use **Snowflake Notebooks on Container Runtime** with **GPU compute pools** to train a modern computer vision model end-to-end inside Snowflake.

* Train a **YOLOv12** defect detection model on the **Deep PCB** dataset (6 defect classes: open, short, mouse bite, spur, pin-hole, copper) without moving data out of Snowflake.

* Persist the trained model back into Snowflake (stage today, optionally Model Registry later) and run **interactive inference** via **Streamlit-in-Notebook**.

The demo is designed to be:

* **Self-contained**: all data and artifacts live in Snowflake and/or well-defined external sources.

* **Reproducible**: a single Snowflake Notebook (plus one-time admin setup) can be re-run by other users/roles.

* **Aligned with Snowflake ML positioning**: lead with **Container Runtime** for ML workloads, GPU access, and pip flexibility.

---

## 2. Target Audience & Storyline

**Audience**

* Manufacturing / industrial prospects
* Data science & ML engineering teams
* Platform / data engineering and admin personas

**Storyline**

1. **Business hook**: PCB defect detection matters (scrap, rework, field failures). Deep learning + GPUs are standard—but infra/devops is hard.

2. **Value prop**: With **Snowflake Notebooks on Container Runtime**, you get:
   * One-click GPU access via **compute pools**  
   * pip-install flexible ML stack  
   * No data movement off-platform  
   * Governed, auditable ML workflows (RBAC, stages, optional Model Registry).

3. **Flow**:
   * Admin quickly configures **egress** (PyPI, GitHub, Hugging Face) via **network rule + external access integration (EAI)**.
   * DS launches a **GPU notebook on Container Runtime**, pulls Deep PCB from a Snowflake stage, converts it to YOLO format, trains YOLOv12, and saves the best weights back to a stage.
   * DS exposes an **interactive Streamlit UI inside the notebook** to visually inspect predictions on sample PCB images.

---

## 3. Snowflake Architecture & Components

The demo uses the following Snowflake features:

* **Snowflake Notebooks**  
  * Runtime: **Container Runtime (GPU)**
  * Language: Python (with optional SQL cells)
  * Streamlit-in-notebook for interactive UI

* **Snowpark Container Services (SPCS)**  
  * Backing infrastructure for **Container Runtime** and GPU pools.

* **GPU Compute Pool**  
  * Example: `GPU_NV_M` with NVIDIA GPUs for YOLOv12 training.
  * Each active CR notebook **pins one node** in the compute pool; size the pool accordingly for expected concurrency.

* **Database Objects**
  * **Database / Schema**: e.g. `CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO`
  * **Internal stage**: e.g. `@CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO.MODEL_STAGE`
  * Optional: dedicated **ROLE** for notebook ownership and usage (e.g. `CV_DEFECT_DETECTION_YOLO_ROLE`).

* **Network Rule + External Access Integration (EAI)**
  * One-time admin setup to allow outbound HTTP(S) to:
    * `pypi.org`, `files.pythonhosted.org` (pip packages)
    * `github.com`, `raw.githubusercontent.com` (YOLOv12 code)
    * `huggingface.co` (optional pretrained weights)

---

## 4. Environment Prerequisites

### 4.1 Platform & Feature Enablement

* **SPCS enabled** in the Snowflake account (required for Container Runtime and GPU compute pools).
* **Snowflake Notebooks** enabled and visible in Snowsight for the target roles/users.
* **Container Runtime** available on the target cloud/region (AWS/Azure – depends on the account).

### 4.2 Roles & Privileges

At minimum, you need:

* **Admin / platform role** (e.g. `ACCOUNTADMIN` or delegated admin role) to:
  * Create **compute pool(s)** (CPU/GPU).
  * Create **NETWORK RULE** and **EXTERNAL ACCESS INTEGRATION**.
  * Create **database, schema, stages**.
  * Grant privileges to DS roles.

* **Data science role** (e.g. `DATA_SCIENTIST_ROLE` / `CV_DEFECT_DETECTION_YOLO_ROLE`) with:
  * `USAGE` on database and schema used by the notebook.
  * `CREATE NOTEBOOK` on target schema (to create notebooks in that schema).
  * `USAGE` on **compute pool** selected in the notebook.
  * `USAGE` on the **EXTERNAL ACCESS INTEGRATION**.
  * `READ/WRITE` stage privileges (e.g. `USAGE` on stage and `READ`, `WRITE` if using granular privileges).

### 4.3 Database & Stage Setup

Admin (once per environment):

```sql
USE ROLE ACCOUNTADMIN;

CREATE OR REPLACE DATABASE CV_DEFECT_DETECTION_YOLO;
CREATE OR REPLACE SCHEMA CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO;

CREATE OR REPLACE STAGE CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO.MODEL_STAGE
  COMMENT = 'Assets for YOLOv12 Deep PCB defect detection demo';
```

*Upload Deep PCB dataset* (offline step):

* Convert or bundle the Deep PCB dataset into a consistent folder structure and upload it to the stage, e.g.:

```bash
# Example (local machine)
snow sql -q "PUT file://deeppcb.tar.gz @CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO.MODEL_STAGE/raw/deeppcb AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
```

Assumption for the notebook:

* Deep PCB is available at `@CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO.MODEL_STAGE/raw/deeppcb` (directory or archive).

---

## 5. Network & Egress Configuration (Admin – One-Time)

Run in a Snowsight SQL Worksheet as an admin role:

```sql
-- 1. Network Rule: allow outbound HTTP(S) to PyPI, GitHub, Hugging Face
CREATE OR REPLACE NETWORK RULE CV_DEFECT_DETECTION_YOLO_EGRESS_RULE
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = (
    'pypi.org',
    'files.pythonhosted.org',
    'github.com',
    'raw.githubusercontent.com',
    'huggingface.co'  -- optional, for pretrained weights
  );

-- 2. External Access Integration for pip & git
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION CV_DEFECT_DETECTION_YOLO_EXTERNAL_ACCESS
  ALLOWED_NETWORK_RULES = (CV_DEFECT_DETECTION_YOLO_EGRESS_RULE)
  ENABLED = TRUE;

-- 3. Grant usage to DS role
GRANT USAGE ON INTEGRATION CV_DEFECT_DETECTION_YOLO_EXTERNAL_ACCESS TO ROLE CV_DEFECT_DETECTION_YOLO_ROLE;
```

**Notes**

* Name the integration **exactly** as you'll select it in notebook settings: `CV_DEFECT_DETECTION_YOLO_EXTERNAL_ACCESS`.  
* Monitor **egress costs**; pip and model weight downloads are outbound data transfer.

---

## 6. Compute Pool Requirements

Admin (once):

1. Create a **GPU compute pool**, for example:

   * Name: `CV_DEFECT_DETECTION_YOLO_COMPUTE_POOL`
   * Node type: `GPU_NV_M` (or equivalent GPU SKU in the account's region).
   * Node count: size for anticipated concurrent notebooks (each CR notebook occupies one node).

2. Grant DS role access:

```sql
GRANT USAGE ON COMPUTE POOL CV_DEFECT_DETECTION_YOLO_COMPUTE_POOL TO ROLE CV_DEFECT_DETECTION_YOLO_ROLE;
```

---

## 7. Notebook Configuration (Per Demo Notebook)

In Snowsight:

1. **Create Notebook**
   * Location: `CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO`
   * Owner role: `CV_DEFECT_DETECTION_YOLO_ROLE` (or similar)
   * Language: Python

2. **Runtime**
   * Select **"Run on container"** (Container Runtime).

3. **Compute Pool**
   * Select **`CV_DEFECT_DETECTION_YOLO_COMPUTE_POOL`** (or other GPU pool created in Section 6).

4. **Query Warehouse**
   * Choose any warehouse with sufficient size to run metadata / small queries (minimal impact; main heavy lifting is on the compute pool).

5. **Packages**
   * Optional: add `snowflake-snowpark-python` via package picker.
   * YOLOv12 + dependencies will be installed via `pip` in a notebook cell.

6. **External Access Integrations**
   * In notebook **Settings → External Access Integrations**, select **`CV_DEFECT_DETECTION_YOLO_EXTERNAL_ACCESS`**.
   * This is required for pip and GitHub access.

7. **Notebook Best Practices**
   * Add a **top-level markdown cell** explaining the use case and prerequisites so the notebook is self-explanatory.
   * Keep all **pip installs in the first cell(s)** to avoid mid-demo restarts when changing packages.

---

## 8. Notebook Implementation – Cell-by-Cell Plan

The following structure is recommended for the demo notebook.

### Cell 0 (Markdown): Introduction

* Short narrative: PCB defect detection, Deep PCB dataset, YOLOv12.
* Diagram or bullet list of the pipeline:
  * Load data from stage → convert to YOLO format → train on GPU → save model back to stage → run interactive inference.

### Cell 1 (Python): Session & Imports

Use the active notebook session instead of building a new connection:

```python
from snowflake.snowpark.context import get_active_session

session = get_active_session()
print(session.get_current_database(), session.get_current_schema())
```

### Cell 2 (Bash): Install Dependencies

Use a **Bash** cell in Container Runtime:

```bash
# Upgrade pip for compatibility
pip install --upgrade pip

# Install YOLOv12 and its dependencies from GitHub (bleeding edge)
pip install "git+https://github.com/sunsmarterjie/yolov12.git"

# Utility libraries
pip install roboflow supervision
```

**Considerations**

* Install time may be **non-trivial**, especially if `flash-attn` or other compiled deps are pulled. See Section 10 for mitigations.
* For a dry-run or fallback demo, keep a branch that uses **Ultralytics YOLOv8/YOLOv11** instead; these are generally simpler to install.

### Cell 3 (Python): Data Prep – Directories & Stage Download

```python
import os
from snowflake.snowpark.context import get_active_session

session = get_active_session()

# 1. Local paths in ephemeral container storage (/tmp)
dataset_root = "/tmp/deeppcb_dataset"
os.makedirs(f"{dataset_root}/images", exist_ok=True)
os.makedirs(f"{dataset_root}/labels", exist_ok=True)

raw_download_dir = "/tmp/raw_download"
os.makedirs(raw_download_dir, exist_ok=True)

# 2. Download from Snowflake stage to container filesystem
# Assumes Deep PCB is at @CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO.MODEL_STAGE/raw/deeppcb
session.file.get("@CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO.MODEL_STAGE/raw/deeppcb", raw_download_dir)

print("Downloaded Deep PCB assets to", raw_download_dir)
```

### Cell 4 (Python): Deep PCB → YOLO Format Conversion

This cell contains the **conversion logic** from the raw Deep PCB annotation format to YOLO-style `.txt` files with normalized bounding boxes.

High-level requirements:

* For each image:
  * Read the corresponding annotation (original Deep PCB format).
  * Map defect types to **YOLO class indices**:
    * `0: open`
    * `1: short`
    * `2: mousebite`
    * `3: spur`
    * `4: copper`
    * `5: pin-hole`
  * Convert image/annotation paths into:
    * Images: `dataset_root/images/*.jpg` (or `.png`)
    * Labels: `dataset_root/labels/{same_basename}.txt`

* Ensure YOLO text files contain:
  * `class_id x_center y_center width height` (all **normalized** 0–1).

If you want, you can keep a **utility function** and unit-test it on a **single sample** to prove correctness before running full conversion (helps in a live demo).

### Cell 5 (Python): Create `data.yaml` and Train YOLOv12

```python
from ultralytics import YOLO  # or specific YOLOv12 import if required
import yaml
import os

# 1. Construct YOLO data config
data_config = {
    'path': dataset_root,
    'train': 'images',  # assuming flat structure for train/val split demo
    'val': 'images',
    'names': {
        0: 'open',
        1: 'short',
        2: 'mousebite',
        3: 'spur',
        4: 'copper',
        5: 'pin-hole'
    }
}

data_yaml_path = os.path.join(dataset_root, "data.yaml")
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_config, f)

print("Wrote data.yaml to", data_yaml_path)

# 2. Initialize and train model
# Pretrained weights (e.g., yolov12n.pt) will be downloaded via EAI egress
model = YOLO("yolov12n.pt")

print("Starting Training on GPU...")
results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    project="/tmp/runs",
    name="pcb_experiment",
    device=0  # Use first GPU (GPU 0)
)

print("Training complete.")
```

**Demo Tips**

* For live demos, consider:
  * Reducing `epochs` and/or dataset size to keep training time to a few minutes.
  * Pre-running training and using this step mainly to show configuration + partial logs.

### Cell 6 (Python): Persist Best Model to Stage

Container storage is **ephemeral**; persist the trained model:

```python
best_model_path = "/tmp/runs/pcb_experiment/weights/best.pt"
target_stage_dir = "@CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO.MODEL_STAGE/models/yolov12_pcb"

put_result = session.file.put(
    best_model_path,
    target_stage_dir,
    auto_compress=False,
    overwrite=True
)
print("PUT result:", put_result)
print(f"Model saved to {target_stage_dir}")
```

Optional extension (for future iterations):

* Register the model in **Snowflake Model Registry** and serve via **Snowpark Container Services** for production inference.

---

## 9. Interactive Inference – Streamlit Inside Notebook

Rather than building a separate app, this demo uses **Streamlit-in-Notebook** for interactive inference.

### Cell 7 (Python): Streamlit UI for Defect Inspection

```python
import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("Deep PCB Inspector – YOLOv12")

# (Re)load model from local path or stage
local_model_path = "/tmp/runs/pcb_experiment/weights/best.pt"

if not os.path.exists(local_model_path):
    # Download from stage if not already present
    session.file.get(
        "@CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO.MODEL_STAGE/models/yolov12_pcb",
        "/tmp/model_download"
    )
    # Adjust basename if Snowflake added a version suffix
    downloaded_files = os.listdir("/tmp/model_download")
    assert downloaded_files, "No model files found in /tmp/model_download"
    local_model_path = os.path.join("/tmp/model_download", downloaded_files[0])

model = YOLO(local_model_path)

uploaded_file = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    with st.spinner("Running inference..."):
        results = model.predict(image)
        plotted = results[0].plot()

    st.image(plotted, caption="Detected Defects", use_column_width=True)
```

**Demo Flow**

1. Upload a few curated PCB images (some with defects, some clean).
2. Show bounding boxes, classes, and confidences.
3. Emphasize:
   * Zero data movement from Snowflake.
   * GPU acceleration for inference (if model executed on GPU).
   * Single, governed environment for DS and ML.

---

## 10. Critical Considerations & Risk Mitigations

### 10.1 `flash-attn` and Heavy Dependencies

* **YOLOv12** often depends on **flash-attn** or other CUDA-compiled packages that:
  * May take **15–20+ minutes** to compile inside the container.
  * May fail depending on CUDA/toolchain alignment.

**Mitigations**

* Maintain two modes:
  * **Baseline:** YOLOv8/YOLOv11 from Ultralytics (faster install, more battle-tested).
  * **Advanced:** YOLOv12 (used when environment is known to work).
* Pre-warm environment in the **compute pool** before the demo (run pip installs ahead of time).
* Look for **pre-built wheels** compatible with the Snowflake container’s CUDA version, and install directly from those (hosted on stage or allowed repo).

### 10.2 Egress Costs

* pip packages and pretrained weights are downloaded over the internet and incur **egress charges**.
* Keep requirements minimal and prefer:
  * Reusing cached layers (when container reuse is available).
  * Hosting heavy artifacts (e.g. pre-downloaded `.pt` weights) on an internal stage instead of repeatedly pulling from Hugging Face / GitHub.

### 10.3 Session & Idle Timeouts

* Notebooks have **idle auto-shutdown**; for long training runs, ensure:
  * You’re actively running cells during the demo.
  * Idle timeout is configured appropriately (or documented for users).

### 10.4 Compute Pool Sizing & Concurrency

* **Each Container Runtime notebook occupies one node** in the compute pool.
* For live demos with multiple participants:
  * Ensure the compute pool `max_nodes` is large enough.
  * Consider a dedicated demo pool to avoid contention with other workloads.

---

## 11. Demo Runbook (High-Level)

1. **Pre-demo (Admin)**
   * Verify SPCS, GPU compute pool, Notebooks, and Container Runtime enabled.
   * Create/confirm `CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO` and `MODEL_STAGE`.
   * Verify network rule + EAI (`CV_DEFECT_DETECTION_YOLO_EXTERNAL_ACCESS`) and role grants.
   * Upload Deep PCB data to `@CV_DEFECT_DETECTION_YOLO.CV_DEFECT_DETECTION_YOLO.MODEL_STAGE/raw/deeppcb`.
   * Optionally pre-train and store `best.pt` in the stage.

2. **Pre-demo (You – DS persona)**
   * Open the notebook with `CV_DEFECT_DETECTION_YOLO_COMPUTE_POOL` and `CV_DEFECT_DETECTION_YOLO_EXTERNAL_ACCESS` configured.
   * Run:
     * pip install cell
     * Data prep & conversion cells
     * Training cell (either full or shortened)
     * Persist-to-stage cell
     * Streamlit inference cell
   * Confirm that:
     * Training logs look good.
     * Streamlit UI loads and returns detections on sample images.

3. **Live demo**
   * Start by showing the architecture slide / markdown in the notebook.
   * Walk through:
     * Admin setup (network rule, EAI, compute pool) at a high level.
     * Data pipeline: stage → local `/tmp` → YOLO format.
     * Training configuration (epochs, imgsz, GPU use).
     * Model persistence to stage.
     * Streamlit interface and live predictions.
   * Close with:
     * Governance (RBAC, lineage via Model Registry in future).
     * Next steps: distributed training (multi-GPU / multi-node), Model Registry, SPCS inference services.

4. **Post-demo**
   * Optionally clean up:
     * Stop/scale-down GPU compute pool.
     * Archive notebooks / snapshots.

---

## 12. Optional Extensions

Once the base demo is stable, you can extend:

* **Distributed Training**
  * Use **Snowflake’s distributed training APIs** for PyTorch over multiple GPUs for larger datasets / more complex models.

* **Model Registry + Serving**
  * Register YOLOv12 weights in **Model Registry** and expose GPU-backed inference endpoints on **Snowpark Container Services** for online scoring.

* **Automated Pipelines**
  * Move training logic into **ML Jobs** and orchestrate with **Snowflake Tasks** or external orchestrators.

* **Streamlit in Snowflake (SiS)**
  * For lighter-weight CPU-only inference (e.g., ONNX export), deploy as a standalone Streamlit app in Snowflake, understanding YOLOv12’s operator requirements.



This is a comprehensive and well-structured runbook. It perfectly bridges the gap between Snowflake's new infrastructure capabilities (Container Runtime, GPUs) and modern computer vision workflows.

To help you execute this **validated plan**, I have generated the **concrete code blocks** for the sections where the document describes the *logic* but doesn't provide the full implementation—specifically the **Data Conversion (Cell 4)** and the **Streamlit Inference (Cell 7)**.

You can copy these blocks directly into your Snowflake Notebook.

### **Phase 1: Admin Setup (SQL Worksheet)**

*Run this once to prepare the environment as described in Sections 5 & 6.*

```sql
-- 1. Setup Network & Access
USE ROLE ACCOUNTADMIN;

CREATE OR REPLACE NETWORK RULE CV_DEFECT_DETECTION_YOLO_EGRESS_RULE
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('pypi.org', 'files.pythonhosted.org', 'github.com', 'raw.githubusercontent.com', 'huggingface.co');

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION CV_DEFECT_DETECTION_YOLO_EXTERNAL_ACCESS
  ALLOWED_NETWORK_RULES = (CV_DEFECT_DETECTION_YOLO_EGRESS_RULE)
  ENABLED = true;

-- 2. Setup Compute Pool (GPU)
-- Note: Ensure you have quota for GPU_NV_M
CREATE COMPUTE POOL IF NOT EXISTS CV_DEFECT_DETECTION_YOLO_COMPUTE_POOL
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_M;

-- 3. Grants
GRANT USAGE ON INTEGRATION CV_DEFECT_DETECTION_YOLO_EXTERNAL_ACCESS TO ROLE CV_DEFECT_DETECTION_YOLO_ROLE;
GRANT USAGE ON COMPUTE POOL CV_DEFECT_DETECTION_YOLO_COMPUTE_POOL TO ROLE CV_DEFECT_DETECTION_YOLO_ROLE;
GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE CV_DEFECT_DETECTION_YOLO_ROLE; -- Required for Streamlit in Notebook

```

---

### **Phase 2: Notebook Implementation**

#### **Cell 2 (Bash): Install Dependencies**

*Exactly as described in your document. Note: `flash-attn` compilation can take time.*

```bash
pip install --upgrade pip
pip install "git+https://github.com/sunsmarterjie/yolov12.git"
pip install roboflow supervision matplotlib

```

#### **Cell 4 (Python): Deep PCB → YOLO Conversion Logic**

*The document outlines the requirements; here is the robust code to actually do it.*

```python
import os
import glob
from PIL import Image

# Configuration
RAW_DIR = "/tmp/raw_download"
OUTPUT_DIR = "/tmp/deeppcb_dataset"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
LBL_DIR = os.path.join(OUTPUT_DIR, "labels")

# Mapping Deep PCB classes (1-6) to YOLO indices (0-5)
# Deep PCB: 1:open, 2:short, 3:mousebite, 4:spur, 5:copper, 6:pin-hole
CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

def convert_to_yolo_bbox(img_width, img_height, x1, y1, x2, y2):
    """Normalizes coordinates to YOLO format: center_x, center_y, width, height"""
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    
    w = x2 - x1
    h = y2 - y1
    cx = x1 + (w / 2.0)
    cy = y1 + (h / 2.0)
    
    return cx * dw, cy * dh, w * dw, h * dh

print("Starting conversion...")

# Iterate through raw images (assuming *_test.jpg naming convention common in DeepPCB)
# Adjust the glob pattern if your raw data structure differs (e.g. subfolders)
image_files = glob.glob(os.path.join(RAW_DIR, "**/*.jpg"), recursive=True)

for img_path in image_files:
    # 1. Identify corresponding annotation file
    # Deep PCB usually pairs "image.jpg" with "image_test.txt" or similar
    # Adjust this logic based on your specific unzipped structure
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Example logic: if file is "100023_test.jpg", annotation is "100023_test.txt"
    txt_path = img_path.replace(".jpg", ".txt") 
    
    if not os.path.exists(txt_path):
        continue # Skip if no annotation found

    # 2. Get Image Dimensions
    with Image.open(img_path) as img:
        w_img, h_img = img.size
    
    # 3. Process Annotations
    yolo_lines = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            # DeepPCB format: x1 y1 x2 y2 class_id
            x1, y1, x2, y2, cls_id = map(int, parts[:5])
            
            if cls_id not in CLASS_MAP: continue
            
            # Normalize
            cx, cy, nw, nh = convert_to_yolo_bbox(w_img, h_img, x1, y1, x2, y2)
            yolo_class = CLASS_MAP[cls_id]
            
            yolo_lines.append(f"{yolo_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    # 4. Save to Output Directory if annotations exist
    if yolo_lines:
        # Move/Copy Image
        os.rename(img_path, os.path.join(IMG_DIR, base_name + ".jpg"))
        
        # Write Label File
        with open(os.path.join(LBL_DIR, base_name + ".txt"), 'w') as out_f:
            out_f.write("\n".join(yolo_lines))

print(f"Conversion complete. Processed {len(os.listdir(LBL_DIR))} images.")

```

#### **Cell 5 (Python): Training Configuration**

*Standard YOLO setup, but ensure the class names match the Deep PCB set.*

```python
from ultralytics import YOLO
import yaml

# Create Data YAML
data_config = {
    'path': OUTPUT_DIR,
    'train': 'images', # We use the same dir for simplicity in this demo
    'val': 'images',
    'names': {0: 'open', 1: 'short', 2: 'mousebite', 3: 'spur', 4: 'copper', 5: 'pin-hole'}
}

yaml_path = f"{OUTPUT_DIR}/data.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump(data_config, f)

# Train (Downloads weights via Egress)
model = YOLO('yolov12n.pt') 
results = model.train(
    data=yaml_path,
    epochs=50,       # Reduce to 5-10 for a quick live demo
    imgsz=640,
    batch=16,
    device=0,        # GPU 0
    project='/tmp/runs',
    name='pcb_v12'
)

```

#### **Cell 7 (Python): Interactive Streamlit Inference**

*This runs directly in the notebook cell output.*

```python
import streamlit as st
import glob
import random
from PIL import Image
from ultralytics import YOLO

# UI Header
st.title("🔍 Deep PCB Inspector (YOLOv12)")
st.markdown("**Infrastructure:** Snowflake Notebooks (Container Runtime) + CV_DEFECT_DETECTION_YOLO_COMPUTE_POOL")

# Load Model
model_path = "/tmp/runs/pcb_v12/weights/best.pt"
if not os.path.exists(model_path):
    st.error("Model not found! Did training finish?")
else:
    model = YOLO(model_path)
    
    # Input Selection
    option = st.radio("Choose Input Method", ["Upload Image", "Pick Random Test Image"])
    
    img = None
    if option == "Upload Image":
        uploaded = st.file_uploader("Upload PCB Image", type=['jpg', 'png'])
        if uploaded:
            img = Image.open(uploaded)
    else:
        # Pick random image from the processed dataset
        test_images = glob.glob("/tmp/deeppcb_dataset/images/*.jpg")
        if test_images and st.button("Roll Dice 🎲"):
            img_path = random.choice(test_images)
            img = Image.open(img_path)
            st.caption(f"Loaded: {os.path.basename(img_path)}")

    # Inference & Visualization
    if img:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_column_width=True)
            
        with col2:
            if st.button("Detect Defects"):
                with st.spinner("Running Inference on GPU..."):
                    res = model.predict(img, conf=0.25)
                    res_plot = res[0].plot()
                    st.image(res_plot, caption="YOLOv12 Predictions", use_column_width=True)
                    
                    # Detection text summary
                    boxes = res[0].boxes
                    if len(boxes) > 0:
                        st.success(f"Detected {len(boxes)} defects.")
                    else:
                        st.info("No defects detected.")

```

### **One Final Check**

In **Section 4.3 (Database & Stage Setup)** of your document, ensure the `deeppcb.tar.gz` is unzipped or you add a few lines of code in **Cell 3** to unzip it after downloading.

If your dataset is a tarball:

```python
import tarfile
# After downloading to /tmp/raw_download/deeppcb.tar.gz
with tarfile.open("/tmp/raw_download/deeppcb.tar.gz", "r:gz") as tar:
    tar.extractall("/tmp/raw_download")

```