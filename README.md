# 🛣️ Pothole Detection & Road Condition Mapping System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-red.svg)](https://flask.palletsprojects.com/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0%2B-orange.svg)](https://www.mysql.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-oriented Computer Vision and Geospatial Intelligence system that ingests vehicle/dashcam video footage, detects road potholes using **Ultralytics YOLOv8**, extracts spatio-temporal telemetry (**GPS coordinates & ISO 6709 timestamps**), estimates surface distress severity, deduplicates multi-frame observations, persists structured records in **MySQL**, exposes operations via a modular **Flask REST API**, and renders an interactive **Streamlit Dashboard** featuring **Folium** maps and analytical charts.

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#executive-summary--problem-statement)
2. [End-to-End System Architecture](#end-to-end-system-architecture)
3. [Key Engineering Features](#key-engineering-features)
4. [Technology Stack](#technology-stack)
5. [Repository Structure](#repository-structure)
6. [Dataset Strategy & YOLOv8 Annotation](#dataset-strategy--yolov8-annotation)
7. [Model Training & Empirical Evaluation](#model-training--empirical-evaluation)
8. [Pipeline Components & Algorithms](#pipeline-components--algorithms)
   - [A. Computer Vision Inference Engine](#a-computer-vision-inference-engine)
   - [B. Rule-Based Severity Classifier](#b-rule-based-severity-classifier)
   - [C. GPS Telemetry Extractor with Multi-Tier Fallback](#c-gps-telemetry-extractor-with-multi-tier-fallback)
   - [D. Spatio-Temporal Detection Aggregator](#d-spatio-temporal-detection-aggregator)
9. [MySQL Relational Schema](#mysql-relational-schema)
10. [Flask REST API Documentation](#flask-rest-api-documentation)
11. [Streamlit Interactive Web Dashboard](#streamlit-interactive-web-dashboard)
12. [Installation & Setup Guide](#installation--setup-guide)
13. [Verification & Automated Test Suite](#verification--automated-test-suite)
14. [Limitations & Real-World Considerations](#limitations--real-world-considerations)
15. [Troubleshooting Guide](#troubleshooting-guide)
16. [Technical Interview Defense Guide](#technical-interview-defense-guide)

---

## 1. Executive Summary & Problem Statement

Road surface degradation (potholes, lateral cracking, cavity depressions) represents a major hazard for passenger safety, vehicular longevity, and municipal maintenance budgets. Traditional road inspection techniques rely heavily on manual surveys or citizen complaints, which are labor-intensive, reactive, and non-scalable.

This project delivers an **automated, local-first road quality audit pipeline** designed to run efficiently on standard consumer laptops (CPU or entry-level GPU) without requiring expensive third-party cloud APIs.

---

## 2. End-to-End System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                            │
│                                                                        │
│   Streamlit Web Dashboard (Multi-Page UI)                             │
│   ├── Home / System Overview                                          │
│   ├── Video Upload & Job Trigger (Page 1)                             │
│   ├── Detection & Road Condition Explorer (Page 2)                    │
│   └── Geospatial Map (Folium / OpenStreetMap) (Page 3)                │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │ HTTP / JSON REST
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                             API LAYER                                  │
│                                                                        │
│   Flask REST API Application                                           │
│   ├── Blueprint Routing (`/api/v1/videos`, `/api/v1/road-conditions`) │
│   ├── Request Validation & Exception Handling Middleware              │
│   └── Structured JSON Serializers                                      │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │ Calls Services
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        CORE APPLICATION LAYER                          │
│                                                                        │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌───────────────┐ │
│  │ Processing Pipeline  │  │ Pothole Detector     │  │ GPS Extractor │ │
│  │ Orchestrator         │  │ (Ultralytics YOLOv8) │  │ (Exif/FFprobe)│ │
│  └──────────┬───────────┘  └──────────┬───────────┘  └───────┬───────┘ │
│             │                         │                      │         │
│  ┌──────────▼───────────┐  ┌──────────▼───────────┐  ┌───────▼───────┐ │
│  │ Severity Classifier  │  │ Detection Aggregator │  │ Video Annotator││
│  │ (Rule-Based Engine)  │  │ (Centroid Tracking)  │  │ (OpenCV)      │ │
│  └──────────────────────┘  └──────────────────────┘  └───────────────┘ │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │ Repository Pattern
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA ACCESS & STORAGE                           │
│                                                                        │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────┐ │
│  │ MySQL Database                  │   │ Local File System Storage   │ │
│  │ ├── `videos`                    │   │ ├── data/raw/               │ │
│  │ ├── `detections`                │   │ ├── outputs/videos/         │ │
│  │ └── `road_conditions`           │   │ └── logs/app.log            │ │
│  └─────────────────────────────────┘   └─────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Engineering Features

- **Decoupled Architecture:** Clean separation of concerns between Computer Vision inference, Business logic, Storage repository, REST APIs, and UI layers.
- **Resilient GPS Extraction:** Extracts embedded ISO 6709 location tags via FFprobe / ExifTool with zero-crash fallback when GPS is missing or corrupted.
- **Spatio-Temporal Deduplication:** Prevents single physical potholes detected over 30 continuous frames from being counted 30 times in database audits.
- **Deterministic Severity Assessment:** Computes normalized surface footprint ratios ($A_{\text{bbox}} / A_{\text{frame}}$) to categorize damage into `LOW`, `MEDIUM`, and `HIGH`.
- **Relational Persistence:** 100% parameterized SQL transactions on normalized MySQL schemas with foreign keys and cascade rules.
- **Local Portability:** Runs seamlessly on standard Windows/Linux student laptops with automatic CPU/GPU fallback and configurable frame-skipping.
- **Strict Empirical Metrics:** Zero fabricated performance numbers. Model benchmarks, latency, and FPS are computed dynamically.

---

## 4. Technology Stack

| Domain | Technology | Justification |
| :--- | :--- | :--- |
| **Language** | Python 3.10 / 3.11 | De-facto standard across PyTorch, OpenCV, and data science ecosystems. |
| **Object Detection** | Ultralytics YOLOv8 (`yolov8n.pt`) | Anchor-free single-stage detector with optimal mAP-to-latency trade-off for CPU inference. |
| **Computer Vision** | OpenCV (`cv2`) | Optimized C++ video stream decoding, aspect-ratio letterboxing, and frame overlay rendering. |
| **Database** | MySQL 8.0+ (`PyMySQL`) | ACID-compliant relational storage with indexed foreign keys and connection pooling. |
| **Backend REST API** | Flask + Flask-CORS + Pydantic | Lightweight microservice routing with strict schema validation and error middleware. |
| **Frontend UI** | Streamlit | Rapid reactive Python UI with video streaming widgets and state management. |
| **Geospatial Mapping**| Folium & `streamlit-folium` | Zero-cost Leaflet.js interactive maps with marker clustering and HTML telemetry popups. |
| **Telemetry Parsing** | FFprobe / PyExifTool | Container-level inspection of ISO 6709 location strings without transcode overhead. |
| **Testing** | Pytest + Pytest-Mock | Unit and integration test coverage across all isolated modules. |

---

## 5. Repository Structure

```
pothole-detection-road-mapping/
│
├── .env.example                     # Template environment variables
├── .env                            # Local development settings (git-ignored)
├── .gitignore                       # Git exclusion rules
├── config.py                        # Validated, typed configuration loader
├── requirements.txt                 # Exact pinned dependency list
├── README.md                        # Master project documentation
│
├── app/                             # Core Python Package
│   ├── __init__.py
│   ├── api/                         # Flask REST API Layer
│   │   ├── __init__.py
│   │   ├── routes.py                # REST endpoints and application factory
│   │   └── schemas.py               # Pydantic validation schemas
│   ├── detection/                   # Computer Vision Sub-package
│   │   ├── __init__.py
│   │   ├── detector.py              # PotholeDetector class wrapping YOLOv8
│   │   ├── preprocessing.py         # Aspect-ratio letterboxing & normalization
│   │   └── postprocessing.py        # Video overlay renderer & HUD banner
│   ├── gps/                         # Geolocation Sub-package
│   │   ├── __init__.py
│   │   └── extractor.py             # Multi-tier GPS extractor with fallback
│   ├── severity/                    # Severity Classification Sub-package
│   │   ├── __init__.py
│   │   └── classifier.py            # Rule-based geometric severity classifier
│   ├── aggregation/                 # Multi-Frame Fusion Sub-package
│   │   ├── __init__.py
│   │   └── aggregator.py            # Centroid temporal deduplication engine
│   ├── database/                    # Persistence Sub-package
│   │   ├── __init__.py
│   │   ├── connection.py            # MySQL connection pool & transactions
│   │   ├── schema.sql               # Normalized DDL schema
│   │   └── repository.py            # Parameterized repository CRUD layer
│   ├── processing/                  # Pipeline Coordinator Sub-package
│   │   ├── __init__.py
│   │   └── pipeline.py              # Master VideoProcessingPipeline
│   └── utils/                       # Shared Utilities
│       ├── __init__.py
│       ├── logger.py                # Structured console + file logger
│       └── file_utils.py            # Path traversal protection & filename sanitizer
│
├── streamlit_app/                   # Frontend Dashboard
│   ├── app.py                       # Overview & system status page
│   ├── components/                  # Reusable dashboard widgets
│   │   ├── __init__.py
│   │   ├── map_view.py              # Folium Leaflet map renderer
│   │   └── charts.py                # Plotly severity & confidence charts
│   └── pages/                       # Multi-page dashboard views
│       ├── 1_Upload_and_Process.py  # Video ingestion & live progress tracker
│       ├── 2_Detections_Explorer.py # Frame-level & condition record explorer
│       └── 3_Road_Map.py            # Interactive GIS map visualization
│
├── dataset/                         # YOLOv8 Training Dataset
│   ├── data.yaml                    # Dataset configuration file
│   ├── images/ {train, val, test}   # Image splits
│   └── labels/ {train, val, test}   # YOLO format annotation files (.txt)
│
├── models/                          # Model Weights Directory
│   └── README.md                    # Weights download & fine-tuning instructions
│
├── data/                            # Runtime Data (git-ignored)
│   ├── raw/                         # Raw uploaded video files
│   ├── processed/                   # Transcoded frames
│   └── sample/                      # Sample assets for quick verification
│
├── outputs/                         # Output Exports (git-ignored)
│   ├── videos/                      # Annotated video output files
│   ├── images/                      # Extracted frame snapshots
│   └── reports/                     # Evaluation & JSON audit reports
│
├── tests/                           # Pytest Test Suite
│   ├── __init__.py
│   ├── conftest.py                  # Test fixtures & mock objects
│   ├── test_detection.py            # Detector & preprocessing unit tests
│   ├── test_severity.py             # Severity classifier unit tests
│   ├── test_gps.py                  # GPS extractor unit tests
│   ├── test_aggregation.py          # Temporal deduplication unit tests
│   └── test_api.py                  # Flask REST API integration tests
│
└── scripts/                         # CLI Automation Utilities
    ├── check_dataset.py             # Dataset sanity, label validation & audit tool
    ├── train.py                     # YOLOv8 fine-tuning CLI runner
    ├── evaluate.py                  # Model evaluation & empirical benchmark reporter
    └── process_video_cli.py         # Headless video processing CLI runner
```

---

## 6. Dataset Strategy & YOLOv8 Annotation

### Acquisition & Public Benchmarks
1. **RDD2020 / RDD2022 (Road Damage Dataset):** Class `D40` (Potholes).
2. **Kaggle Pothole Detection Benchmarks:** Dashcam perspective road datasets.
3. **Roboflow Universe Pothole Benchmarks:** Pre-formatted YOLO splits.

### YOLO Format Specification
Labels are stored as individual `.txt` files matching image names. Each line represents a normalized bounding box:
$$\text{Format:} \quad \langle\text{class\_id}\rangle \quad \langle x_{\text{center}}\rangle \quad \langle y_{\text{center}}\rangle \quad \langle\text{width}\rangle \quad \langle\text{height}\rangle$$

### Dataset Quality Assurance Tool
Run the dataset auditor before training:
```powershell
python scripts/check_dataset.py --data dataset/data.yaml --visualize 3
```

---

## 7. Model Training & Empirical Evaluation

### Training with CLI Arguments
```powershell
python scripts/train.py --data dataset/data.yaml --model yolov8n.pt --epochs 40 --batch 16 --device auto
```

### Empirical Evaluation & Metric Reporting
```powershell
python scripts/evaluate.py --model models/pothole_yolov8n.pt --data dataset/data.yaml --split val
```
- **Precision:** $\frac{TP}{TP + FP}$ — Ratio of true potholes among all model detections.
- **Recall:** $\frac{TP}{TP + FN}$ — Ratio of actual potholes successfully detected.
- **mAP@0.50:** Mean Average Precision calculated at $0.50$ IoU overlap.
- **mAP@0.50:0.95:** Primary COCO benchmark averaged across IoU thresholds from $0.50$ to $0.95$.

---

## 8. Installation & Quickstart Guide

### Prerequisites
- Python 3.10 or 3.11 installed.
- MySQL Server installed and active.

### Step 1: Clone Repository & Create Virtual Environment
```powershell
git clone <repository_url>
cd "pothole detection and flagging system"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Python Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Database & Environment
1. In MySQL, initialize the database:
   ```sql
   CREATE DATABASE IF NOT EXISTS pothole_detection_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```
2. Import the schema:
   ```powershell
   # Using MySQL CLI or phpMyAdmin / Workbench:
   mysql -u root -p pothole_detection_db < app/database/schema.sql
   ```
3. Update `.env` with your database credentials.

### Step 4: Run the Application Suite

#### Option A: Run Full Web Experience (Flask + Streamlit)
**Terminal 1 (Flask REST API):**
```powershell
python -m app.api.routes
# API active at: http://127.0.0.1:5000
```

**Terminal 2 (Streamlit Web Dashboard):**
```powershell
streamlit run streamlit_app/app.py
# Dashboard active at: http://localhost:8501
```

#### Option B: Headless CLI Video Processing
```powershell
python scripts/process_video_cli.py --video "data/sample/sample_road.mp4" --conf 0.35 --frame-skip 2
```

---

## 9. Verification & Automated Test Suite

Run the full automated pytest suite:
```powershell
pytest tests/ -v
```

---

## 10. Technical Interview Defense Guide

### 30-Second Elevator Pitch
> *"I built an end-to-end Pothole Detection and Road Condition Mapping System that ingests vehicle video streams, detects potholes using a fine-tuned YOLOv8 model, extracts embedded GPS telemetry with zero-crash fallbacks, classifies visual severity through geometric surface area ratios, deduplicates consecutive frame detections into unique physical defects, persists structured audit logs in MySQL, and presents results on an interactive Folium geospatial dashboard via Flask REST APIs."*

### Key Architectural Q&A

**Q: How do you prevent counting the same pothole multiple times as the car drives over it?**
> *"I implemented a Spatial-Temporal Aggregator in `app/aggregation/aggregator.py`. For every frame detection, we compute the bounding box centroid $(c_x, c_y)$ and timestamp $t$. Detections within a Euclidean centroid distance threshold ($\le 75\text{px}$) and temporal window ($\Delta t \le 1.0\text{s}$) are clustered into a single `RoadConditionEvent`, tracking observation count, peak confidence, and maximum severity."*

**Q: Can your system measure actual 3D pothole depth from monocular 2D video?**
> *"No, monocular 2D RGB video capture cannot measure physical vertical cavity depth without LiDAR or calibrated stereo cameras. To maintain engineering honesty, our `SeverityClassifier` explicitly computes normalized 2D surface disruption area ($A_{\text{bbox}} / A_{\text{frame}}$) to classify relative visual severity (LOW, MEDIUM, HIGH) rather than making unfounded claims about physical cavity depth."*

---

## 📄 License
This project is licensed under the MIT License.
