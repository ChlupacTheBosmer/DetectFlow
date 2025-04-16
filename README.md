![DetectFlow Banner](https://github.com/ChlupacTheBosmer/DetectFlow/assets/29023670/81525eff-f59a-4067-aa33-e8b6b66bcc4e)

# DetectFlow: Object Detection Pipeline for Flower Visitor Analysis 🌸🐝

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License:-see_LICENSE.md-blue)](https://opensource.org/licenses/MIT) 
[![](https://img.shields.io/badge/Under%20development...-8A2BE2)]()

Welcome to `detectflow`, a comprehensive Python package designed for building and managing high-performance object detection data pipelines. Its primary focus is the analysis of video recordings of flowers to detect and track their visitors (e.g., pollinators) using state-of-the-art YOLO architecture models. `detectflow` is engineered for high throughput and efficiency, optimized for deployment on High-Performance Computing (HPC) clusters like the Czech Metacentrum via PBS jobs. It features a robust multi-processing architecture to handle large-scale video analysis, decoupling tasks across multiple CPUs. The package includes extensive support for preparing training data, potentially auto-annotating data during analysis for future model improvements. It also offers advanced post-processing capabilities, including the transformation of frame-level detections into coherent visit events, and a dedicated GUI application for refining, validating, and manually identifying results. Further enhancing its capabilities, `detectflow` provides basic support for AI-powered flow direction using LLMs and incorporates sophisticated prediction techniques like reference flower detection, image tiling for small object detection, model ensembling, and confidence voting. Automation is central to `detectflow`, with a dedicated `Scheduler` class enabling streamlined scheduling and management of PBS jobs to minimize manual intervention and ensure data integrity.

## ✨ Key Features (Highlights)

* **🔬 YOLO-based Object Detection:** Leverages YOLO architecture models for accurate flower and visitor detection.
* **⚙️ HPC Optimized:** Designed for efficient execution on computational clusters (specifically Metacentrum) using multiple CPUs and PBS job management.
* **🚀 Parallel Processing:** Implements a robust multi-process architecture (Orchestrator, Workers, Database Manager) for scalable video processing.
* **📊 Data Preparation & Training Support:** Includes tools for preparing training data, with auto-annotation capabilities for continuous model improvement.
* **📈 Post-processing & Validation GUI:** Offers advanced features for data transformation and a GUI for results cleaning, validation, and manual identification.
* **🤖 LLM Integration:** Basic support for AI-driven decision-making using external or self-hosted Large Language Models.
* **💡 Sophisticated Prediction:** Incorporates reference flower detection, image tiling, model ensembling, and confidence voting.
* **📅 Automated Scheduling:** The `Scheduler` class automates and optimizes PBS job submission and management.

## 🏗️ Architecture and Components

The `detectflow` package features a modular structure designed for flexibility and scalability.

### 1. Video Processing Pipeline (`/process`)
The core of the package is a multi-process pipeline optimized for multi-CPU execution:
* **Orchestrator (`process/orchestrator.py`):** The main process that schedules and manages the processing of individual videos, communicating with worker processes.
* **Worker Processes:** Decentralized processes running on assigned CPUs. Each worker analyzes assigned video frames using YOLO models.
* **Database Manager (`process/database_manager.py`):** A separate process responsible for safely and efficiently storing detection results from worker processes into a database.

### 2. Prediction Engine (`/predict`)
This component handles the detection process itself:
* **Flower Detection:** Flowers are detected first to serve as reference areas.
* **Tiling:** Images are sliced into smaller tiles (e.g., 640x640) to improve the detection of small objects.
* **Ensembling and Voting:** Results from multiple models can be combined (`predict/ensembler.py`), and confidence voting is implemented to optimize precision and recall.
* **Reconstruction and Filtering:** Results from tiles are reconstructed for the original frame, and relevant detections (visitors on flowers) are filtered using the reference flower boxes.
* **Tracking:** Implements tracking algorithms (e.g., ByteTrack, BoT-SORT - see `/predict/cfg`) to link detections across frames.

### 3. Job Management and Scheduling (`/jobs`, `/process/scheduler.py`)
* **PBS Jobs:** Designed for deployment as PBS jobs on the Metacentrum HPC cluster (`jobs/bash_templates.py`, `utils/pbs_job_report.py`).
* **Scheduler (`jobs/scheduler.py`, `process/scheduler.py`):** A class for automating the creation, submission, and management of PBS jobs, optimizing resource usage and data handling.

### 4. Data Preparation and Training (`/utils`, `/jobs/train.py`)
Includes supporting infrastructure for:
* Preparing datasets for training YOLO models (`utils/dataset.py`).
* Potential for generating auto-annotated data during analysis.
* Launching training processes (`jobs/train.py`).

### 5. Post-processing and Validation GUI (`/app`)
* **Data Transformation:** Tools to convert frame-level detections into meaningful events (visits).
* **GUI Application (`app/app.py`):** An interactive tool (likely built with PyQt/PySide) for:
    * Visual inspection of detection results.
    * Manual cleaning and correction of detections/visits.
    * Identification of visitor species and behaviors.
    * Exporting curated data.

### 6. Utility Modules (`/utils`, `/manipulators`, `/handlers`, `/validators`, etc.)
The package contains numerous other modules for:
* Configuration (`/config`, `utils/config.py`, `handlers/config_handler.py`).
* Data manipulation (video, frames, boxes - `/manipulators`).
* Interaction with external systems (S3, SSH, Email, LLM - `/handlers`, `manipulators/s3_manipulator.py`).
* Data and input validation (`/validators`).
* Image and video processing (`/image`, `/video`).
* Logging and reporting (`utils/log_file.py`).

## 💻 HPC Integration (Metacentrum)

`detectflow` is designed with the specifics of the Metacentrum environment in mind:
* **PBS Scripts:** Generation and management of job scripts for `qsub`.
* **Multi-CPU Usage:** Efficient utilization of allocated CPUs through the multi-process architecture, helping to overcome limited GPU availability.
* **Dependency Management:** Tools for setting up the environment on cluster nodes (`jobs/setup_env.py`).
* **Resource Optimization:** The Scheduler helps plan jobs for efficient use of computation time.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd detectflow
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **Install dependencies:**
    The package includes a `requirements.txt` file listing necessary libraries.
    ```bash
    pip install -r config/requirements.txt
    # Additional system dependencies might be required (e.g., for OpenCV)
    ```
4.  **Install the package itself (if structured as an installable package):**
    ```bash
    pip install .
    ```
    Alternatively, use it directly from the cloned repository.

## 🚀 Usage (Basic Overview)

Using `detectflow` typically involves several steps:

1.  **Configuration:** Modify configuration files (e.g., in `/config/*.json`) to specify data paths, model parameters, cluster settings, etc.
2.  **Run the Orchestrator (locally or as a PBS job):**
    * For local testing or smaller tasks, you might run the orchestrator directly.
    * For large-scale processing on the cluster, use the `Scheduler` to create and submit a PBS job that launches the `Orchestrator`, which in turn manages worker processes and the database manager.
    ```python
    # Example (conceptual - actual code may vary)
    from detectflow.jobs import Scheduler
    from detectflow.process import Orchestrator
    from detectflow.handlers import ConfigHandler # Assuming ConfigHandler exists

    # Option 1: Direct execution (simplified)
    # config = ConfigHandler.load_config('config/orchestrator.json')
    # orchestrator = Orchestrator(config)
    # orchestrator.run()

    # Option 2: Using Scheduler for PBS job
    # scheduler_config = ConfigHandler.load_config('config/scheduler.json')
    # scheduler = Scheduler(scheduler_config)
    # job_id = scheduler.submit_orchestrator_job()
    # print(f"Submitted PBS job with ID: {job_id}")
    ```
3.  **Monitoring:** Track the progress of jobs on the cluster and monitor logs generated by the package.
4.  **Post-processing and Validation:** After the analysis completes, use the tools in `/app` and `/utils` to process, clean, and validate the results stored by the Database Manager.

## ⚙️ Configuration

Configuration is primarily managed through JSON files in the `/config` directory. Key configuration files might include:
* `orchestrator.json`: Settings for the processing pipeline (video paths, models, detection parameters).
* `scheduler.json`: Settings for automatic PBS job submission (cluster parameters, resource limits).
* Other configurations for the database, models, LLMs, etc.

A `ConfigHandler` class (`handlers/config_handler.py`) likely facilitates loading and managing these configurations.

---
