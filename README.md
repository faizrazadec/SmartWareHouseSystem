<!-- filepath: /home/faizraza/Personal/Projects/SmartWareHouseSystem/README.md -->
# Smart Warehouse System

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

<!-- Add other badges here: e.g., build status, code coverage -->

A Streamlit-based web application for intelligent warehouse operations management, encompassing defect detection, demand forecasting, and comprehensive sales analytics.

## Overview

The Smart Warehouse System is designed to optimize warehouse efficiency by leveraging machine learning and data analytics. It provides a user-friendly interface for monitoring key performance indicators, identifying product defects, predicting future demand, and tracking sales performance.

## Features

-   **Centralized Dashboard**: An interactive main dashboard (`app.py`) offering a holistic view of critical warehouse metrics.
-   **Automated Defect Detection**:
    -   Utilizes machine learning models (see `pages/Defect Detection.py` and `Defect-detection/`) to identify defects in products.
    -   Provides detailed defect statistics and analysis (`pages/Defect_Analysis.py`).
    -   Generates alerts for newly detected defects (`pages/Alerts.py`).
-   **Predictive Demand Forecasting**:
    -   Employs historical data and machine learning algorithms (`pages/Forecasting.py`, `inference/forecasting.py`) to forecast product demand.
    -   Leverages pre-trained models stored in `models/forecasting_model/`.
-   **In-depth Sales Tracking**:
    -   Facilitates the management and visualization of sales data (`pages/Sales.py`).
-   **Robust Database Integration**:
    -   Seamlessly interacts with an SQLite database (`database.py`) for persistent storage and retrieval of data related to defects, sales, and products.

## Technologies Utilized

-   **Core Language**: Python (3.12+)
-   **Web Framework**: Streamlit
-   **Data Manipulation**: Pandas
-   **Data Visualization**: Altair
-   **Machine Learning**: Scikit-learn
-   **Deep Learning**: TensorFlow (Keras), PyTorch
-   **Object Detection**: RFDetr (specific models for defect detection)
-   **Database**: SQLite

## Project Structure

```
├── app.py                        # Main Streamlit application
├── database.py                   # Database interaction logic
├── db.db                         # SQLite database file
├── pyproject.toml                # Project metadata and dependencies
├── README.md                     # This file
├── dataset/                      # Datasets for training and testing
│   ├── Historical Product Demand.csv
│   └── box_dataset/              # Defect detection image dataset
├── Defect-detection/
│   └── model.py                  # Defect detection model definition
├── inference/                    # Scripts for model inference
│   ├── forecasting.py
│   └── rfdetr_inference.py
├── models/                       # Trained machine learning models
│   ├── Farah_model5.keras
│   ├── forecasting_model/
│   └── rfdetr_model/
├── pages/                        # Streamlit pages for different features
│   ├── Alerts.py
│   ├── Defect Detection.py
│   ├── Defect_Analysis.py
│   ├── Forecasting.py
│   └── Sales.py
└── ... (other configuration and utility files)
```

## Setup and Installation

This project use `uv` as it's package manager instead of traditional `pip`. Learn more about [uv](https://docs.astral.sh/uv/)

1.  **Clone the Repository:**
    ```bash
    https://github.com/faizrazadec/SmartWareHouseSystem.git
    cd SmartWareHouseSystem
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    uv venv
    ```

3.  **Install dependencies:**

    ```bash
    uv pip install -r pyproject.toml
    ```

4.  **Database Initialization:**
    The application utilizes an SQLite database (`db.db`). Ensure this file is correctly placed in the project root. If the database schema needs to be initialized or migrations are required, please refer to any accompanying database setup scripts (not detailed here, assuming `db.db` is pre-configured or schema creation is handled within the application).

## Usage

To launch the Smart Warehouse System application:

```bash
streamlit run app.py
```

The application will typically be accessible via `http://localhost:8501` in your web browser.

## Modules Deep Dive

### 1. Defect Detection Module
   - **Objective**: To automatically identify and classify defects in warehouse items using image processing and machine learning.
   - **Core Model**: Employs an RFDetr-based architecture (e.g., `rf-detr-base.pth`, with inference logic in `inference/rfdetr_inference.py`).
   - **Training Data**: Utilizes image datasets located under `dataset/box_dataset/`.
   - **Application Interface**:
     - `pages/Defect Detection.py`: For real-time defect detection or uploading images for analysis.
     - `pages/Defect_Analysis.py`: For reviewing aggregated defect statistics and trends.

### 2. Demand Forecasting Module
   - **Objective**: To predict future product demand, enabling proactive inventory management.
   - **Core Model**: Leverages a Keras-based time series forecasting model (e.g., `models/Farah_model5.keras`, with inference logic in `inference/forecasting.py`).
   - **Training Data**: Based on historical demand data, typically found in `dataset/Historical Product Demand.csv`.
   - **Output**: Generates demand forecasts (e.g., `forecast_next_30_days.csv`).
   - **Application Interface**: `pages/Forecasting.py`.

### 3. Sales Analytics Module
   - **Objective**: To provide insights into sales performance through data visualization and summaries.
   - **Data Source**: Retrieves data from `sales` and related tables within the `db.db` SQLite database.
   - **Application Interface**: `pages/Sales.py`.

### 4. Alerting System
   - **Objective**: To notify users of critical events, such as low stock levels or high defect rates.
   - **Application Interface**: `pages/Alerts.py`.

---

### ⭐ **Support This Project!**  
If you found this useful, **please consider leaving a star ⭐ on GitHub**!  
It motivates me to keep building more **open-source projects** 🚀  

---
