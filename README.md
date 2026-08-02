# 🚍 UK Bus Operator Benchmarking System

An end-to-end **Big Data analytics** and **Machine Learning** pipeline built with **PySpark (Apache Spark 4.1.1)** to benchmark UK bus operator performance using data from the **Bus Open Data Service (BODS)**. The project integrates timetable, vehicle telemetry, and disruption data to identify operational patterns through unsupervised clustering and presents the results in an interactive Streamlit dashboard.

---

## 📖 Overview

This project combines multiple UK public transport datasets into a single service-level dataset and applies clustering techniques to group bus services based on operational performance.

The pipeline consists of:

- Data ingestion and preprocessing
- Multi-source data integration
- Feature engineering
- Machine learning with PySpark MLlib
- Interactive dashboard for analysis and benchmarking

---

## 🎯 Features

- 📂 Ingests **19,306** TransXChange XML timetable files
- 🚍 Integrates **GTFS-Realtime** vehicle telemetry
- ⚠️ Processes **SIRI-SX** disruption feeds
- ⚡ Distributed processing using **Apache Spark (PySpark)**
- 📊 Calculates a custom **Service Performance Index (SPI)**
- 🤖 Compares three clustering algorithms:
  - K-Means
  - Bisecting K-Means
  - Gaussian Mixture Model (GMM)
- 📈 Interactive Streamlit dashboard with Plotly visualisations
- 📥 CSV export and filtering capabilities

---

# 🏗️ System Architecture

```text
                +----------------------+
                |  TransXChange XML    |
                +----------------------+
                           |
                +----------------------+
                | GTFS-Realtime Feed   |
                +----------------------+
                           |
                +----------------------+
                |    SIRI-SX Feed      |
                +----------------------+
                           |
                           ▼
                 Notebook 1 (ETL)
      Parse → Clean → Join → Feature Engineering
                           |
                           ▼
              benchmark_dataset (Parquet)
                           |
                           ▼
              Notebook 2 (EDA & Validation)
                           |
                           ▼
            Notebook 3 (ML Clustering)
     StandardScaler → K-Means → CSV Export
                           |
                           ▼
              Streamlit Dashboard (app.py)
```

---

# 📊 Dataset

| Source | Description |
|---------|-------------|
| **TransXChange** | Timetable and route information |
| **GTFS-Realtime** | Live vehicle positions and speeds |
| **SIRI-SX** | Service disruption information |

### Final Dataset

- **19,306** raw XML files
- **19,304** unique services
- Joined into a single service-level dataset

---

# ⚙️ Feature Engineering

The custom **Service Performance Index (SPI)** combines:

- Trips (service frequency)
- Stops (route coverage)
- Average route speed
- Route disruptions

These features are standardised before clustering using:

- `VectorAssembler`
- `StandardScaler`

---

# 🤖 Machine Learning

Three clustering algorithms were evaluated.

| Model | Silhouette Score | WCSS | Status |
|------|----------------:|-------------:|---------|
| **K-Means (k=4)** | **0.8828** | **28,754.34** | ✅ Selected |
| Bisecting K-Means | 0.4563 | 36,399.97 | Evaluated |
| Gaussian Mixture | 0.3714 | N/A | Evaluated |

### Final Clusters

- 🟢 Premium Operations
- 🔵 Efficient Operations
- 🟡 Standard Operations
- 🔴 Underperforming Operations

---

# 📈 Dashboard

The Streamlit dashboard includes:

- KPI scorecards
- Cluster summaries
- UK operator map
- SPI distribution
- Correlation analysis
- Operator performance table
- Searchable data explorer
- CSV export
- Sidebar filtering

Filters include:

- Operator
- Cluster
- SPI range
- Trips range
- Stops range

---

# 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3.10 / 3.11 |
| Big Data | Apache Spark 4.1.1 (PySpark) |
| Machine Learning | PySpark MLlib |
| Data Processing | Pandas, NumPy, PyArrow |
| Parsing | xml.etree.ElementTree, gtfs-realtime-bindings |
| Dashboard | Streamlit, Plotly |
| Storage | Parquet, CSV |


---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/<username>/uk-bus-operator-benchmarking.git

cd uk-bus-operator-benchmarking
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### 1. Execute the notebooks

```bash
jupyter lab
```

Run in order:

1. **ETL_Pipeline.ipynb**
2. **EDA.ipynb**
3. **ML_Clustering.ipynb**

This produces:

```
output/benchmark_dataset/
output/dashboard_data.csv
```

---

### 2. Launch the dashboard

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

# 📊 Results

- Successfully integrated three independent BODS data sources.
- Created a unified service-level dataset.
- K-Means achieved the highest clustering quality with a **Silhouette Score of 0.8828**.
- Dashboard provides interactive benchmarking of UK bus operators and routes.

---

# 🔮 Future Improvements

- Parallel XML ingestion using Spark `binaryFiles()`
- Spark Structured Streaming for live telemetry
- Supervised delay prediction
- Real geographic mapping
- Integration of fare datasets

---

# 📄 Licence

This project uses publicly available UK Department for Transport **Bus Open Data Service (BODS)** datasets provided under the applicable Open Government Licence.

---

# 👤 Author

**Suchit Ratna Bajracharya**

