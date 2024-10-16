
# Efficient Data Stream Anomaly Detection

## Project Overview

This project demonstrates real-time anomaly detection using a continuous data stream. The system uses Z-score-based anomaly detection with a sliding window of recent data points. The data stream simulates seasonal variations, noise, and gradual drift, mimicking real-world data patterns such as financial transactions or system metrics.

### Key Components:
- **Algorithm**: Modified Z-score outlier detection.
- **Data Simulation**: A function simulates real-time data with noise, drift, and seasonality.
- **Real-Time Anomaly Detection**: A system detects anomalies in the data stream using a thresholded Z-score.
- **Real-Time Visualization**: A visualization tool plots the data stream and highlights anomalies.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/efficient-anomaly-detection.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run the anomaly detection system:
```bash
python anomaly_detection.py
```

## Rationale for Methodologies

### Gradient Boosting Machines (GBM) for Data Stream Generation

Gradient Boosting Machines (GBM) are employed for generating financial transaction-like data streams due to their ability to model complex relationships and handle non-linear patterns effectively. The GBM algorithm captures underlying data trends and seasonal variations while introducing noise and drift, thus simulating realistic scenarios. This makes it an ideal choice for mimicking the dynamic nature of financial transactions, which often exhibit gradual shifts and unexpected spikes.

### Modified Z-Score for Anomaly Detection

The Modified Z-Score is utilized for anomaly detection because it provides a robust measure of deviation from the median, making it less sensitive to outliers compared to the traditional Z-score. This is crucial in real-time applications where data can be influenced by extreme values or anomalies. By using the Modified Z-Score, the system maintains high detection accuracy while minimizing false positives, allowing for more reliable identification of anomalies in financial data streams.
