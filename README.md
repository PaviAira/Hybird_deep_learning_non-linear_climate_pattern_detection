# 🌍 Hybrid Deep Learning for Non-Linear Climate Pattern Detection Using Global Temperature Data
## 📌 Project Overview

Climate change has become one of the most significant global challenges, requiring advanced analytical methods to understand long-term temperature variations and future climate trends. This project presents a **Hybrid Deep Learning Framework** that combines **Long Short-Term Memory (LSTM)** networks and **Transformer architecture** to detect complex non-linear climate patterns and forecast future global temperature changes.

The system analyzes **165 years of global temperature data (1860–2025)** and provides multiple climate analysis functionalities through an interactive dashboard. By leveraging the strengths of both sequence learning and attention mechanisms, the model captures temporal dependencies, seasonal variations, long-term trends, and anomalies in climate data.

---

## 🎯 Objectives

* Analyze historical global temperature records.
* Detect non-linear climate patterns and temperature anomalies.
* Identify seasonal climate variations.
* Visualize long-term climate trends.
* Forecast future global temperature changes.
* Provide an interactive and user-friendly climate analytics dashboard.

---

## 🏗️ System Architecture

### Transformer Branch

* Captures long-range dependencies in climate data.
* Utilizes self-attention mechanisms.
* Performs climate trend analysis.

### LSTM Branch

* Learns temporal relationships from sequential temperature records.
* Identifies seasonal climate variations.
* Models recurring temperature patterns.

### Hybrid Branch

* Combines features extracted from both Transformer and LSTM models.
* Detects climate anomalies.
* Generates future temperature forecasts.

---

## 📊 Dataset

**Dataset:** Global Temperature Dataset

**Time Span:** 1860 – 2025

**Total Coverage:** 165 Years

### Dataset Features

* Date
* Global Average Temperature

### Data Preprocessing

* Missing value handling
* Date formatting
* Feature scaling using MinMaxScaler
* Sequence generation for deep learning models
* Train-test splitting

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Deep Learning Frameworks

* TensorFlow
* Keras

### Data Analysis Libraries

* Pandas
* NumPy

### Visualization Libraries

* Plotly
* Matplotlib

### Web Application Framework

* Streamlit

### Model Serialization

* Pickle

---

## 🔍 Key Functionalities

### 1. Climate Trend Analysis

* Visualizes long-term global temperature trends.
* Powered by Transformer attention mechanisms.
* Helps identify warming and cooling periods.

### 2. Seasonal Variations Analysis

* Detects recurring seasonal temperature patterns.
* Utilizes LSTM sequence learning capabilities.
* Provides year-wise seasonal insights.

### 3. Climate Anomaly Detection

* Identifies unusual temperature deviations.
* Uses hybrid feature representations.
* Highlights significant climate events.

### 4. Future Forecasting

* Predicts future temperature patterns.
* Generates forecasts up to future years.
* Assists in climate trend assessment and planning.

---

## 📈 Model Evaluation

The model was evaluated using regression metrics:

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted temperatures.

**MSE:** 0.7619

### Mean Absolute Error (MAE)

Measures the average absolute deviation between actual and predicted temperatures.

**MAE:** 0.7298°C

---

## 🌟 Results

* Successfully analyzed 165 years of global temperature records.
* Captured long-term climate trends using Transformer architecture.
* Detected seasonal temperature variations using LSTM.
* Identified climate anomalies through hybrid feature learning.
* Achieved reliable future temperature forecasting with low prediction error.

---

## 🔮 Future Enhancements

* Integration of additional climate variables such as precipitation, humidity, and atmospheric CO₂ levels.
* Incorporation of real-time climate data streams.
* Extension to regional and country-specific climate forecasting.
* Development of ensemble forecasting approaches.
* Deployment on cloud platforms for large-scale climate analytics.

---

