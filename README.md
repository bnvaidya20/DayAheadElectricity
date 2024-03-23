# Electricity Day-Ahead (DAH) Price Analysis

This repository contains a series of Python scripts for exploratory data analysis (EDA), feature engineering, and predictive modeling of day-ahead (DAH) electricity prices. The scripts are designed to load, preprocess, visualize, and analyze electricity price data, as well as to build and evaluate predictive models using machine learning techniques, including Random Forest, XGBoost, and LSTM.


## Features

- Data loading and preprocessing utilities.

- Comprehensive exploratory data analysis with visualizations.

- Feature engineering for machine learning readiness.

- Implementation and evaluation of Random Forest and XGBoost models.

- Time series forecasting with an LSTM neural network.

## Project Structure

The project is structured into multiple scripts, each serving a different purpose in the data analysis and modeling pipeline:

- `elec_dah_price_base_eda.py`: Performs basic exploratory data analysis on electricity DAH price data, including loading data, visualizing price trends across different countries, handling missing values, and identifying outliers.

- `elec_dah_price_main_eda.py`: Continues the EDA process with more advanced techniques, including renaming columns, extracting datetime features, and creating lag and rolling features to better understand the time series data.

- `elec_dah_price_rf_xgb.py`: Applies machine learning models, specifically Random Forest (RF) and XGBoost (XGB), to predict electricity prices based on the features engineered in the previous steps.

- `elec_dah_price_lstm.py`: Utilizes a Long Short-Term Memory (LSTM) neural network model to predict electricity prices, showcasing a deep learning approach to time series forecasting.

## Getting Started

### Prerequisites

To run these scripts, you will need Python 3.6+ and the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- keras

## Installation

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost keras

