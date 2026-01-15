# Student Performance Prediction — End-to-End Machine Learning Pipeline

This repository contains a complete end-to-end Machine Learning pipeline built to predict student academic performance, including Midterm-I, Midterm-II, and Final Exam marks using real-world educational data.

The project follows industry-standard ML practices, covering everything from raw data preprocessing to model evaluation and deployment via an interactive dashboard.

# 🔍 Project Objectives

We answer three research questions:

RQ1: How accurately can we predict Midterm-I marks?

RQ2: How accurately can we predict Midterm-II marks?

RQ3: How accurately can we predict Final Exam marks?

# ⚙️ ML Pipeline Overview

The project implements a full production-style ML workflow:

Data cleaning and merging of multi-sheet datasets

Leakage-safe preprocessing and feature engineering

Exploratory Data Analysis (EDA) with visualizations

Regression models:

Simple Linear Regression

Multiple Linear Regression

Polynomial Regression

Bootstrapping (500 samples) with 95% MAE confidence intervals

Model evaluation using MAE, RMSE, and R²

Baseline comparison using Dummy Regressor

Overfitting and underfitting analysis

Interactive dashboard using Streamlit / Gradio

Complete workflow diagram

# 📁 Repository Structure
├── data/                 # Preprocessed dataset
├── notebook.ipynb       # Full ML pipeline & analysis
├── dashboard/           # Streamlit/Gradio app
├── pipeline_diagram/    # Workflow diagram
└── README.md

# 📈 Evaluation Metrics

Each model is evaluated using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Bootstrapped 95% confidence intervals

Comparison with Dummy Baseline model

# 🚀 How to Run
jupyter notebook notebook.ipynb
streamlit run dashboard/app.py
