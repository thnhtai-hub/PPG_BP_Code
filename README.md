# Hypertension Classification and Prediction Using Machine Learning and PPG Signal Analysis

Author: The project is developed by **Nguyen Thanh Tai** as part of the Undergraduate Final Year Project at **Greenwich Vietnam FPT University**.

This Final Project uses **machine learning** to classify and predict hypertension stages based on **Photoplethysmography (PPG)** signals and **physiological data**. The project implement various machine learning models, deep learning algorithm, and a user-friendly GUI for real-time predictions, data processing and model performance visualization.

---

## Acknowledgments

Special thanks to:

- **Ms. Tran** and **Mr. Bao**, to whom I extend my deepest gratitude, are my esteemed lecturers who provided invaluable guidance, support, and insightful direction throughout the development of this project. Their expertise and encouragement have been instrumental in shaping the progress and outcomes of this work.
- **Guilin People's Hospital** for providing essential data.
- **World Health Organization (WHO)** for global hypertension statistics and insights.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Resources](#resources)

---

## Introduction

Hypertension is a leading cause of early mortality globally. This project aims to:

- Develop a machine learning pipeline capable of classifying hypertension stages: **Normal, Prehypertension, Stage 1 Hypertension, and Stage 2 Hypertension** based on PPG waveform data and physiological factors such as blood pressure, age, BMI, etc.
- Utilize PPG signals and physiological data for predictions.
- Create a GUI to make predictions accessible and actionable for easier model performance visualization.

---

## Features

- **Data Exploration and Preprocessing**:

  - Perform statistical analysis on data, identify distributions, and handle missing values.
  - Clean and transform data, encode categorical features, and label hypertension stages.
  - Finding relationships between features and performing multivariate visualization.

- **Machine Learning Models**:

  - Train and evaluate multiple models, including Decision Tree, Random Forest, SVM, Logistic Regression, Gradient Boosting, K-Nearest Neighbors, and Naive Bayes.
  - Optimize models using hyperparameter tuning and GridSearchCV.
  - Perform model evaluation, clustering, and visualization

- **Signal Processing**:

  - Assess signal quality using Skewness Signal Quality Index (SQI) to identify high-quality PPG signals.
  - Normalize signal amplitudes and extract time-domain and frequency-domain features.

- **Deep Learning**:

  - Develop and train Multilayer Perceptron (MLP) models for advanced predictions.
  - Evaluate performance using metrics like loss, accuracy, and confusion matrices.

- **Graphical User Interface (GUI)**:
  - Interactive GUI built with Streamlit for real-time data analysis and predictions.
  - Allow users to upload datasets, select features, choose machine learning models, and view detailed performance metrics, including accuracy and confusion matrices.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - **Data Processing**: NumPy, Pandas
  - **Visualization**: Matplotlib, Seaborn
  - **Machine Learning**: Scikit-learn, TensorFlow, Keras
- **GUI Development**: Streamlit
- **Tools**:
  - Jupyter Notebooks
  - Visual Studio Code

## Installation

1.  **Create a virtual environment**:

- **Using `venv`**:
  ```bash
  python -m venv env
  source env/bin/activate   # Linux/MacOS
  env\Scripts\activate      # Windows
  ```
- **Using Anaconda**:
  ```bash
  conda create --name ppg_env python=3.8
  conda activate ppg_env
  ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/thnhtai-hub/PPG_BP_CODE.git
   cd PPG_BP_CODE
   ```
3. **Install packages in requirements.txt**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

How to use the GUI:

1. Navigate to the app.py file:
   ```bash
   cd PPG_BP_Code/code/GUI
   ```
2. Launch the app:
   ```bash
   streamlit run app.py
   ```
3. Then you can upload the dataset, scroll through EDA pages for detailed insights on how the steps are done. To run models: Choose features and machine learning models from the sidebar to train or evaluate. (app page) For model performance, click "Compare Models" to observe predictions, accuracy scores, classification reports, and detailed performance metrics.

## Resources

| Path                                                         | Description                                  |
| :----------------------------------------------------------- | :------------------------------------------- |
| [PPG_BP_Code]()                                              | Main folder.                                 |
| &boxv;&nbsp; &boxvr;&nbsp; [code]()                          | Main source code folder.                     |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [data]()            | Contains code for data files.                |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [GUI]()             | Code for the graphical user interface.       |
| &boxv;&nbsp; &boxvr;&nbsp; [dataset]()                       | Folder for storing datasets.                 |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [normalization]()   | Contains normalized dataset files.           |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [raw]()             | Raw dataset files.                           |
| &boxv;&nbsp; &boxvr;&nbsp; [images]()                        | Folder for storing generated images.         |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [images_gui]()      | GUI-related image outputs.                   |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [images_signal]()   | Signal-related plots.                        |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [images_spectrum]() | Spectrum-related visualizations.             |
| &boxv;&nbsp; &boxvr;&nbsp; [results]()                       | Stores results of best 219 SQI signal files. |
| &boxv;&nbsp; &boxvr;&nbsp; [save_models]()                   | Folder for saved deep learning model.        |
