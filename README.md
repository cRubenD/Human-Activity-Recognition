# Human-Activity-Recognition
Human Activity Recognition (HAR) Ensemble System
## 📌 Project Overview

This project implements a robust machine learning pipeline for **Human Activity Recognition (HAR)** using the UCI HAR Dataset. The goal is to classify 6 distinct human activities (Walking, Standing, Laying, etc.) based on sensor data.

Unlike standard implementations using high-level APIs (like `sklearn.linear_model`), this project features **custom, scratch-built implementations** of core algorithms wrapped in a modular Object-Oriented architecture. It leverages **PyTorch** for the neural network components and implements advanced Ensemble Learning techniques (Voting and Bagging) to maximize accuracy and estimate model uncertainty.

## 🚀 Key Features

* **Custom Model Implementation**: 
    * `LogisticRegression` and `MLPModel` built using `torch.nn.Module`.
    * `DecisionTreeClassifierSimple`: A complete decision tree implementation from scratch using NumPy (calculating Gini impurity, recursive splitting).
* **Ensemble Learning**: Implemented Voting Classifiers (Weighted/Unweighted) and Bagging strategies manually.
* **Experiment Tracking**: Integrated **TensorBoard** to visualize loss, accuracy, and recall curves in real-time.
* **Advanced Python**: Extensive use of **Type Hinting**, **Classes**, and modular design patterns.
* **Data Pipeline**: Custom `torch.utils.data.Dataset` implementation for efficient data loading and preprocessing.

## 🛠️ Technologies & Tools

* **Language**: Python 3.x
* **Deep Learning**: PyTorch
* **Data Manipulation**: NumPy, Pandas
* **Visualization**: Matplotlib, Seaborn, TensorBoard
* **Machine Learning**: Scikit-learn (for metrics and utilities)

