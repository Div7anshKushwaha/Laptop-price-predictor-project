# 💻 Laptop Price Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge&logo=streamlit">
</p>

---

## 📖 Project Overview

Laptop prices vary significantly depending on hardware specifications, display quality, processor type, storage configuration, and brand value.

This project uses Machine Learning to estimate the market price of a laptop based on its specifications.

The application allows users to enter laptop features through an interactive Streamlit interface and instantly receive a predicted price.

---

## 🎯 Problem Statement

Predict the selling price of a laptop using:

- Brand
- Laptop Type
- RAM
- Processor
- Storage
- GPU
- Display Specifications
- Operating System
- Weight

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Pickle

### Deployment
- Streamlit

---

## 📊 Dataset Features

| Feature | Description |
|----------|------------|
| Company | Laptop Brand |
| TypeName | Laptop Category |
| Ram | RAM Capacity |
| Weight | Laptop Weight |
| Touchscreen | Touch Support |
| IPS | IPS Display |
| PPI | Pixels Per Inch |
| CPU Brand | Processor Brand |
| HDD | HDD Storage |
| SSD | SSD Storage |
| GPU Brand | Graphics Brand |
| Operating System | Installed OS |

---

## ⚙️ Feature Engineering

Several new features were engineered from raw specifications:

### Display Features
- Screen Resolution
- PPI Calculation

### Storage Features
- HDD Capacity
- SSD Capacity
- Hybrid Storage
- Flash Storage

### Encoding
- One Hot Encoding for categorical variables

---

## 🤖 Machine Learning Models Tested

- Linear Regression
- Ridge Regression
- Lasso Regression
- KNN Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor
- Extra Trees Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Voting Regressor (Final Model)

---

## 🏆 Final Model

### Voting Regressor

Base Models:

- Random Forest
- Gradient Boosting
- XGBoost
- Extra Trees

Why Voting Regressor?

✔ Better generalization

✔ Reduced overfitting

✔ Combines strengths of multiple algorithms

✔ Improved prediction accuracy

---

## 📷 Application Preview

### Home Screen

<img src="images/app-ui.png" width="900">

---

## 🚀 How to Run Locally

### Clone Repository

```bash
git clone https://github.com/your-username/Laptop-Price-Predictor.git
```

### Move into Project Directory

```bash
cd Laptop-Price-Predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
Laptop-Price-Predictor
│
├── app.py
├── pipe.pkl
├── df.pkl
├── requirements.txt
├── laptop-price-predictor.ipynb
├── README.md
```

---

## 📈 Workflow

Dataset
↓
Data Cleaning
↓
Feature Engineering
↓
EDA
↓
Model Building
↓
Model Evaluation
↓
Voting Regressor
↓
Pickle Serialization
↓
Streamlit Deployment

---

## 🌟 Key Features

✅ Interactive Web Application

✅ Real-Time Price Prediction

✅ Automated PPI Calculation

✅ Ensemble Learning

✅ Responsive Dark UI

✅ Production-Ready Pipeline

✅ End-to-End Machine Learning Project

---

## 🎓 Learning Outcomes

Through this project I learned:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Model Selection
- Ensemble Learning
- Hyperparameter Tuning
- Model Serialization
- Streamlit Deployment
- End-to-End ML Project Development

---

## 👨‍💻 Author

### Divyansh Kushwaha

🎓 BS in Data Science

🏛 Indian Institute of Technology Madras

📊 Aspiring Data Scientist

🔗 LinkedIn: www.linkedin.com/in/divyansh-kushwaha-603616383

🔗 GitHub: https://github.com/Div7anshKushwaha

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It motivates me to build more Data Science and Machine Learning projects.
