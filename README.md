# 🌱 Dry Bean Classification using Machine Learning

## 📌 Project Overview

This project implements a machine learning classification system for
predicting the type of dry bean using different classification algorithms.

The project includes data preprocessing, model training, model evaluation,
model saving, and a Streamlit web application for prediction.

---

## 📊 Dataset

The project uses the Dry Bean Dataset.

The dataset contains 16 numerical features describing the physical
characteristics of dry beans.

The target variable is:

`Class`

The dataset contains 7 different dry bean classes:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

---

## 🤖 Machine Learning Models

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest

---

## 📏 Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

A confusion matrix and classification report are also generated.

---

## 🌐 Streamlit Application

A Streamlit web application was developed for the project.

The application provides:

- CSV file upload
- Machine learning model selection
- Prediction
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion Matrix
- Classification Report
- Prediction Results

---

## 📁 Project Structure

```text
A2/
│
├── app.py
├── U.ipynb
├── requirements.txt
├── README.md
├── Dry_Bean_Dataset.xlsx
├── test_data.csv
├── metrics.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── scaler.joblib
    └── label_encoder.joblib
```

---

## 🔗 GitHub Repository

[View the source code on GitHub](https://github.com/amityya/dry-bean-classification)

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/amityya/dry-bean-classification.git
```

### 2. Move into the project directory

```bash
cd dry-bean-classification
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📈 Logistic Regression Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.9214 |
| AUC | 0.9934 |
| Precision | 0.9222 |
| Recall | 0.9214 |
| F1 Score | 0.9216 |
| MCC | 0.9050 |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## 🎯 Conclusion

The project demonstrates the complete machine learning workflow,
starting from dataset preparation and preprocessing to model training,
evaluation, model saving, and deployment through a Streamlit application.

