# 🧠 Early Prediction of Diabetes Disease Using Machine Learning

This project demonstrates an end-to-end machine learning pipeline to **predict the early onset of diabetes** using various supervised learning algorithms. The primary objective is to leverage clinical and demographic data to build a reliable predictive model that can be used in healthcare settings to assist with early diagnosis and prevention of diabetes.

---

## 📌 Highlights

- 📊 Dataset: Kaggle - *Diabetes Early Disease Prediction*
- ⚙️ Algorithms Used: Decision Tree, Random Forest, K-Nearest Neighbors, XGBoost, AdaBoost, Logistic Regression
- 🎯 Best Performing Model: **XGBoost (Accuracy: 91.48%)**
- 🧪 Evaluation Metrics: Accuracy, Recall, Confusion Matrix, ROC Curve, AUC Score
- 🛠 Tools & Libraries: `pandas`, `scikit-learn`, `xgboost`, `seaborn`, `matplotlib`, `joblib`
- 🌐 Deployment: Model saved as `XGBoost.pkl` for future web integration (e.g., Gradio)

---

## 🧬 Dataset Description

The dataset contains clinical and demographic information of patients including:

- `age`, `gender`, `smoking_history`
- `bmi`, `HbA1c_level`, `blood_glucose_level`
- `hypertension`, `heart_disease`
- `diabetes` *(target variable: 0 = No Diabetes, 1 = Diabetes)*

---

## 🧪 Methodology

1. **Data Preprocessing**
   - Handling missing values
   - Removing duplicates
   - Label encoding categorical features
   - Resampling (RandomUnderSampler)

2. **Exploratory Data Analysis**
   - Distribution plots
   - Feature correlation heatmaps
   - Count plots by class distribution

3. **Feature Selection**
   - Feature importance via RandomForest

4. **Model Training**
   - Hyperparameter tuning with `GridSearchCV`
   - Models:
     - Decision Tree Classifier
     - Random Forest Classifier
     - K-Nearest Neighbors
     - XGBoost Classifier
     - AdaBoost Classifier
     - Logistic Regression

5. **Model Evaluation**
   - Accuracy & Recall scores
   - Confusion matrices
   - ROC-AUC curves for both training and testing data

6. **Model Comparison**
   - Accuracy scores for all models visualized
   - XGBoost selected as the best model (Accuracy: **91.48%**)

---

## 📈 Model Performance

| Model               | Accuracy | Recall  |
|--------------------|----------|---------|
| Decision Tree       | 81.70%   | 67.48%  |
| Random Forest       | 89.83%   | 75.70%  |
| KNN                 | ~83%     | ~71%    |
| XGBoost             | **91.48%** | **83.05%** |
| AdaBoost            | ~82.4%   | ~67%    |
| Logistic Regression | ~80.6%   | ~67.5%  |

---

## 🧠 Model Deployment

- The final **XGBoost model** is exported using `joblib` for deployment:
  joblib.dump(model, 'XGBoost.pkl')
