# 🎓 Student Performance ML Predictor

A machine learning application that predicts student pass/fail outcomes using academic performance indicators. Built with Logistic Regression and SVM models, deployed via Streamlit for real-time inference.

## 📋 Project Overview

The **Student Performance ML Predictor** leverages historical student data to build predictive models that estimate the probability of a student passing based on six key academic indicators. The application provides an intuitive interface for educators and academic advisors to identify at-risk students early.

**Model Performance:** ~93% test accuracy with Logistic Regression

## 📊 Dataset

- **Source:** `student_performance_dataset.csv`
- **Size:** 1,000 student records
- **Target Variable:** Pass/Fail binary classification
- **Train/Test Split:** 80-20 split with stratification (random_state=42)

## 🔧 Data Preprocessing

The preprocessing pipeline (`final_data_preprocessing.ipynb`) performs:

1. **Exploratory Data Analysis (EDA)** — Statistical summaries and distribution analysis
2. **Missing Value Handling** — Identification and treatment of null values
3. **Feature Scaling** — RobustScaler applied to handle outliers (median & IQR normalization)
4. **Train-Test Separation** — Stratified split to maintain class distribution
5. **Output:** Preprocessed train and test CSV files

## ✨ Features Used

| Feature | Description | Range |
|---------|-------------|-------|
| Study_Hours | Weekly study time | 0–100 hrs |
| Attendance_Percentage | Class attendance rate | 0–100% |
| Assignment_Score | Average assignment score | 0–100 |
| Quiz_Score | Cumulative quiz performance | 0–100 |
| Project_Score | Group project evaluation | 0–100 |
| Midterm_Exam | Midterm examination score | 0–100 |

## 🤖 Machine Learning Models

### Logistic Regression
- **Algorithm:** Binary logistic regression with L2 regularization
- **Hyperparameters:** max_iter=1000, random_state=42
- **Test Accuracy:** 93.2%
- **Status:** Primary model (deployed in Streamlit app)

### Support Vector Machine (SVM)
- **Variants Tested:**
  - RBF Kernel (non-linear)
  - Linear Kernel
- **Hyperparameters:** C=1.0, gamma='scale'
- **Purpose:** Comparison and validation

## 📈 Model Evaluation

| Metric | Logistic Regression | SVM (RBF) | SVM (Linear) |
|--------|-------------------|-----------|--------------|
| Accuracy | 93.2% | ~89% | ~91% |
| Precision | High | Moderate | Good |
| Recall | Balanced | Good | Good |

*Confusion matrices and classification reports generated in `ml_models.ipynb`*

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor
```

### Step 2: Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install streamlit numpy pandas joblib plotly scikit-learn
```

**Or use requirements.txt:**
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

### Run the Streamlit Application
```bash
python -m streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### Access the Prediction Studio
1. Navigate to **Prediction Studio** in the sidebar
2. Enter academic indicators in the input form
3. Click **"✦ RUN AI PREDICTION"**
4. View the AI verdict with confidence score and visualizations

### Explore the Model
Visit the **About the Model** page to understand:
- Algorithm architecture
- Feature descriptions
- Training accuracy metrics
- Preprocessing pipeline explanation

## 📁 Project Structure

```
student-performance-predictor/
├── README.md
├── app.py                                    # Streamlit application
├── notebooks/
│   ├── final_data_preprocessing.ipynb       # Data cleaning & scaling
│   └── ml_models.ipynb                      # Model training & evaluation
├── data/
│   ├── student_performance_dataset.csv      # Raw dataset
│   ├── student_performance_preprocessed_train.csv
│   └── student_performance_preprocessed_test.csv
└── models/
    ├── student_model.pkl                    # Trained Logistic Regression
    └── scaler.pkl                           # Fitted RobustScaler + StandardScaler
```

## 🔍 Technical Details

### Preprocessing Pipeline (Critical)
The application uses a **two-stage scaling approach** replicated from training:

1. **RobustScaler** — Applied first (median/IQR normalization)
2. **StandardScaler** — Applied second (z-score normalization)

This exact sequence ensures consistency between training and inference. Skipping the RobustScaler step would cause predictions to drift toward 100% pass rate.

### Model Inference
Raw input → RobustScaler → Logistic Regression → Probability → Verdict

## ⚠️ Important Notes

- Predictions are **probabilistic estimates** based on historical data, not academic guarantees
- Model is trained on a specific student population; generalization to other contexts requires retraining
- Scaling parameters are computed from the training split and must be applied consistently
- Test accuracy (~93%) is measured on a held-out test set

## 📝 License

This project is provided as-is for educational purposes.

## 👨‍💻 Author

Youssef Ahmed

---

**Last Updated:** September 2026  
**Status:** Production Ready ✓
