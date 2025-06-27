# ✈️ Flight Fare Prediction for Bangladesh International Routes using Machine Learning

This project focuses on predicting international flight ticket prices originating from Bangladesh using machine learning techniques. Built using Python and Streamlit, the system helps travelers, analysts, and agencies make better fare-related decisions based on historical patterns.

---

## 📌 Project Overview

- 🔍 **Objective**: Predict accurate flight prices using features such as airline, departure/arrival time, stopovers, duration, etc.
- 📊 **Approach**: Machine Learning regression modeling with full EDA and preprocessing pipeline
- 🌍 **Scope**: Routes originating from Bangladesh to international destinations

---

## 🧰 Tech Stack

| Technology    | Purpose                           |
|---------------|-----------------------------------|
| **Python**    | Core programming language         |
| **Pandas**    | Data manipulation & preprocessing |
| **Matplotlib & Seaborn** | Exploratory Data Analysis (EDA) |
| **Scikit-learn** | Model building & evaluation    |
| **XGBoost**   | Final regression model (best performer) |
| **Streamlit** | Interactive dashboard & UI        |

---

## 📂 Project Structure

| File / Folder              | Description                                      |
|----------------------------|--------------------------------------------------|
| `Flight_Price_Dataset.csv` | Cleaned dataset for modeling                     |
| `Airline_Streamlit_Code.py`| Streamlit app code with user input + prediction  |
| `EDA_Notebook.ipynb`       | Notebook for analysis, visualization, modeling   |
| `xgb_model.pkl`            | Trained ML model (XGBoost)                       |
| `README.md`                | Project documentation                            |
| `images/`                  | Screenshots and visual assets                    |

---

## 🔍 Key Features

- ✅ Feature engineering on categorical variables (Airline, Source, Destination)
- 🧹 Complete data preprocessing: missing values, label encoding, datetime parsing
- 📈 Correlation analysis, duration normalization, trend visualizations
- 🤖 Regression model comparison (Linear, Random Forest, XGBoost)
- 📊 MAE, MSE, RMSE, R² evaluation
- 🖥️ Live Streamlit interface for fare prediction

---

## 📸 Screenshots
![Screenshot 2025-06-27 185334](https://github.com/user-attachments/assets/fa148c42-1677-4c40-bf92-6bf68004911f)

![Screenshot 2025-06-27 185447](https://github.com/user-attachments/assets/163c8f41-d40e-477f-976f-2e1a1bead793)


---

## 🚀 Run App_link
https://balram-airline-flight-fare-predictor.streamlit.app


