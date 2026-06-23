# 📈 Sales Forecasting

**CodTech IT Solutions — ML Internship**

| Field             | Details             |
|-------------------|---------------------|
| **Intern ID**     | CITS5099            |
| **Full Name**     | Abhishek Prasad     |
| **No. of Weeks**  | 4                   |
| **Project Name**  | Sales Forecasting   |
| **Project Scope** | Machine Learning    |

---

## 📌 Project Overview

A machine learning project that forecasts future retail sales using historical data. The system analyzes sales trends, seasonality, and patterns across multiple stores and product categories, then trains regression models to predict future revenue.

---

## 🗂️ Project Structure
sales-forecasting/

├── data/

│   └── sales_data.csv

├── outputs/

├── generate_data.py

├── visualize.py

├── model.py

├── main.py

└── requirements.txt
## ⚙️ How to Run
pip install -r requirements.txt

python main.py

## 🤖 Models Used

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~1200 | ~1800 | ~0.78 |
| Random Forest | ~800 | ~1100 | ~0.91 |
| Gradient Boosting | ~750 | ~1050 | ~0.93 |

## 🛠️ Tech Stack
- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
