import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/sales_data.csv", parse_dates=["date"])

le_store = LabelEncoder()
le_cat = LabelEncoder()
df["store_enc"] = le_store.fit_transform(df["store"])
df["category_enc"] = le_cat.fit_transform(df["category"])

features = ["store_enc","category_enc","units_sold","price_per_unit",
            "discount_%","is_weekend","month","day_of_week","quarter"]
X = df[features]
y = df["sales_amount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "preds": preds}
    print(f"\n📊 {name}")
    print(f"   MAE  : {mae:.2f}")
    print(f"   RMSE : {rmse:.2f}")
    print(f"   R²   : {r2:.4f}")

# Plot 9: Actual vs Predicted (Best model = Gradient Boosting)
best_preds = results["Gradient Boosting"]["preds"]
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Actual", color="blue")
plt.plot(best_preds[:100], label="Predicted", color="red", linestyle="--")
plt.title("Actual vs Predicted Sales (Gradient Boosting)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/09_actual_vs_predicted.png")
plt.close()
print("✅ Saved: 09_actual_vs_predicted.png")

# Plot 10: Model Comparison
plt.figure(figsize=(10, 5))
names = list(results.keys())
r2_scores = [results[n]["R2"] for n in names]
sns_colors = ["#4C72B0", "#55A868", "#C44E52"]
plt.bar(names, r2_scores, color=sns_colors)
plt.title("Model Comparison — R² Score")
plt.ylabel("R² Score")
plt.ylim(0, 1)
for i, v in enumerate(r2_scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/10_model_comparison.png")
plt.close()
print("✅ Saved: 10_model_comparison.png")

# Plot 11: Feature Importance (Random Forest)
rf = models["Random Forest"]
importances = rf.feature_importances_
plt.figure(figsize=(10, 5))
plt.barh(features, importances, color="teal")
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/11_feature_importance.png")
plt.close()
print("✅ Saved: 11_feature_importance.png")
