import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/sales_data.csv", parse_dates=["date"])

# Plot 1: Monthly Sales Trend
monthly = df.groupby(df["date"].dt.to_period("M"))["sales_amount"].sum().reset_index()
monthly["date"] = monthly["date"].astype(str)
plt.figure(figsize=(14, 5))
plt.plot(monthly["date"], monthly["sales_amount"], color="steelblue", linewidth=2)
plt.xticks(rotation=90)
plt.title("Monthly Total Sales Trend (2021-2023)")
plt.xlabel("Month"); plt.ylabel("Sales Amount (₹)")
plt.tight_layout()
plt.savefig("outputs/01_monthly_sales_trend.png")
plt.close()
print("✅ Saved: 01_monthly_sales_trend.png")

# Plot 2: Sales by Store
plt.figure(figsize=(10, 5))
store_sales = df.groupby("store")["sales_amount"].sum().sort_values(ascending=False)
sns.barplot(x=store_sales.index, y=store_sales.values, palette="viridis")
plt.title("Total Sales by Store")
plt.ylabel("Sales Amount (₹)")
plt.tight_layout()
plt.savefig("outputs/02_sales_by_store.png")
plt.close()
print("✅ Saved: 02_sales_by_store.png")

# Plot 3: Sales by Category
plt.figure(figsize=(10, 5))
cat_sales = df.groupby("category")["sales_amount"].sum().sort_values(ascending=False)
sns.barplot(x=cat_sales.index, y=cat_sales.values, palette="Set2")
plt.title("Total Sales by Category")
plt.ylabel("Sales Amount (₹)")
plt.tight_layout()
plt.savefig("outputs/03_sales_by_category.png")
plt.close()
print("✅ Saved: 03_sales_by_category.png")

# Plot 4: Weekend vs Weekday Sales
plt.figure(figsize=(7, 5))
wk = df.groupby("is_weekend")["sales_amount"].mean()
sns.barplot(x=["Weekday", "Weekend"], y=wk.values, palette="coolwarm")
plt.title("Average Sales: Weekday vs Weekend")
plt.ylabel("Avg Sales Amount (₹)")
plt.tight_layout()
plt.savefig("outputs/04_weekend_vs_weekday.png")
plt.close()
print("✅ Saved: 04_weekend_vs_weekday.png")

# Plot 5: Sales by Month (seasonality)
plt.figure(figsize=(10, 5))
month_avg = df.groupby("month")["sales_amount"].mean()
sns.lineplot(x=month_avg.index, y=month_avg.values, marker="o", color="orange")
plt.xticks(range(1,13), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
plt.title("Average Sales by Month (Seasonality)")
plt.ylabel("Avg Sales Amount (₹)")
plt.tight_layout()
plt.savefig("outputs/05_seasonality.png")
plt.close()
print("✅ Saved: 05_seasonality.png")

# Plot 6: Discount Impact on Sales
plt.figure(figsize=(10, 5))
disc = df.groupby("discount_%")["sales_amount"].mean()
sns.barplot(x=disc.index, y=disc.values, palette="magma")
plt.title("Impact of Discount on Average Sales")
plt.xlabel("Discount (%)"); plt.ylabel("Avg Sales Amount (₹)")
plt.tight_layout()
plt.savefig("outputs/06_discount_impact.png")
plt.close()
print("✅ Saved: 06_discount_impact.png")

# Plot 7: Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df[["units_sold","price_per_unit","sales_amount","discount_%","is_weekend","month","quarter"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/07_correlation_heatmap.png")
plt.close()
print("✅ Saved: 07_correlation_heatmap.png")

# Plot 8: Sales by Quarter
plt.figure(figsize=(7, 5))
q_sales = df.groupby("quarter")["sales_amount"].sum()
sns.barplot(x=["Q1","Q2","Q3","Q4"], y=q_sales.values, palette="Blues_d")
plt.title("Total Sales by Quarter")
plt.ylabel("Sales Amount (₹)")
plt.tight_layout()
plt.savefig("outputs/08_quarterly_sales.png")
plt.close()
print("✅ Saved: 08_quarterly_sales.png")
