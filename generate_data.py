import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="D")
stores = ["Store_A", "Store_B", "Store_C", "Store_D", "Store_E"]
categories = ["Electronics", "Clothing", "Food", "Sports", "Beauty"]

rows = []
for date in dates:
    for store in stores:
        for cat in categories:
            base = np.random.randint(5000, 20000)
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            weekend = 1.2 if date.dayofweek >= 5 else 1.0
            trend = 1 + 0.0002 * (date - dates[0]).days
            noise = np.random.normal(1, 0.05)
            sales = base * seasonal * weekend * trend * noise
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "store": store,
                "category": cat,
                "units_sold": np.random.randint(10, 200),
                "price_per_unit": round(np.random.uniform(50, 500), 2),
                "sales_amount": round(sales, 2),
                "discount_%": np.random.choice([0, 5, 10, 15, 20]),
                "is_weekend": int(date.dayofweek >= 5),
                "month": date.month,
                "day_of_week": date.dayofweek,
                "quarter": date.quarter,
            })

df = pd.DataFrame(rows)
df.to_csv("data/sales_data.csv", index=False)
print(f"✅ Dataset created: {len(df)} rows saved to data/sales_data.csv")
