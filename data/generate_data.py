import numpy as np
import pandas as pd
import os

np.random.seed(42)

NUM_ACCOUNTS = 50
DAYS = 90

records = []

for i in range(NUM_ACCOUNTS):
    account_id = f"acc_{i:03d}"
    plan = np.random.choice(["free", "pro", "enterprise"])
    
    if plan == "free":
        base = np.random.uniform(1000, 5000)
    elif plan == "pro":
        base = np.random.uniform(5000, 20000)
    else:
        base = np.random.uniform(20000, 100000)

    for day in range(DAYS):
        noise = np.random.normal(0, base * 0.1)
        spike = base * np.random.choice([0, 3], p=[0.95, 0.05])
        tokens = max(0, base + noise + spike)
        
        records.append({
            "account_id": account_id,
            "plan": plan,
            "day": day,
            "tokens": round(tokens)
        })

os.makedirs("data", exist_ok=True)
df = pd.DataFrame(records)
df.to_csv("data/usage.csv", index=False)
print(f"Generated {len(df)} rows")
print(df.head(10))