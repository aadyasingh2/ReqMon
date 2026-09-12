import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1000

# Reference dataset
ref_df = pd.DataFrame({
    "transaction_amount": np.random.exponential(scale=50, size=n_samples),
    "credit_score": np.random.normal(700, 50, size=n_samples),
    "account_age_months": np.random.uniform(1, 120, size=n_samples),
    "num_recent_transactions": np.random.poisson(lam=5, size=n_samples),
})

# Production dataset with drift in transaction_amount and num_recent_transactions
prod_df = pd.DataFrame({
    "transaction_amount": np.random.exponential(scale=120, size=n_samples),  # Drifted
    "credit_score": np.random.normal(700, 50, size=n_samples),              # Stable
    "account_age_months": np.random.uniform(1, 120, size=n_samples),         # Stable
    "num_recent_transactions": np.random.poisson(lam=12, size=n_samples),    # Drifted
})

ref_df.to_csv("data/reference.csv", index=False)
prod_df.to_csv("data/production.csv", index=False)
print("Generated data/reference.csv and data/production.csv successfully.")
