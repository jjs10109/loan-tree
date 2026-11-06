import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

df = pd.read_csv("loan_applications.csv")

print(df.shape)
print(df.describe())

n_default = int((df["default"] == 1).sum())
share = n_default / len(df)
print(f"Rows with default == 1: {n_default}")
print(f"Share of all rows: {share:.6f}")

default_rate = df.groupby("late_payments_12mo")["default"].mean()
print(default_rate.to_frame(name="default_rate"))
