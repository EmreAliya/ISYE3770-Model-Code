import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt  # <-- added

df = pd.read_csv("college4.csv")

print(df.head())
print(df.columns)

y = df["SPEND"]

X = df[["SAT", "TOP10", "ACCRATE", "PHD", "RATIO", "GRADRATE", "ALUMNI"]]
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print("\n=== Full model summary ===")
print(model.summary())

current_X = X.copy()
current_model = model

while True:
    pvals = current_model.pvalues.drop("const", errors="ignore")
    worst_p = pvals.max()
    worst_var = pvals.idxmax()

    print("\nCurrent variables:", list(current_X.columns))
    print("Worst p-value:", worst_p, "for variable:", worst_var)

    if worst_p <= 0.05:
        print("\nAll remaining variables are significant at alpha = 0.05.")
        break

    current_X = current_X.drop(columns=[worst_var])
    current_model = sm.OLS(y, current_X).fit()

print("\n=== Final model summary ===")
print(current_model.summary())

hsu_values = {
    "SAT": 1100,
    "TOP10": 50,
    "ACCRATE": 50,
    "PHD": 95,
    "RATIO": 10,
    "GRADRATE": 70,
    "ALUMNI": 30,
}

hsu_used = {col: hsu_values[col] for col in current_X.columns if col != "const"}
hsu_df = pd.DataFrame([hsu_used])
hsu_df = sm.add_constant(hsu_df, has_constant="add")

hsu_pred = current_model.predict(hsu_df)[0]

print("\nPredicted SPEND for HSU:", round(hsu_pred, 2))
print("Actual SPEND for HSU:   16000.00")
print("Difference (Actual - Predicted):", round(16000 - hsu_pred, 2))

summary_text = current_model.summary().as_text()

plt.figure(figsize=(10, 6))
plt.text(0.01, 0.99, summary_text, family="monospace", fontsize=8, va="top")
plt.axis("off")
plt.tight_layout()
plt.savefig("regression_summary.png", dpi=300)
plt.close()

pred = current_model.predict(current_X)

plt.figure(figsize=(6, 5))
plt.scatter(pred, y)
plt.xlabel("Predicted SPEND")
plt.ylabel("Actual SPEND")
plt.title("Predicted vs Actual Educational Spending")

min_val = min(pred.min(), y.min())
max_val = max(pred.max(), y.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.tight_layout()
plt.savefig("pred_vs_actual.png", dpi=300)
plt.close()

sat = df["SAT"]
sat_X = sm.add_constant(sat)
sat_model = sm.OLS(y, sat_X).fit()
sat_pred = sat_model.predict(sat_X)

plt.figure(figsize=(6, 5))
plt.scatter(sat, y, label="Data")
plt.plot(sat, sat_pred, label="Fit line")
plt.xlabel("SAT")
plt.ylabel("SPEND")
plt.title("SAT vs SPEND with Regression Line")
plt.legend()

plt.tight_layout()
plt.savefig("sat_vs_spend.png", dpi=300)
plt.close()
