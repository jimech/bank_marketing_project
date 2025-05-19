import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


os.makedirs("results", exist_ok=True)

# 1. Load 
df = pd.read_csv("data/bank-clean.csv")

# 2. Class balance plot- Target 'y' Variable Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="y", palette="pastel") #showing how many "yes" and "no" responses
plt.title("Target Variable Distribution")
plt.xlabel("Subscribed to Term Deposit")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/class_balance.png")
plt.close()
print(" Saved: results/class_balance.png")

# 3. Age histogram
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="age", bins=30, kde=True, color="skyblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("results/age_histogram.png")
plt.close()
print(" Saved: results/age_histogram.png")

# 4. Correlation heatmap (numeric columns only)
numeric_cols = df.select_dtypes(include=["int64", "float64"])
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png")
plt.close()
print(" Saved: results/correlation_heatmap.png")
