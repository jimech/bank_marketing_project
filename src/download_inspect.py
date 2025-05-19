import pandas as pd

# 1.Load dataset
df = pd.read_csv("data/bank-full.csv", sep=";") #UCI dataset use ; 

# 2. We drop duration column- it leaks target info. (call duration is known after call is made)
if 'duration' in df.columns:
    df = df.drop(columns=["duration"])
    print("\nDropped 'duration' column.")

# 3. Show shape
print("Dataset Shape:", df.shape)

# 4. Show column types and non-null counts
print("\nInfo:")
print(df.info())


# 5. Show class distribution
print("\n Target variable distribution ('y'):")
print(df['y'].value_counts())
print("(% of total):")
print(df['y'].value_counts(normalize=True).round(3))

# 6.Check for missing values (NaNs)
total_missing = df.isna().sum().sum()      

if total_missing:                         
    print("\nMissing values  in dataset:")
    print(df.isna().sum())                 
else:
    print("\nNo NaN values found in the dataset ")


# 7. It checks for 'unknown' strings in object columns. not NaNs, but real categories. 
# count and logs them to document 
unknown_log = []

print("\n Columns containing 'unknown':")
for col in df.select_dtypes(include="object"):
    if "unknown" in df[col].unique():
        count = (df[col] == "unknown").sum()
        print(f"- {col}: {count} 'unknown' values")
        unknown_log.append(f"{col}: {count}")

# Save log to a text file
with open("docs/unknown_counts.txt", "w") as f:
    f.write(f"Total rows: {len(df)}\n\n")
    f.write("Unknown values per column:\n")
    f.write("\n".join(unknown_log))

print(" Saved unknown value log to 'docs/unknown_counts.txt'")

# 8. Show unique values per categorical column
print("\n Unique values per categorical column:")
for col in df.select_dtypes(include="object"):
    print(f"\nColumn: {col}")
    print(df[col].unique())

# 9. Clean dataset (without duration column)
df.to_csv("data/bank-clean.csv", index=False)
print("\n Cleaned dataset saved as 'data/bank-clean.csv'")
