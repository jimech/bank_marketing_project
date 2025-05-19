import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def main():
    # 1. Load cleaned dataset
    df = pd.read_csv("data/bank-clean.csv")

    # 2. Split into features (X) and target (y)
    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})  # Convert to binary

    # 3. Auto-detect columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    # 4. Define transformers
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num_scaler = StandardScaler()

    # 5. Combine with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_encoder, categorical_cols),
            ("num", num_scaler, numeric_cols)
        ]
    )

    # 6. Fit on full data — transform later in train script
    preprocessor.fit(X)

    # 7. Save the pipeline for reuse
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    print(" Preprocessing pipeline saved to 'models/preprocessor.pkl'")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numeric columns: {numeric_cols}")

if __name__ == "__main__":
    main()
