# Bank Marketing Project

This project predicts whether a bank customer will subscribe to a term deposit, using supervised machine learning on the UCI Bank Marketing dataset.

## Project Structure

- `data/` → Raw and cleaned datasets (excluded from Git)
- `src/` → All python files
- `models/` → Saved models 
- `results/` → EDA plots, profit curves
- `docs/` → decision logs, baseline findings

## 🔧 Setup Instructions

1. **Create and activate a virtual environment**  
   On Windows:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate

2. **Install dependencies**  
pip install -r requirements.txt

3. **Run Scripts**
python src/preprocess.py
python src/make_plots.py
python src/train_logreg.py

---

##  Current Progress

###  Dataset
- Bank dataset (45,211 rows, 17 features)
- Dropped `duration` column due to target leakage
- Target is imbalanced: only ~11% said "yes" to term deposit

###  Preprocessing
- Categorical features: OneHotEncoded
- Numeric features: StandardScaled
- Preprocessing pipeline saved as `preprocessor.pkl`

### Exploratory Data Analysis (EDA)
- `results/class_balance.png`: Shows class imbalance
- `results/age_histogram.png`: Age distribution
- `results/correlation_heatmap.png`: Numeric feature correlation

###  Baseline Model
- Logistic Regression (with `class_weight='balanced'`)
- Evaluated with Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Markdown summary: `docs/baseline_metrics.md`
- PR curve saved as: `results/pr_curve_logreg.png`
- Model saved to: `models/logreg.pkl`

---