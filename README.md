# Credit Risk Modeling (CRM) - PD · LGD · EAD · Expected Loss

> As my most challenging and complex project to date, this required a powerful combination of Python expertise and rigorous statistical theory to deliver an end-to-end Credit Risk Model. Built on the LendingClub 2007–2014 dataset (~466k loans), it comprehensively implements all three Basel II pillars of Expected Loss: Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD)

---

## 📌 Project Overview

| Component | Method | Output |
|-----------|--------|--------|
| **PD** - Probability of Default | Logistic Regression + WoE Binning | Score 300–850 + Default probability |
| **LGD** - Loss Given Default | Two-stage: Logistic → Linear Regression | Recovery rate [0, 1] |
| **EAD** - Exposure at Default | Linear Regression (CCF) | Remaining exposure [0, funded amount] |
| **Expected Loss** | EL = PD × LGD × EAD | Dollar value at risk |

---

## 🗂️ Repository Structure

```
├── CRM_Data_Prep.ipynb      # Feature engineering, WoE binning, train/test split
├── CRM_PD_Model.ipynb       # PD model, scorecard, AUROC / Gini / KS evaluation
├── CRM_LGD_EAD.ipynb        # LGD (2-stage) + EAD model, Expected Loss calculation
├── CRM_Variables.ipynb      # Features and Categories to build models
└── README.md
```

---

## 🔧 Pipeline

```
Raw CSV (466k loans)
        │
        ▼
┌───────────────────┐
│  Data Prep        │  Date parsing · Dummy encoding · WoE binning
│ (Data_Prep.ipynb) │  NaN imputation · Train/test split (80/20, stratified)
└─────────┬─────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
 PD Model   LGD + EAD Model
    │            │
    └──────┬─────┘
           ▼
    Expected Loss = PD × LGD × EAD
```

---

## 1️⃣ Data Preparation - Feature Engineering

The raw dataset contains mixed types (strings, dates, categoricals). Key transformations:

**Date features** - converted to "months since" relative to reference date:
```python
loan_data['mths_since_issue_d'] = np.floor(
    (pd.to_datetime('2026-01-01') - loan_data['issue_d']).dt.days / 30.44
)
```

**Weight of Evidence (WoE) binning** - all continuous variables are discretised into bins optimised by WoE before model training. The WoE framework quantifies how strongly each bin predicts default:

| IV Range | Predictive Power |
|----------|-----------------|
| < 0.02 | None |
| 0.02 – 0.10 | Weak |
| 0.10 – 0.30 | Medium |
| 0.30 – 0.50 | Strong |
| > 0.50 | Suspiciously high |

**Variables engineered (selected):**

| Variable | Bins | Notes |
|----------|------|-------|
| `int_rate` | 5 bins | `<9.5%` → `>20.3%` |
| `annual_inc` | 12 bins | `<20K` → `>140K` |
| `dti` | 10 bins | `≤1.4` → `>35` |
| `mths_since_issue_d` | 8 bins | Vintage effect |
| `emp_length_int` | 6 bins | 0 → 10 years |
| `addr_state` | 12 grouped bins | States grouped by WoE similarity |
| `mths_since_last_delinq` | 5 bins + N/A flag | Missing treated as separate category |

---

## 2️⃣ PD Model - Probability of Default

**Target variable:** `good_bad` (1 = good borrower, 0 = defaulted)

Default is defined as any of:
- `Charged Off`
- `Late (31-120 days)`
- `Default`
- `Does not meet the credit policy. Status: Charged Off`

**Model:** Logistic Regression with custom p-value calculation (Cramér-Rao approximation), trained on ~116 WoE-binned dummy features.

```python
class LogisticRegression_with_p_values:
    def fit(self, X, y):
        self.model.fit(X, y)
        # Fisher information matrix + Ridge penalty for stability
        F_ij = np.dot((X / denom).T, X) + 1e-6
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        self.p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
```

### Scorecard (300–850 scale)

The logistic regression coefficients are linearly scaled to a credit scorecard in the 300–850 range (same as FICO), allowing human-readable interpretation:

```
Score = Σ(coefficient_i × feature_i) scaled to [300, 850]
```

Higher score = lower default probability.

### Model Evaluation

| Metric | Benchmark |
|--------|-----------|
| AUROC | > 0.7 = good model |
| Gini | `2 × AUROC − 1` |
| KS Statistic | Peak distance between Good and Bad cumulative curves |

---

## 3️⃣ LGD Model - Loss Given Default

Trained only on **defaulted loans** (`Charged Off`).

**Target:** `recovery_rate = recoveries / funded_amount` - capped to [0, 1]

Two-stage model (Beta regression approximation in Python):

```
Stage 1 - Logistic Regression
    Is recovery_rate > 0?
    → Predicts probability of any recovery

Stage 2 - Linear Regression
    Given recovery_rate > 0, what is the actual rate?
    → Trained only on loans with positive recovery

LGD = 1 − (Stage1_prediction × Stage2_prediction)
```

This approach handles the spike at 0 in the recovery rate distribution - a common characteristic of LGD data.

---

## 4️⃣ EAD Model - Exposure at Default

**Target:** Credit Conversion Factor `CCF = (funded_amount − total_principal_received) / funded_amount`

CCF represents what fraction of the original loan amount is still outstanding at default.

```python
loan_data_defaults['CCF'] = (
    loan_data_defaults['funded_amnt'] - loan_data_defaults['total_rec_prncp']
) / loan_data_defaults['funded_amnt']

# EAD in dollar terms:
loan_data['EAD'] = loan_data['CCF'] * loan_data['funded_amnt']
```

Linear regression is used directly (no two-stage needed - CCF is more uniformly distributed than recovery rate).

---

## 5️⃣ Expected Loss

```python
loan_data['EL'] = loan_data['PD'] * loan_data['LGD'] * loan_data['EAD']
```

| Column | Meaning |
|--------|---------|
| `PD` | Probability the borrower defaults |
| `LGD` | Fraction of EAD that won't be recovered |
| `EAD` | Dollar amount at risk at time of default |
| `EL` | Expected dollar loss per loan |

---

## ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

| Library | Usage |
|---------|-------|
| `pandas` | Data manipulation, feature engineering |
| `numpy` | Numerical operations, WoE calculations |
| `scikit-learn` | Logistic & Linear Regression, train/test split |
| `scipy.stats` | p-value computation |
| `matplotlib` / `seaborn` | WoE plots, ROC curves |
| `pickle` | Model serialisation |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model

# 2. Install dependencies
pip install pandas numpy scikit-learn scipy matplotlib seaborn jupyter

# 3. Place the dataset
# Download loan_data_2007_2014.csv and put it in the project root

# 4. Run in order
jupyter notebook CRM_Data_Prep.ipynb    # generates train/test CSVs
jupyter notebook CRM_PD_Model.ipynb     # trains PD model, saves pd_model.sav
jupyter notebook CRM_LGD_EAD.ipynb      # trains LGD/EAD, computes Expected Loss
```

> **Dataset:** [LendingClub Loan Data 2007–2014](https://www.kaggle.com/datasets/wordsforthewise/lending-club) on Kaggle (~466k rows, ~75 features)

---

## 📊 Key Results

| Model | Metric | Interpretation |
|-------|--------|---------------|
| PD | AUROC > 0.7 | Good discriminatory power |
| PD | Gini = 2×AUROC−1 | Measures concentration of bad loans |
| PD | KS statistic | Max separation of Good vs Bad CDF |
| LGD Stage 1 | Accuracy + AUROC | Classification of zero vs non-zero recovery |
| LGD Stage 2 | Pearson correlation | Predicted vs actual recovery rate |
| EAD | Pearson correlation | Predicted vs actual CCF |

---

## 📝 Known Limitations & Future Improvements

- **Beta regression for LGD** - the two-stage OLS approximation works but a proper Beta regression (available in R's `betareg`) would model the [0,1] bounded distribution more precisely.
- **Time-based validation** - current split is random; a vintage-based split (train on older loans, test on newer) would better simulate real deployment.
- **Feature leakage risk** - some features (`int_rate`, `grade`) are set by the lender partly based on creditworthiness, introducing potential endogeneity.
- **WoE bins are static** - bins were determined manually from fine classing; automating this with monotonicity constraints would improve reproducibility.

