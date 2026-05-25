# Customer-Churn-Prediction

An end-to-end machine learning project that predicts which e-commerce customers are at risk of churning, and more importantly, **why**. Built to demonstrate the full data science workflow: data cleaning, exploratory analysis, feature engineering, model selection, evaluation, and explainability.

## Motivation

This project was built around a real Data Scientist job description from Vox Media, which listed churn risk modelling as a core responsibility. The goal was to go beyond a basic notebook and produce something portfolio-ready, with rigorous evaluation, actionable outputs, and clear business interpretation.

## Dataset

- **Source:** E-commerce customer churn dataset
- **Size:** 50,000 customers × 25 features
- **Target:** `Churned` (binary: 0 = Active, 1 = Churned)
- **Class balance:** 71% Active / 29% Churned

Features cover customer demographics, behavioural engagement (login frequency, session duration, cart abandonment), purchase history, and support interactions.

## Project Structure

```
├── Data/
│   ├── raw/                        # Original dataset
│   └── processed/                  # Cleaned and feature-engineered CSVs
├── Models/
│   ├── xgb_churn_model.pkl         # Final trained XGBoost model
│   ├── random_forest_churn_model.pkl         # Final trained XGBoost model
│   └── model_features.pkl          # Feature list for inference
├── Notebooks/
│   ├── 01_data_cleaning.ipynb      # Null handling, type fixes, export
│   ├── 02_eda.ipynb                # Distributions, correlations, churn patterns
│   ├── 03_feature_engineering.ipynb # Engineered features (see below)
│   ├── 04_modeling.ipynb           # Logistic Regression + Random Forest baseline
│   ├── 05_xgboost_model.ipynb      # XGBoost, full evaluation pipeline
│   └── 06_shap_explainability.ipynb # SHAP global + individual explanations
├── Reports/
│   ├── shap_summary_bar.png
│   ├── shap_beeswarm.png
│   ├── shap_dependence_top3.png
│   ├── shap_waterfall_individual.png
│   ├── xgb_confusion_matrix.png
│   ├── xgb_feature_importance.png
│   ├── xgb_pr_curve.png
│   └── xgb_roc_curve.png
└── README.md
```

## Workflow

### 1. Data Cleaning (`01`)
- Imputed missing numerical values with column medians
- Imputed missing categorical values with column mode
- Exported cleaned dataset to `Data/processed/`

### 2. Exploratory Data Analysis (`02`)
- Confirmed 71/29 class split, moderate imbalance, handled in modelling
- Correlation heatmap revealed that churn is driven by **behavioural patterns**, not demographics
- Key EDA findings:
  - Customers with low login frequency churn significantly more
  - High cart abandonment rate is a strong early churn signal
  - Engagement metrics form a tightly correlated cluster, highly engaged users interact across multiple touchpoints consistently

### 3. Feature Engineering (`03`)
Five features were engineered from the raw data:

| Feature | Formula | Rationale |
|---|---|---|
| `Tenure_Months` | `Membership_Years × 12` | Normalises tenure to months |
| `Engagement_Score` | Sum of 5 engagement metrics | Single behavioural engagement index |
| `Purchase_Intensity` | `Total_Purchases / (Tenure_Months + 1)` | Purchase rate adjusted for tenure |
| `Recency_Score` | `1 / (Days_Since_Last_Purchase + 1)` | Higher = more recently active |
| `High_Risk` | `Cart_Abandonment > 0.5 AND Login_Frequency < 5` | Rule-based early warning flag |

`Engagement_Score` ranked in the **top 5 most important features** by SHAP, validating the feature engineering step.

### 4. Modelling (`04`, `05`)
Three models were trained and compared:

| Model | ROC-AUC | Churner Recall | Notes |
|---|---|---|---|
| Logistic Regression | ~0.79 | 0.73 | Baseline (linear only) |
| Random Forest | ~0.88 | 0.77 | Strong but untuned |
| **XGBoost** | **0.9293** | **0.85** | **Final model** |

**Why XGBoost won:** Sequential tree boosting learns the hard borderline cases (customers on the fence between staying and leaving) better than Random Forest's independent averaging approach.

**Handling class imbalance:** `scale_pos_weight = 2.46` was passed to XGBoost, weighting churners 2.46× higher during training, equivalent to `class_weight="balanced"` in sklearn. This is preferred over resampling as it avoids information loss.

**City was excluded** from the final model. With 40 unique values it generated 39 dummy columns (mostly noise). The EDA confirmed churn is driven by behaviour, not location. Country (8 values) was retained.

### 5. Evaluation (`05`)

```
Accuracy   : 91.5%
ROC-AUC    : 0.9293
Avg Precision : 0.9116

Confusion Matrix (10,000 test customers):
  True Positives  (churners caught)  : 2,456
  False Negatives (churners missed)  : 434
  False Positives (wrong alarms)     : 419
  True Negatives  (correct active)   : 6,691

5-Fold Cross-Validation ROC-AUC:
  Mean: 0.9274  |  Std: 0.0042
```

The low CV standard deviation (0.0042) confirms the model generalises well, it learned real patterns, not quirks of a single train/test split.

### 6. Explainability: SHAP (`06`)

SHAP (SHapley Additive exPlanations) was used to explain both global patterns and individual predictions.

**Top churn drivers:**

| Feature | Direction | Business meaning |
|---|---|---|
| `Customer_Service_Calls` | ↑ high = more churn | Frustration signal, frequent support contact precedes leaving |
| `Lifetime_Value` | U-shaped | Both low and high LTV customers churn at high rates (see below) |
| `Cart_Abandonment_Rate` | ↑ high = more churn | Disengagement from purchase intent |
| `Engagement_Score` | ↓ low = more churn | Customers not interacting across touchpoints are at risk |
| `Days_Since_Last_Purchase` | ↑ high = more churn | Recency is a strong retention signal |

**Notable finding: Lifetime Value is U-shaped:**

A simple correlation suggested high LTV = more churn, which seemed wrong. Digging into the quartile-level SHAP values revealed a U-shaped relationship:

```
Q1 (low LTV)   - 42% churn rate   - disengaged from the start
Q2             - 20% churn rate
Q3             - 14% churn rate
Q4 (high LTV)  - 39% churn rate   - high-value, actively targeted by competitors
```

Both ends of the LTV spectrum churn at high rates, for different reasons. Mid-LTV customers are the most loyal segment. This kind of non-linear pattern is exactly what SHAP surfaces that a simple feature importance score cannot.

## Key Takeaways

- **Behaviour beats demographics.** Login frequency, cart abandonment, and engagement score outperform age, gender, and location as churn predictors.
- **SHAP makes the model actionable.** Rather than a black-box score, each at-risk customer can be explained: *why* are they likely to churn, and which specific behaviours to address.
- **Engineered features contributed.** `Engagement_Score`, a composite of 5 raw metrics, ranked in the top 5 SHAP features, justifying the feature engineering step.
- **The model generalises.** CV std of 0.0042 across 5 folds means the 0.9274 AUC is reliable, not a lucky split.

## How to Run

```bash
# Clone the repo
git clone https://github.com/Chosen-Data26/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn joblib

# Run notebooks in order
jupyter notebook
```

Open notebooks 01 - 06 in sequence. Each notebook is self-contained and saves its outputs before the next picks up.

## Tools & Libraries

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `SHAP` · `matplotlib` · `seaborn` · `joblib`