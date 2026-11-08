# Credit Card Fraud Detection with Anomaly Detection

A machine learning project that treats credit card fraud as an **anomaly detection** problem: instead of learning a boundary between two balanced classes, the models learn what normal transactions look like and flag the ones that deviate. A One-Class SVM is evaluated first, then compared against an Isolation Forest (also label-free) and a supervised gradient boosting model used as a reference.

## Overview

- **Task:** detect fraudulent card transactions in a highly imbalanced dataset (about 1.7% fraud)
- **Models:** One-Class SVM, Isolation Forest, and a supervised `HistGradientBoostingClassifier` reference
- **Evaluation:** ROC-AUC and PR-AUC from continuous anomaly scores, precision/recall/F1 on the fraud class, score distributions, and a 2D PCA visualization
- **Main finding:** the Isolation Forest, trained without fraud labels, reaches a PR-AUC about 2.6x higher than the One-Class SVM and close to the supervised model

## Dataset

`credit_card_fraud_2026.csv`: 20,000 transactions, 26 columns, no missing values.

Source: `<add dataset link here>`

| Group | Features |
|---|---|
| Transaction | `amount_usd`, `merchant_category`, `channel`, `time_of_day_hour`, `day_of_week`, `is_foreign_transaction`, `is_new_merchant` |
| Card and customer | `card_type`, `card_age_months`, `customer_age`, `account_balance_usd`, `prior_disputes` |
| Authentication and device | `auth_method`, `device_type`, `cvv_retry_count`, `used_vpn`, `ip_country_mismatch`, `billing_shipping_mismatch` |
| Behavioral and risk signals | `hours_since_last_txn`, `txn_count_last_24h`, `distance_from_home_km`, `velocity_score`, `merchant_risk_score`, `is_ai_generated_scam_attempt` |
| Identifier (dropped) | `transaction_id` |
| Target | `is_fraud` (about 1.7% of transactions) |

## Methodology

1. **Exploration:** data types, missing values, category values and summary statistics.
2. **Preprocessing** (nothing is learned from the test set):
   - drop `transaction_id`
   - stratified 80/20 train/test split (`random_state=42`): 16,000 train, 4,000 test (3,932 normal, 68 fraud)
   - frequency encoding of categorical features, computed on the training split only
   - boolean features cast to 0/1
   - numeric features standardized with a `StandardScaler` fit on the training split
3. **Validation split:** the training data is split again (80/20, stratified) into a fitting set and a validation set, used for hyperparameter search and threshold selection.
4. **One-Class SVM:** trained on normal transactions only. A grid search over `nu` and `gamma` is scored by validation PR-AUC, and the final model uses the grid's best `nu` (0.1) with `gamma='scale'`. Anomaly scores are the negated `decision_function`.
5. **Baselines:**
   - Isolation Forest (300 trees), trained on normal transactions only
   - `HistGradientBoostingClassifier` with `class_weight='balanced'`, trained on the labeled fitting set
6. **Evaluation:** all models are compared on the held-out test set using ROC-AUC and PR-AUC.

## Results

### Model comparison (test set)

| Model | Uses fraud labels | ROC-AUC | PR-AUC |
|---|---|---|---|
| One-Class SVM | No | 0.787 | 0.070 |
| Isolation Forest | No | 0.875 | 0.182 |
| Gradient boosting (supervised) | Yes | 0.900 | 0.215 |

The no-skill PR-AUC equals the fraud rate, about 0.017.

### One-Class SVM at a fixed threshold (fraud class, test set)

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| Model's default boundary | 0.06 | 0.37 | 0.11 |
| Tuned on validation (max F1) | 0.08 | 0.29 | 0.13 |

### Key findings

- **The One-Class SVM is the weak link, not the anomaly detection framing.** The Isolation Forest never sees fraud labels yet reaches roughly 85% of the supervised model's PR-AUC.
- **Supervised learning wins, but modestly** (PR-AUC 0.215 vs 0.182). With labels available it is the best option, but the margin is small.
- **A plausible explanation** (not tested here): the One-Class SVM draws a radial, distance-based boundary, while fraud appears shifted in one direction and sits mostly inside the dense cluster in the PCA projection.
- **Precision stays low for every model**, so flagged transactions would still need manual review in practice.

## Limitations and Next Steps

Known limitations:

- The test set has only 68 fraud cases from a single split, so the metrics are noisy and the gap between Isolation Forest and the supervised model may be within noise.
- The final One-Class SVM uses the grid's best `nu` (0.1) but `gamma='scale'` instead of the grid winner's `gamma=0.1`, and that winner sits on the edge of the search grid.
- Isolation Forest and gradient boosting use default hyperparameters, and no threshold was chosen for them.

Planned improvements:

- [ ] Widen the grid and use stratified cross-validation, with confidence intervals on the metrics
- [ ] Tune the Isolation Forest and choose a threshold (for example precision at a fixed recall)
- [ ] Feature importance analysis (such as `permutation_importance`)
- [ ] One-hot encoding instead of frequency encoding, and an autoencoder trained on normal data
- [ ] Error analysis of fraud cases missed by all models

## Project Structure

```
.
├── anomaly_detection_with_fraud_dataset.ipynb   # analysis and modelling notebook
├── data/
│   └── credit_card_fraud_2026.csv               # dataset (not included)
└── README.md
```

## Getting Started

**Requirements:** Python 3.8+ and scikit-learn 1.2 or later (needed for `class_weight` in `HistGradientBoostingClassifier`).

```
pandas
numpy
scikit-learn>=1.2
matplotlib
jupyter
```

**Run it**

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install pandas numpy "scikit-learn>=1.2" matplotlib jupyter
jupyter notebook anomaly_detection_with_fraud_dataset.ipynb
```

Before running, update the path in the data-loading cell (`pd.read_csv(...)`) to point to your copy of the dataset, and run the cells in order.

**Troubleshooting**

On recent Windows versions, joblib may print a harmless `wmic` / "Could not find the number of physical cores" warning. To silence it, set `LOKY_MAX_CPU_COUNT` before importing scikit-learn:

```python
import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
```

## Author

**Khaleed**
[GitHub](https://github.com/adeleke1508) · [LinkedIn](www.linkedin.com/in/khaleed-ismail-14b572298)
