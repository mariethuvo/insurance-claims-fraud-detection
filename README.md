# Insurance Claims Fraud Detection

Binary classification project comparing ensemble methods to prioritize which insurance claims a fraud investigation team should review, built as a case study on Pinnacle Insurance Group, a mid-sized P&C insurer.

## Business Problem

Pinnacle's Special Investigations Unit (SIU) can only investigate 500 claims per month out of roughly 10,000 filed. Their current hit rate — the share of investigated claims that turn out to actually be fraudulent — was 18%, meaning most investigation capacity was being spent on clean claims while real fraud went unreviewed.

The hard part isn't just detecting fraud, it's ranking claims correctly under a fixed capacity constraint. Fraud made up only 6.04% of claims, so a naive model that predicts "not fraud" for everything would score ~94% accuracy while catching nothing — accuracy is the wrong metric here. The real objective was **precision at the SIU's top ~5% capacity**: of the top 125 highest-risk claims (in a 2,500-claim test set), how many are genuinely fraudulent.

## Data

10,000 synthetic claims (one month's volume), with claim amount, premium, policy age, witness/police-report/documentation flags, and a prior-fraud history flag.

## Key EDA Findings

- Fraudulent claims average $17,835 vs. $7,492 for legitimate claims
- Fraud claims are less likely to have witnesses or a police report
- A prior fraud flag is over 5x more common among fraudulent claims
- Fraudulent claims take longer to report (6.3 days avg. vs. 4.4 days)
- Fraud is only 6.04% of the dataset — accuracy is not a usable metric here

## Feature Engineering

| Feature | Rationale |
|---|---|
| `claim_to_premium_ratio` | Fraudulent claims tend to be disproportionately large relative to the policy's premium |
| `corroboration_score` | Combines police report, witnesses, and documentation completeness — fraud claims score lower on all three individually, so combining amplifies the signal |
| `policy_age_bucket` | Newer policies show higher fraud rates; some fraudsters open a policy specifically to file a claim |
| `reporting_delay_flag` | Delayed reporting correlates with fraud, possibly reflecting time needed to construct a false narrative |
| `financial_stress_composite` | Combines signals that individually showed elevated fraud rates in EDA |

## Approach

Trained and compared five models: Decision Tree and Logistic Regression as baselines, then Random Forest, Gradient Boosting, and XGBoost. Evaluated with AUC-ROC and Average Precision rather than accuracy, since Average Precision is far more informative on a 6% positive-class problem. A Stacking Classifier (RF + GB + XGBoost, with a logistic regression meta-learner) was also tested as a bonus comparison.

| Model | AUC-ROC | Average Precision |
|---|---|---|
| Decision Tree (baseline) | 0.8496 | 0.4000 |
| Logistic Regression (baseline) | 0.8144 | 0.2530 |
| Random Forest | 0.8908 | 0.4499 |
| Gradient Boosting | 0.8874 | 0.5094 |
| **XGBoost** | **0.8914** | **0.5347** |
| Stacking Ensemble | 0.8961 | 0.5237 |

XGBoost was selected as the final model based on highest Average Precision. The stacking ensemble scored marginally higher on AUC but lower on Average Precision, and added meaningful training time and complexity for no real gain — worth noting as a case where the "fancier" approach wasn't the right call.

## Threshold Optimization

Rather than using the default 0.5 classification threshold, I tuned the decision threshold to match the SIU's actual capacity (top 5%, ~125 claims per 2,500-claim test set):

- **Optimal threshold:** 0.287
- **Precision at this threshold:** 56.8% (SIU hit rate)
- **Recall at this threshold:** 47.0% of all fraud caught
- **Lift vs. random selection:** 9.4x

## Business Impact

| | Current System | Model-Based System |
|---|---|---|
| Investigations/month | 500 | 500 |
| Frauds caught | 90 | 284 |
| Hit rate | 18.0% | 56.8% |

Same investigation budget, over 3x more fraud caught. Projected to ~2,328 additional frauds caught annually, for an estimated **$34.9M in annual financial impact**.

## What I'd do differently

The stacking ensemble result was a useful reminder not to assume more complexity means a better model — it's worth validating against the actual metric that matters (Average Precision here) before defaulting to the more sophisticated option. I'd also want to test threshold stability across multiple months of data before trusting 0.287 as a fixed cutoff, since fraud patterns and claim volume both drift over time.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

## Files

- `VoMarie_InsuranceClaimFraudDetection.ipynb` has full notebook: EDA, feature engineering, model comparison, threshold optimization, and business impact analysis
