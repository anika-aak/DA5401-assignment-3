# DA5401 A3: Addressing Class Imbalance with Clustering and Resampling

## Objective
This project addresses class imbalance in the Credit Card Fraud Detection dataset using Logistic Regression. Four training sets were compared:

1. **Baseline** (imbalanced data)  
2. **SMOTE** (naive oversampling)  
3. **CBO** (Clustering-Based Oversampling)  
4. **CBU** (Clustering-Based Undersampling)  

Performance was evaluated with precision, recall, F1-score (fraud class), ROC-AUC, and PR-AUC.

## Dataset
- Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- Highly imbalanced: fraud ≈ 0.17% of transactions.

## Methods
- **Baseline:** Logistic Regression on original split, test set remains imbalanced.  
- **SMOTE:** Synthetic samples generated between minority neighbors.  
- **CBO:** Minority clustered with KMeans, oversampling performed within each cluster.  
- **CBU:** Majority clustered, undersampling performed proportionally across clusters.  

## Results (Fraud Class)

| Model    | Precision | Recall | F1   | ROC-AUC | PR-AUC |
|----------|-----------|--------|------|---------|--------|
| Baseline | 0.85      | 0.64   | 0.73 | 0.956   | 0.719  |
| SMOTE    | 0.06      | 0.89   | 0.11 | 0.972   | 0.712  |
| CBO      | 0.062     | 0.89   | 0.116| 0.973   | 0.707  |
| CBU      | 0.062     | 0.878  | 0.115| 0.975   | 0.603  |

## Key Insights
- Baseline performs unusually well, likely due to class weighting, but recall is lower than with resampling.  
- SMOTE increases recall sharply but reduces precision.  
- CBO balances recall and precision better than SMOTE and is the strongest overall.  
- CBU typically improves precision, but here it behaved similarly to SMOTE.

## Requirements
```bash
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.5.1
imbalanced-learn==0.12.3
matplotlib==3.9.2
kaggle==1.6.17
