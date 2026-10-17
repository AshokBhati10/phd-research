| Model | Metric | Winner | Value | Family |
|---|---|---|---|---|
| LogisticRegression | Accuracy | No Balancing | 0.8330 | Baseline |
| LogisticRegression | Recall | ADASYN | 0.7224 | Basic oversampling |
| LogisticRegression | F1 | Random Oversampling | 0.4435 | Basic oversampling |
| LogisticRegression | ROC_AUC | ENN | 0.7715 | Cleaning & hybrid |
| LogisticRegression | PR_AUC | Tomek Links | 0.4011 | Cleaning & hybrid |
| RandomForest | Accuracy | Tomek Links | 0.8349 | Cleaning & hybrid |
| RandomForest | Recall | ADASYN | 0.7716 | Basic oversampling |
| RandomForest | F1 | Random Oversampling | 0.4441 | Basic oversampling |
| RandomForest | ROC_AUC | Random Oversampling | 0.7777 | Basic oversampling |
| RandomForest | PR_AUC | Tomek Links | 0.4139 | Cleaning & hybrid |
| XGBoost | Accuracy | No Balancing | 0.8354 | Baseline |
| XGBoost | Recall | ADASYN | 0.7689 | Basic oversampling |
| XGBoost | F1 | Random Oversampling | 0.4453 | Basic oversampling |
| XGBoost | ROC_AUC | Tomek Links | 0.7803 | Cleaning & hybrid |
| XGBoost | PR_AUC | Random Oversampling | 0.4176 | Basic oversampling |