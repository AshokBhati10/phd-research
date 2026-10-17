| Family | Model | n_techniques | F1_mean | Recall_mean | Accuracy_mean | ROC_AUC_mean | PR_AUC_mean | F1_best_technique |
|---|---|---|---|---|---|---|---|---|
| Baseline | LogisticRegression | 1 | 0.2166 | 0.1354 | 0.8330 | 0.7713 | 0.4010 | No Balancing |
| Baseline | RandomForest | 1 | 0.1531 | 0.0876 | 0.8347 | 0.7775 | 0.4137 | No Balancing |
| Baseline | XGBoost | 1 | 0.2034 | 0.1232 | 0.8354 | 0.7802 | 0.4176 | No Balancing |
| Basic oversampling | LogisticRegression | 3 | 0.4402 | 0.7069 | 0.6932 | 0.7689 | 0.3973 | Random Oversampling |
| Basic oversampling | RandomForest | 3 | 0.4393 | 0.7472 | 0.6744 | 0.7752 | 0.4084 | Random Oversampling |
| Basic oversampling | XGBoost | 3 | 0.4390 | 0.7491 | 0.6732 | 0.7749 | 0.4085 | Random Oversampling |
| Advanced oversampling | LogisticRegression | 4 | 0.4297 | 0.6012 | 0.7285 | 0.7565 | 0.3841 | Borderline-SMOTE |
| Advanced oversampling | RandomForest | 4 | 0.4203 | 0.5720 | 0.7341 | 0.7608 | 0.3838 | SVM-SMOTE |
| Advanced oversampling | XGBoost | 4 | 0.4186 | 0.5734 | 0.7310 | 0.7587 | 0.3816 | SVM-SMOTE |
| Cleaning & hybrid | LogisticRegression | 4 | 0.3526 | 0.3820 | 0.7872 | 0.7660 | 0.3943 | SMOTE-Tomek |
| Cleaning & hybrid | RandomForest | 4 | 0.3404 | 0.3792 | 0.7861 | 0.7680 | 0.4043 | SMOTE-Tomek |
| Cleaning & hybrid | XGBoost | 4 | 0.3606 | 0.4052 | 0.7831 | 0.7701 | 0.4062 | ENN |