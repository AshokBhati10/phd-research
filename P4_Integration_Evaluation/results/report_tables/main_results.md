| Display | Person | Model | Accuracy | Precision | Recall | F1 | ROC_AUC | PR_AUC |
|---|---|---|---|---|---|---|---|---|
| No Balancing | P1 | LogisticRegression | 0.8330 | 0.5416 | 0.1354 | 0.2166 | 0.7713 | 0.4010 |
| No Balancing | P1 | RandomForest | 0.8347 | 0.6076 | 0.0876 | 0.1531 | 0.7775 | 0.4137 |
| No Balancing | P1 | XGBoost | 0.8354 | 0.5832 | 0.1232 | 0.2034 | 0.7802 | 0.4176 |
| Random Oversampling | P1 | LogisticRegression | 0.6990 | 0.3239 | 0.7033 | 0.4435 | 0.7715 | 0.3995 |
| Random Oversampling | P1 | RandomForest | 0.6847 | 0.3176 | 0.7385 | 0.4441 | 0.7777 | 0.4137 |
| Random Oversampling | P1 | XGBoost | 0.6829 | 0.3173 | 0.7463 | 0.4453 | 0.7802 | 0.4176 |
| SMOTE | P1 | LogisticRegression | 0.6995 | 0.3230 | 0.6951 | 0.4410 | 0.7685 | 0.3974 |
| SMOTE | P1 | RandomForest | 0.6843 | 0.3162 | 0.7315 | 0.4415 | 0.7751 | 0.4084 |
| SMOTE | P1 | XGBoost | 0.6840 | 0.3160 | 0.7321 | 0.4415 | 0.7742 | 0.4079 |
| ADASYN | P1 | LogisticRegression | 0.6812 | 0.3122 | 0.7224 | 0.4359 | 0.7669 | 0.3952 |
| ADASYN | P1 | RandomForest | 0.6541 | 0.3001 | 0.7716 | 0.4321 | 0.7727 | 0.4032 |
| ADASYN | P1 | XGBoost | 0.6528 | 0.2988 | 0.7689 | 0.4303 | 0.7703 | 0.3998 |
| Borderline-SMOTE | P2 | LogisticRegression | 0.6933 | 0.3195 | 0.7062 | 0.4399 | 0.7686 | 0.3962 |
| Borderline-SMOTE | P2 | RandomForest | 0.6640 | 0.3055 | 0.7618 | 0.4361 | 0.7740 | 0.4023 |
| Borderline-SMOTE | P2 | XGBoost | 0.6655 | 0.3063 | 0.7601 | 0.4367 | 0.7734 | 0.3991 |
| KMeans-SMOTE | P2 | LogisticRegression | 0.7757 | 0.3753 | 0.4740 | 0.4189 | 0.7569 | 0.3773 |
| KMeans-SMOTE | P2 | RandomForest | 0.7986 | 0.4039 | 0.3803 | 0.3918 | 0.7635 | 0.3752 |
| KMeans-SMOTE | P2 | XGBoost | 0.7995 | 0.4080 | 0.3896 | 0.3986 | 0.7701 | 0.3809 |
| SVM-SMOTE † | P2 | LogisticRegression | 0.7396 | 0.3473 | 0.5992 | 0.4398 | 0.7631 | 0.3914 |
| SVM-SMOTE † | P2 | RandomForest | 0.7512 | 0.3573 | 0.5741 | 0.4405 | 0.7676 | 0.3980 |
| SVM-SMOTE † | P2 | XGBoost | 0.7446 | 0.3509 | 0.5857 | 0.4389 | 0.7646 | 0.3965 |
| SMOTE-N † | P2 | LogisticRegression | 0.7054 | 0.3162 | 0.6255 | 0.4201 | 0.7374 | 0.3716 |
| SMOTE-N † | P2 | RandomForest | 0.7226 | 0.3231 | 0.5720 | 0.4130 | 0.7383 | 0.3597 |
| SMOTE-N † | P2 | XGBoost | 0.7146 | 0.3119 | 0.5583 | 0.4002 | 0.7265 | 0.3496 |
| Tomek Links | P3 | LogisticRegression | 0.8329 | 0.5385 | 0.1416 | 0.2242 | 0.7713 | 0.4011 |
| Tomek Links | P3 | RandomForest | 0.8349 | 0.6023 | 0.0946 | 0.1636 | 0.7776 | 0.4139 |
| Tomek Links | P3 | XGBoost | 0.8354 | 0.5769 | 0.1314 | 0.2140 | 0.7803 | 0.4176 |
| ENN | P3 | LogisticRegression | 0.7948 | 0.4075 | 0.4467 | 0.4262 | 0.7715 | 0.4006 |
| ENN | P3 | RandomForest | 0.8011 | 0.4207 | 0.4401 | 0.4301 | 0.7772 | 0.4127 |
| ENN | P3 | XGBoost | 0.7935 | 0.4100 | 0.4794 | 0.4420 | 0.7799 | 0.4168 |
| SMOTE-Tomek | P3 | LogisticRegression | 0.6998 | 0.3232 | 0.6943 | 0.4410 | 0.7684 | 0.3973 |
| SMOTE-Tomek | P3 | RandomForest | 0.6843 | 0.3162 | 0.7319 | 0.4416 | 0.7750 | 0.4083 |
| SMOTE-Tomek | P3 | XGBoost | 0.6848 | 0.3163 | 0.7301 | 0.4414 | 0.7740 | 0.4076 |
| SMOTE-ENN | P3 | LogisticRegression | 0.8213 | 0.4556 | 0.2455 | 0.3191 | 0.7526 | 0.3781 |
| SMOTE-ENN | P3 | RandomForest | 0.8239 | 0.4697 | 0.2501 | 0.3264 | 0.7425 | 0.3823 |
| SMOTE-ENN | P3 | XGBoost | 0.8188 | 0.4500 | 0.2797 | 0.3449 | 0.7461 | 0.3829 |