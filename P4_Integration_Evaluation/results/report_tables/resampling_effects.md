| Display | Family | Rows_added | Rows_removed | Net_change | Final_train_size | Train_source |
|---|---|---|---|---|---|---|
| No Balancing | Baseline | 0 | 0 | 0 | 1837734 | full corrected train split (1,837,734 rows) |
| Random Oversampling | Basic oversampling | 1210810 | 0 | 1210810 | 3048544 | full corrected train split (1,837,734 rows) |
| SMOTE | Basic oversampling | 1210810 | 0 | 1210810 | 3048544 | full corrected train split (1,837,734 rows) |
| ADASYN | Basic oversampling | 1201468 | 0 | 1201468 | 3039202 | full corrected train split (1,837,734 rows) |
| Borderline-SMOTE | Advanced oversampling | 1210810 | 0 | 1210810 | 3048544 | full corrected train split (1,837,734 rows) |
| KMeans-SMOTE | Advanced oversampling | 1210828 | 0 | 1210828 | 3048562 | full corrected train split (1,837,734 rows) |
| SVM-SMOTE † | Advanced oversampling | 32944 | 0 | 32944 | 82944 | shared stratified 50k subsample (41,472/8,528, seed 42) |
| SMOTE-N † | Advanced oversampling | 32944 | 0 | 32944 | 82944 | shared stratified 50k subsample (41,472/8,528, seed 42) |
| Tomek Links | Cleaning & hybrid | 0 | 7161 | -7161 | 1830573 | full corrected train split (1,837,734 rows); cleaning removes majority/overlap rows |
| ENN | Cleaning & hybrid | 0 | 470885 | -470885 | 1366849 | full corrected train split (1,837,734 rows); cleaning removes majority/overlap rows |
| SMOTE-Tomek | Cleaning & hybrid | 1208265 | 0 | 1208265 | 3045999 | full corrected train split (1,837,734 rows); cleaning removes majority/overlap rows |
| SMOTE-ENN | Cleaning & hybrid | 112454 | 0 | 112454 | 1950188 | full corrected train split (1,837,734 rows); cleaning removes majority/overlap rows |