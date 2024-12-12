# Classifiers on Datasets Project

## Abstract
This project explores the application of machine learning classifiers on multiple datasets from the UCI Machine Learning Repository. The classifiers evaluated include but were not limited to Random Forest, Gradient Boosting, and Logistic Regression. Experiments are conducted on binary classification datasets from the UCI repository, such as the Car Evaluation Dataset and the Breast Cancer Dataset, using varying train-test splits. Notable insights include the effectiveness of ensemble methods on complex datasets and the challenges faced by linear models with categorical features. Performance metrics, including cross-validation accuracy and test set accuracy, are analyzed and compared across classifiers and datasets.

## Introduction
Machine learning classifiers play a crucial role in understanding patterns and making predictions based on data. It is arguably the most revolutionary method to making predictions on data and understanding patterns across said data. This project aims to assess the performance of commonly used classifiers such as Random Forest, Gradient Boosting, and SVM (other classifiers are utilized but were not listed). Such classifiers are widely used due to their versatility and robustness in handling a variety of data types. The datasets used are selected for their diversity in structure and features and relatively high number of data entries, requiring different preprocessing and modeling considerations. Previous studies have shown that ensemble methods like Random Forest and Gradient Boosting often outperform simpler models like Logistic Regression on complex datasets. By conducting experiments on three datasets with binary classification tasks, we aim to identify the strengths and weaknesses of each classifier and their suitability for various data types.

## Method
### Datasets
1. **Car Evaluation Dataset**:
    - Source: UCI Machine Learning Repository
    - Task: Binary classification (positive: `good`, `vgood`; negative: `unacc`, `acc`)
    - Features: Categorical, transformed using Label Encoding

2. **Breast Cancer Wisconsin Dataset**:
    - Source: UCI Machine Learning Repository
    - Task: Binary classification (positive: malignant, negative: benign)
    - Features: Numerical, scaled using StandardScaler

3. **Wine Quality Dataset**:
    - Source: UCI Machine Learning Repository
    - Task: Binary classification (positve: malignment, negative: benign)
    - Features: Continuous

### Classifiers

**Car Evaluation**
1. Decision Tree:
    - A simple, interpretable model that splits data based on feature values to form a tree structure. It is prone to overfitting on small datasets without proper tuning. Key hyperparameter: `max_depth` was tuned to prevent overfitting.


2. Random Forest
    - An ensemble method that builds multiple decision trees and averages their results to improve robustness. It handles overfitting better than a single decision tree. Key hyperparameters include the number of trees (`n_estimators=100`) and `max_features` for each split.

3. SVM
    - Support Vector Machines (SVMs) find the hyperplane that best separates classes in the feature space. SVMs are effective for high-dimensional data and use kernels to handle non-linear relationships. For this project, a radial basis function (RBF) kernel was used, with parameters `C=1` and `gamma=0.1`.

**Breast Cancer Wisconsin**:
1. Random Forest:
    - *Read from Car Evaluation classifiers overview*

2. Gradient Boosting:
    - A boosting algorithm that sequentially minimizes errors by focusing on difficult samples. The learning rate was set to 0.1, and the number of boosting stages (`n_estimators`) was tuned to 100.

3. Logistic Regression:
    - A linear model for binary classification that predicts probabilities using the sigmoid function. The `l2` regularization was used with a solver (`saga`) appropriate for large datasets.

**Wine Quality**
1. SVM
    - Support Vector Machines (SVMs) find the hyperplane that best separates classes in the feature space. SVMs are effective for high-dimensional data and use kernels to handle non-linear relationships. For this project, data was balanced using SVC and SelectKBest was used to pick the top 10 most relevant features

2. Random Forest
    - *Read from Car Evaluation classifiers overview*

3. Logistic Regression
    - *Read from Breast Cancer classifiers overview*

### Experimental Setup
- Train-test splits:
  - 20% train / 80% test
  - 50% train / 50% test
  - 80% train / 20% test
- Multiple trials (3 runs with different random seeds) to average results and ensure robustness. Random seeds were set using a fixed value (random_state=42) to ensure reproducibility across runs.

- Metrics:
    - Cross-validation accuracy (mean and standard deviation)
    - Test set accuracy

**Wine Quality** (Exception):

One train-test split was initially defined

- Data Balancing
    - Data was balanced and then relevant features were selected as the new feature set (`X_new`). After, new train-test splits were defined.
- Hyperparameter tuning:
    - Hyperparamter tuning using GridSearch was performed across all three classifiers to optimize n_estimators, max_depth, and learning_rate. Logistic Regression was tuned for its regularization strength (C).
- Metrics
    - Preicision and recall

## Experiment

### Car Evaluation Dataset
1. **Preprocessing**:
   - As stated before, the features (`X`) encoded using Label Encoding.
   - Target (`y`) converted to binary classes.

2. **Results**:
   - Decision Tree: Performed with similar precision to RandomForest, but not as effective by a slight margin.
   - Random Forest: Achieved the highest test accuracy across all splits
   - SVM: Struggled with categorical features, requiring careful preprocessing and scaling.

### Breast Cancer Dataset
1. **Preprocessing**:
   - As stated before, the feature set (`X`) was scaled using StandardScaler to improve convergence in Logistic Regression.
   - Target values (`y`) encoded as binary classes.

2. **Results**:
   - Gradient Boosting: Performed exceptionally well, particularly on imbalanced splits.
   - Logistic Regression: Required increased iterations and solver adjustments (e.g., `saga`) for convergence.
   - Random Forest: Delivered consistently high accuracy across all splits.

### Wine Quality Dataset
1. **Preprocessing**
    - StandardScaler() was applied to the feature set (`X`), no other encoding was done after
    - Target value (`y`) was already set as floats, so no encoding was necessary and was left as is

2. **Results**
    - SVM: Did not perform the best for the split. Every precicison marker was in between 0.3 and 0.65
    - Random Forest: Classifier that performed the best across the split.
    - Logistic Regression: Performed the worst out of the three. Some precision markers were under 0.1. Most likely due to lack of encoding, even though encoding was not necessary

## Conclusion
The project demonstrates the importance of selecting the appropriate classifier and preprocessing steps based on the dataset characteristics. Random Forest and Gradient Boosting emerged as robust choices for both datasets, while Logistic Regression required additional adjustments to perform effectively. Overall however, it was consistent across the three datasets that the Random Forest classifier performed the best across all the utilized classifiers. Future work could explore additional datasets and classifiers. While hyperparameter tuning was utilized for the Wine Quality dataset, hyperparamter tuning would have been useful for the other datasets for further optimization.

## References
1. UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/](https://archive.ics.uci.edu/ml/)
2. Scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
3. Caruana, R., & Niculescu-Mizil, A. (2006). An Empirical Comparison of Supervised Learning Algorithms. Proceedings of ICML 2006.
