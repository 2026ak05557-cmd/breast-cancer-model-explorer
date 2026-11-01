# Breast Cancer Model Explorer

## a. Problem statement
Compare five classification algorithms for classifying breast mass
samples as benign or malignant using numerical cell-nucleus measurements.
Provide an interactive Streamlit application to evaluate saved models
on uploaded test data.

## b. Dataset description
Dataset: UCI Breast Cancer Wisconsin (Diagnostic).
Source: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic

The dataset contains 569 records and 30 numerical input features.
The features describe cell-nucleus characteristics, including radius,
texture, perimeter and area.

Target column: diagnosis.
Encoding used in this project: 0 = benign, 1 = malignant.
Class distribution: 357 benign and 212 malignant samples.

A stratified 80:20 split produces 455 training records and 114 test records,
using random_state=42. The test set contains 72 benign and 42 malignant
samples. StandardScaler is fitted only on training data within the
Logistic Regression, KNN and SVM pipelines.

## c. GitHub Repository Link
https://github.com/2026ak05557-cmd/breast-cancer-model-explorer

## d. Models used

- Logistic Regression: StandardScaler and max_iter=2000.
- Decision Tree: default settings with random_state=42.
- KNN: StandardScaler and five nearest neighbors.
- Gaussian Naive Bayes: default settings.
- SVM: StandardScaler, RBF kernel, C=1.0, probability=True,
  random_state=42.

### Evaluation metrics
All models are evaluated on the same held-out test set.
Precision, recall and F1 use malignant (1) as the positive class.
AUC uses predicted malignant-class probabilities.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |


### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Accuracy 96.49%, with the highest AUC of 0.9960. It missed 3 malignant samples and incorrectly classified 1 benign sample as malignant. |
| Decision Tree | Accuracy 92.98%, the lowest among these five models on this split. It produced 4 false negatives and 4 false positives. |
| KNN | Accuracy 95.61%. It produced 4 false negatives and 1 false positive. |
| Naive Bayes | Precision 1.0000, with no false positives, but recall 0.8333. It missed 7 malignant samples. |
| Support Vector Machine | Highest accuracy of 97.37%, F1 of 0.9630 and MCC of 0.9442. It missed 3 malignant samples and produced no false positives. |
| Overall winner for this test split | SVM, based on accuracy, F1 and MCC. Logistic Regression achieved the highest AUC. |

These observations describe one held-out split and do not establish
performance on other populations. No hyperparameter search was performed.

## Running the application
Use Python 3.12 and install the dependencies:

    pip install -r requirements.txt

Launch the app:

    streamlit run app.py

Upload test_data.csv and select a model from the sidebar.
The CSV must contain all 30 original feature columns and diagnosis.
The app loads saved models without retraining and displays six metrics,
a confusion matrix, a classification report and a comparison of all models.
AUC is shown as N/A when the uploaded data contains only one class.

## Project files
- app.py: Streamlit application.
- requirements.txt: dependency versions.
- README.md: project description and findings.
- test_data.csv: held-out test records with target labels.
- model/: five saved models and metadata.json.
- model_comparison.csv: evaluation results.
- Training notebook (.ipynb): source code and execution outputs.

## Live Streamlit App Link
https://breast-cancer-model-explorer-88tavblgx4a56fxscxzjvq.streamlit.app

## Execution environment
Training and local application execution were performed on BITS Virtual Lab.
