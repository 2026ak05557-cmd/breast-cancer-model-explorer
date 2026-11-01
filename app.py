from pathlib import Path
import json
import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Model Explorer",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "model" / "metadata.json") as file:
    metadata = json.load(file)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

st.title("Breast Cancer Model Explorer")
st.write(
    "Compare five classification models using "
    "cell-nucleus measurements from the Wisconsin dataset."
)
st.caption("Machine Learning Assignment 2 | Educational demonstration")

selected_model = st.sidebar.selectbox(
    "Choose a classification model",
    list(MODEL_FILES)
)

st.sidebar.info("Target labels: 0 = benign, 1 = malignant")

st.subheader("Upload your test data")
st.write(
    "Upload a CSV containing the 30 input features and "
    "the diagnosis column."
)

uploaded_file = st.file_uploader(
    "Choose a test CSV",
    type=["csv"]
)

sample_path = BASE_DIR / "test_data.csv"
if sample_path.exists():
    st.download_button(
        "Download sample test CSV",
        data=sample_path.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv"
    )

if uploaded_file is None:
    st.info("Upload test data to begin evaluating a model.")
else:
    st.success("CSV uploaded. Model evaluation will be added next.")

# Continue only after a CSV is uploaded
if uploaded_file is not None:
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score,
        recall_score, f1_score, matthews_corrcoef,
        confusion_matrix, classification_report
    )

    try:
        test_df = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read the CSV: {error}")
        st.stop()

    features = metadata["feature_names"]
    target = metadata["target_column"]
    required_columns = features + [target]

    missing = [
        column for column in required_columns
        if column not in test_df.columns
    ]

    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    if test_df.empty:
        st.error("The CSV has no data rows.")
        st.stop()

    try:
        inputs = test_df[features].apply(
            pd.to_numeric, errors="raise"
        )
        actual = pd.to_numeric(test_df[target], errors="raise")
    except (ValueError, TypeError):
        st.error("All features and diagnosis must be numeric.")
        st.stop()

    if not np.isfinite(inputs.to_numpy(dtype=float)).all():
        st.error("Features must not contain missing or infinite values.")
        st.stop()

    if not actual.isin([0, 1]).all():
        st.error("Diagnosis must contain only 0 (benign) or 1 (malignant).")
        st.stop()

    actual = actual.astype(int)

    st.write(f"Test records: **{len(test_df)}**")
    with st.expander("Preview uploaded test data"):
        st.dataframe(test_df.head())

    @st.cache_resource
    def load_saved_model(filename):
        return joblib.load(BASE_DIR / "model" / filename)

    def evaluate_model(model):
        predictions = model.predict(inputs)
        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(inputs)[:, positive_index]

        scores = {
            "Accuracy": accuracy_score(actual, predictions),
            "AUC": (
                roc_auc_score(actual, probabilities)
                if actual.nunique() == 2 else np.nan
            ),
            "Precision": precision_score(
                actual, predictions, zero_division=0
            ),
            "Recall": recall_score(
                actual, predictions, zero_division=0
            ),
            "F1": f1_score(actual, predictions, zero_division=0),
            "MCC": matthews_corrcoef(actual, predictions)
        }
        return scores, predictions

    model = load_saved_model(MODEL_FILES[selected_model])
    scores, predictions = evaluate_model(model)

    st.subheader(f"Results: {selected_model}")
    metric_columns = st.columns(3)

    for index, (name, value) in enumerate(scores.items()):
        metric_columns[index % 3].metric(
            name,
            "N/A" if pd.isna(value) else f"{value:.4f}"
        )

    if actual.nunique() < 2:
        st.info("AUC requires both classes in the uploaded test data.")

    st.caption("Precision, recall and F1 use malignant (1) as positive.")

    st.subheader("Confusion matrix")
    matrix = confusion_matrix(actual, predictions, labels=[0, 1])
    st.dataframe(pd.DataFrame(
        matrix,
        index=["Actual benign", "Actual malignant"],
        columns=["Predicted benign", "Predicted malignant"]
    ))

    st.subheader("Classification report")
    report = classification_report(
        actual, predictions,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4))

    st.subheader("Compare all five models")
    comparison_rows = []

    for name, filename in MODEL_FILES.items():
        saved_model = load_saved_model(filename)
        model_scores, _ = evaluate_model(saved_model)
        comparison_rows.append({"Model": name, **model_scores})

    st.dataframe(pd.DataFrame(comparison_rows).round(4))
