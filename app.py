import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# PAGE TITLE
# =========================================================

st.title("🌱 Dry Bean Classification")

st.write(
    "Upload the test CSV file and select a machine learning model "
    "to predict the type of dry bean."
)


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = "model"


# =========================================================
# LOAD TRAINED MODELS
# =========================================================

logistic_model = joblib.load(
    os.path.join(MODEL_PATH, "logistic_regression.joblib")
)

decision_tree_model = joblib.load(
    os.path.join(MODEL_PATH, "decision_tree.joblib")
)

knn_model = joblib.load(
    os.path.join(MODEL_PATH, "knn.joblib")
)

nb_model = joblib.load(
    os.path.join(MODEL_PATH, "naive_bayes.joblib")
)

random_forest_model = joblib.load(
    os.path.join(MODEL_PATH, "random_forest.joblib")
)


# =========================================================
# LOAD SCALER AND LABEL ENCODER
# =========================================================

scaler = joblib.load(
    os.path.join(MODEL_PATH, "scaler.joblib")
)

label_encoder = joblib.load(
    os.path.join(MODEL_PATH, "label_encoder.joblib")
)


# =========================================================
# MODEL SELECTION
# =========================================================

model_name = st.selectbox(
    "Select a Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest"
    ]
)


# =========================================================
# CSV UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file is not None:

    # Read uploaded CSV
    data = pd.read_csv(uploaded_file)

    # Display uploaded data
    st.subheader("Uploaded Data")

    st.dataframe(
        data.head(),
        use_container_width=True
    )


    # =====================================================
    # RUN PREDICTION BUTTON
    # =====================================================

    if st.button("🚀 Run Prediction"):

        # -------------------------------------------------
        # Separate features and target
        # -------------------------------------------------

        X_uploaded = data.drop("Class", axis=1)

        y_actual = label_encoder.transform(
            data["Class"]
        )


        # -------------------------------------------------
        # Select trained model
        # -------------------------------------------------

        if model_name == "Logistic Regression":

            selected_model = logistic_model

        elif model_name == "Decision Tree":

            selected_model = decision_tree_model

        elif model_name == "kNN":

            selected_model = knn_model

        elif model_name == "Naive Bayes":

            selected_model = nb_model

        else:

            selected_model = random_forest_model


        # -------------------------------------------------
        # Scaling
        # -------------------------------------------------

        if model_name in [
            "Logistic Regression",
            "kNN"
        ]:

            X_input = scaler.transform(
                X_uploaded
            )

        else:

            X_input = X_uploaded


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        y_pred = selected_model.predict(
            X_input
        )


        # Convert encoded predictions
        # back to original class names

        predicted_classes = label_encoder.inverse_transform(
            y_pred
        )


        # -------------------------------------------------
        # Prediction probabilities
        # -------------------------------------------------

        y_prob = selected_model.predict_proba(
            X_input
        )


        # =================================================
        # EVALUATION METRICS
        # =================================================

        accuracy = accuracy_score(
            y_actual,
            y_pred
        )


        auc = roc_auc_score(
            y_actual,
            y_prob,
            multi_class="ovr",
            average="weighted"
        )


        precision = precision_score(
            y_actual,
            y_pred,
            average="weighted"
        )


        recall = recall_score(
            y_actual,
            y_pred,
            average="weighted"
        )


        f1 = f1_score(
            y_actual,
            y_pred,
            average="weighted"
        )


        mcc = matthews_corrcoef(
            y_actual,
            y_pred
        )


        # =================================================
        # DISPLAY METRICS
        # =================================================

        st.subheader(
            "Model Evaluation Metrics"
        )


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )


        col2.metric(
            "AUC",
            f"{auc:.4f}"
        )


        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )


        col4, col5, col6 = st.columns(3)


        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )


        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )


        col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )


        # =================================================
        # CONFUSION MATRIX
        # =================================================

        cm = confusion_matrix(
            y_actual,
            y_pred
        )


        st.subheader(
            "Confusion Matrix"
        )


        fig, ax = plt.subplots(
            figsize=(10, 7)
        )


        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax
        )


        ax.set_xlabel(
            "Predicted Class"
        )


        ax.set_ylabel(
            "Actual Class"
        )


        ax.set_title(
            f"{model_name} - Confusion Matrix"
        )


        st.pyplot(fig)


        plt.close(fig)


        # =================================================
        # CLASSIFICATION REPORT
        # =================================================

        st.subheader(
            "Classification Report"
        )


        report = classification_report(
            y_actual,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )


        report_df = pd.DataFrame(
            report
        ).transpose()


        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )


        # =================================================
        # PREDICTION RESULTS
        # =================================================

        st.subheader(
            "Prediction Results"
        )


        results = X_uploaded.copy()


        results["Actual Class"] = (
            data["Class"].values
        )


        results["Predicted Class"] = (
            predicted_classes
        )


        st.dataframe(
            results.head(20),
            use_container_width=True
        )