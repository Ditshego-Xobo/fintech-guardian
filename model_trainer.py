# model_trainer.py

import os
import pickle

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import recall_score


ETHICAL_WARNING = "⚠️ NEVER use for automatic loan denial"


class ModelTrainer:
    """Train churn model with fairness constraints and ethical safeguards."""

    def __init__(self, X_train, y_train, region_col="region"):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.region_col = region_col

        self.pipeline = None
        self.bias_audit_results = {}
        self.shap_results = {}
        self.recall = 0.0
        self.max_disparity = 0.0
        self.decision_threshold = 0.50
        self.calibrated_probabilities = None
        self.feature_metadata = {}

    def build_fair_pipeline(self):
        """Create preprocessing and calibrated Random Forest pipeline."""

        if self.X_train.empty:
            raise ValueError("Training data is empty.")

        numeric_features = self.X_train.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_features = self.X_train.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore"
            ))
        ])

        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ])

        classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=5,
            n_jobs=-1
        )

        # Sigmoid calibration implements Platt-style probability calibration.
        calibrated_classifier = CalibratedClassifierCV(
            estimator=classifier,
            method="sigmoid",
            cv=3
        )

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", calibrated_classifier)
        ])

        return self.pipeline

    def audit_bias(self):
        """Check regional recall disparity using cross-validation."""

        if self.pipeline is None:
            raise ValueError(
                "Pipeline must be built before auditing bias."
            )

        if self.region_col not in self.X_train.columns:
            raise ValueError(
                f"Region column '{self.region_col}' is missing."
            )

        # Generate out-of-fold probabilities using training data only.
        probabilities = cross_val_predict(
            self.pipeline,
            self.X_train,
            self.y_train,
            cv=3,
            method="predict_proba"
        )[:, 1]

        # Reset indexes so positional indexing is consistent.
        audit_data = pd.DataFrame({
            "region": self.X_train[self.region_col].reset_index(drop=True),
            "actual": self.y_train.reset_index(drop=True)
        })

        probabilities = pd.Series(
            probabilities
        ).reset_index(drop=True)

        # Test several probability thresholds.
        candidate_thresholds = np.arange(
            0.10,
            0.51,
            0.01
        )

        best_threshold = None
        best_disparity = np.inf
        best_recall = 0.0
        best_predictions = None

        for threshold in candidate_thresholds:

            candidate_predictions = (
                probabilities >= threshold
            ).astype(int)

            candidate_recalls = []

            for region, group in audit_data.groupby("region"):

                region_indices = group.index

                actual = audit_data.loc[
                    region_indices,
                    "actual"
                ]

                predicted = candidate_predictions.iloc[
                    region_indices
                ]

                # Recall is meaningful only when the group has
                # at least one positive churn case.
                if actual.sum() > 0:
                    region_recall = recall_score(
                        actual,
                        predicted,
                        zero_division=0
                    )

                    candidate_recalls.append(
                        region_recall
                    )

            if not candidate_recalls:
                continue

            disparity = (
                max(candidate_recalls)
                - min(candidate_recalls)
            )

            overall_recall = recall_score(
                audit_data["actual"],
                candidate_predictions,
                zero_division=0
            )

            # Select the threshold with the highest recall
            # while keeping disparity at or below 15%.
            if disparity <= 0.15:

                if (
                    best_threshold is None
                    or overall_recall > best_recall
                ):
                    best_threshold = threshold
                    best_disparity = disparity
                    best_recall = overall_recall
                    best_predictions = candidate_predictions.copy()

        if best_threshold is None:
            raise ValueError(
                "No decision threshold produced regional recall "
                "disparity at or below 15%."
            )

        self.decision_threshold = best_threshold

        predictions = best_predictions

        # Calculate final regional recall.
        audit_data["predicted"] = predictions

        regional_recalls = {}

        for region, group in audit_data.groupby("region"):

            if group["actual"].sum() == 0:
                continue

            regional_recalls[str(region)] = recall_score(
                group["actual"],
                group["predicted"],
                zero_division=0
            )

        if not regional_recalls:
            raise ValueError(
                "Unable to calculate recall for any region."
            )

        recall_values = list(
            regional_recalls.values()
        )

        max_recall = max(recall_values)
        min_recall = min(recall_values)

        self.max_disparity = (
            max_recall - min_recall
        )

        self.recall = recall_score(
            audit_data["actual"],
            predictions,
            zero_division=0
        )

        self.bias_audit_results = {
            "regional_recalls": regional_recalls,
            "max_recall": max_recall,
            "min_recall": min_recall,
            "max_disparity": self.max_disparity,
            "decision_threshold": self.decision_threshold,
            "fairness_pass": self.max_disparity <= 0.15
        }

        if not self.bias_audit_results["fairness_pass"]:
            raise ValueError(
                f"Regional recall disparity is "
                f"{self.max_disparity:.1%}, "
                f"exceeding the 15% limit."
            )

        return self.bias_audit_results

    def calibrate_probabilities(self):
        """Verify that the trained pipeline provides calibrated probabilities."""

        if self.pipeline is None:
            raise ValueError(
                "Pipeline must be trained before calibration."
            )

        if not hasattr(self.pipeline, "predict_proba"):
            raise ValueError(
                "The pipeline does not support probability predictions."
            )

        probabilities = self.pipeline.predict_proba(
            self.X_train
        )

        if probabilities.shape[1] != 2:
            raise ValueError(
                "Expected binary churn probabilities."
            )

        self.calibrated_probabilities = probabilities[:, 1]

        self.feature_metadata["calibration"] = (
            "Sigmoid (Platt-style) probability calibration is used "
            "to produce more reliable churn risk probabilities."
        )

        return self.calibrated_probabilities

    def generate_shap_values(self):
        """Generate top three SHAP feature explanations."""

        if self.pipeline is None:
            raise ValueError(
                "Pipeline must be trained before generating explanations."
            )

        try:
            import shap

            preprocessor = (
                self.pipeline.named_steps["preprocessor"]
            )

            calibrated_model = (
                self.pipeline.named_steps["classifier"]
            )

            # Get the Random Forest from the first calibrated model.
            random_forest = (
                calibrated_model
                .calibrated_classifiers_[0]
                .estimator
            )

            # Transform the training data using the fitted preprocessor.
            transformed_data = preprocessor.transform(
                self.X_train
            )

            # Convert sparse/object output to a numeric NumPy array.
            if hasattr(transformed_data, "toarray"):
                transformed_data = transformed_data.toarray()

            transformed_data = np.asarray(
                transformed_data,
                dtype=float
            )

            feature_names = (
                preprocessor.get_feature_names_out()
            )

            # Use a smaller sample for SHAP to keep execution efficient.
            sample_size = min(
                1000,
                transformed_data.shape[0]
            )

            shap_sample = transformed_data[
                :sample_size
            ]

            explainer = shap.TreeExplainer(
                random_forest
            )

            shap_values = explainer.shap_values(
                shap_sample,
                check_additivity=False
            )

            # Handle different SHAP output formats.
            if isinstance(shap_values, list):

                values = np.abs(
                    shap_values[-1]
                ).mean(axis=0)

            elif getattr(
                shap_values,
                "ndim",
                0
            ) == 3:

                values = np.abs(
                    shap_values[:, :, -1]
                ).mean(axis=0)

            else:

                values = np.abs(
                    shap_values
                ).mean(axis=0)

            importance = pd.Series(
                values,
                index=feature_names
            ).sort_values(
                ascending=False
            )

            top_features = importance.head(3)

            self.shap_results = {}

            for feature, value in top_features.items():

                self.shap_results[feature] = {
                    "importance": float(value),
                    "business_interpretation": (
                        f"{feature} contributes to the model's "
                        "churn-risk prediction and should be "
                        "reviewed alongside customer context."
                    )
                }

        except ImportError:

            self.shap_results = {
                "SHAP_unavailable": {
                    "importance": 0.0,
                    "business_interpretation": (
                        "SHAP was not available in the execution "
                        "environment. Random Forest feature "
                        "importance can be used as a fallback."
                    )
                }
            }

        return self.shap_results

    def generate_model_card(
        self,
        output_path="reports/model_card.md"
    ):
        """Create ethical model documentation."""

        output_directory = os.path.dirname(
            output_path
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True
            )

        regional_recalls = (
            self.bias_audit_results.get(
                "regional_recalls",
                {}
            )
        )

        if regional_recalls:

            recall_lines = "\n".join(
                f"- {region}: {recall:.1%}"
                for region, recall
                in regional_recalls.items()
            )

        else:

            recall_lines = (
                "- Regional recall results unavailable."
            )

        shap_lines = "\n".join(
            f"- **{feature}**: "
            f"{details['business_interpretation']}"
            for feature, details
            in self.shap_results.items()
            if feature != "SHAP_unavailable"
        )

        if not shap_lines:

            shap_lines = (
                "- Feature explanations should be reviewed "
                "using model feature importance."
            )

        card_content = f"""# Churn Prediction Model Card

## Model Purpose

This model predicts the likelihood that a fintech customer may churn.
It is intended to support human decision-making and customer retention
strategies, not to make automatic lending decisions.

## Performance

- Overall Recall: {self.recall:.1%}
- Maximum Regional Recall Disparity: {self.max_disparity:.1%}
- Decision Threshold: {self.decision_threshold:.2f}

### Regional Recall

{recall_lines}

## Interpretability

The model uses feature-level explanations to help identify important
drivers of predicted churn.

### Top Feature Insights

{shap_lines}

## Critical Limitations

- Predictions may be less reliable for customers or regions that are
  poorly represented in the training data.
- Economic conditions can change over time, reducing model performance.
- Correlation does not establish causation.
- Predictions should not be interpreted as proof that a customer will churn.
- Model performance may change as customer behaviour changes.

## Failure Modes

- Missing or inaccurate customer information may produce unreliable predictions.
- Changes in economic conditions may cause model drift.
- Regional representation differences may affect subgroup performance.
- High financial-strain predictions require additional human assessment.
- A prediction is a risk estimate, not a certainty.

## Ethical Constraints

⚠️ NEVER use for automatic loan denial

- Human review is required before any high-impact financial decision.
- Regional performance must be monitored regularly.
- The model must not be used to discriminate against customers based on
  location or socioeconomic context.
- Fairness checks must be repeated whenever the model is retrained.
- Customers should not be denied financial services solely because of
  this model's prediction.

## Monitoring Plan

- Monitor regional recall monthly.
- Alert if regional recall disparity exceeds 15%.
- Monitor model performance and calibration over time.
- Review changes in economic and customer behaviour patterns.
- Retrain the model when significant model drift is identified.

## Responsible Use

This model is a decision-support tool only. A qualified human reviewer
must consider the customer's circumstances before taking any high-impact
action.
"""

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(card_content)

        return output_path

    def train(self):
        """Train the model with full ethical validation."""

        self.build_fair_pipeline()

        self.pipeline.fit(
            self.X_train,
            self.y_train
        )

        self.audit_bias()

        self.calibrate_probabilities()

        self.generate_shap_values()

        return self.pipeline

    def __str__(self):
        """Return a summary of model performance and fairness."""

        return (
            f"Recall: {self.recall:.1%} | "
            f"Max disparity: {self.max_disparity:.1%} | "
            f"Threshold: {self.decision_threshold:.2f}"
        )


def main():
    """Load engineered data, train the model and save outputs."""

    input_path = (
        "data/processed/engineered_features.csv"
    )

    model_path = (
        "models/churn_pipeline.pkl"
    )

    model_card_path = (
        "reports/model_card.md"
    )

    try:

        # Load engineered data from Milestone 2.
        df = pd.read_csv(input_path)

        # Check that the target exists.
        if "churned" not in df.columns:
            raise ValueError(
                "Required target column 'churned' is missing."
            )

        # Remove rows where the churn outcome is unknown.
        labelled_data = df.dropna(
            subset=["churned"]
        ).copy()

        # Separate predictors and target.
        X = labelled_data.drop(
            columns=["churned"]
        )

        y = labelled_data["churned"]

        # Keep a separate test set to avoid using the entire
        # dataset for model development.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # Train using the training portion only.
        trainer = ModelTrainer(
            X_train,
            y_train,
            region_col="region"
        )

        trainer.train()

        # Create output directories.
        os.makedirs(
            "models",
            exist_ok=True
        )

        os.makedirs(
            "reports",
            exist_ok=True
        )

        # Save the complete trained pipeline.
        with open(
            model_path,
            "wb"
        ) as file:

            pickle.dump(
                trainer.pipeline,
                file
            )

        # Generate model card.
        trainer.generate_model_card(
            model_card_path
        )

        # Display results.
        print(trainer)

        print(
            f"\nSaved model to {model_path}"
        )

        print(
            f"Saved model card to {model_card_path}"
        )

        print(
            "\nEthical warning:"
        )

        print(
            ETHICAL_WARNING
        )

    except Exception as e:

        raise ValueError(
            f"Model training failed: {str(e)}"
        ) from e


if __name__ == "__main__":
    main()