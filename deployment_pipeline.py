# deployment_pipeline.py

import os
import hashlib
import logging
import joblib

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


ETHICAL_WARNING = (
    "Do NOT target high-strain customers with predatory offers"
)


class DeploymentPipeline:
    """Production inference pipeline with validation and safeguards."""

    REQUIRED_COLUMNS = [
        "income",
        "debt",
        "region",
        "load_shedding_hours",
        "financial_strain"
    ]

    ACTUAL_STRAIN_COLUMN = "financial_strain_ratio"

    MODEL_VERSION = "1.2"
    TRAINING_DATE = "2024-03-09"

    ETHICAL_CONSTRAINTS = [
        "Human review required for debt_to_income > 0.6",
        "Never deny service based solely on model output",
        "Monthly fairness audits mandatory"
    ]

    def __init__(self, model_path):
        self.model_path = model_path

        self.pipeline = None
        self.metadata = {}

        self.error_count = 0
        self.total_predictions = 0
        self.consecutive_errors = 0

        self.circuit_breaker_threshold = 0.05

    # ---------------------------------------------------------
    # HASH FUNCTIONS
    # ---------------------------------------------------------

    def _calculate_file_hash(self, path):
        """Calculate SHA-256 hash for a file."""

        sha256 = hashlib.sha256()

        with open(path, "rb") as file:
            for chunk in iter(
                lambda: file.read(8192),
                b""
            ):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _calculate_data_hash(self):
        """Calculate SHA-256 hash for engineered training data."""

        data_path = "data/processed/engineered_features.csv"

        if not os.path.exists(data_path):
            return "N/A"

        data = pd.read_csv(data_path)

        data_string = data.to_csv(
            index=False
        ).encode("utf-8")

        return hashlib.sha256(
            data_string
        ).hexdigest()

    # ---------------------------------------------------------
    # MODEL LOADING
    # ---------------------------------------------------------

    def load_model(self):
        """Load and validate the production model."""

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        artifact = joblib.load(
            self.model_path
        )

        # Milestone 5 artifact
        if isinstance(artifact, dict):

            if "pipeline" not in artifact:
                raise ValueError(
                    "Model artifact does not contain a 'pipeline' key."
                )

            self.pipeline = artifact["pipeline"]

            self.metadata = artifact.get(
                "metadata",
                {}
            ).copy()

        # Original Milestone 3 artifact
        else:

            self.pipeline = artifact
            self.metadata = {}

        # Add default metadata if required information is missing.
        self.metadata.setdefault(
            "version",
            self.MODEL_VERSION
        )

        self.metadata.setdefault(
            "training_date",
            self.TRAINING_DATE
        )

        self.metadata.setdefault(
            "data_hash",
            self._calculate_data_hash()
        )

        self.metadata.setdefault(
            "ethical_constraints",
            self.ETHICAL_CONSTRAINTS
        )

        # Validate version.
        if self.metadata["version"] != self.MODEL_VERSION:

            raise ValueError(
                "Model version mismatch. "
                f"Expected {self.MODEL_VERSION}, "
                f"found {self.metadata['version']}."
            )

        # Validate that the loaded object can predict.
        if not hasattr(
            self.pipeline,
            "predict"
        ):

            raise ValueError(
                "Loaded artifact is not a valid prediction pipeline."
            )

        # Calculate the actual file hash.
        actual_hash = self._calculate_file_hash(
            self.model_path
        )

        self.metadata["loaded_file_hash"] = actual_hash

        logger.info(
            "Loaded pipeline v%s",
            self.metadata["version"]
        )

        logger.info(
            "Training date: %s",
            self.metadata["training_date"]
        )

        logger.info(
            "Model hash: %s",
            actual_hash[:12]
        )

        return self

    # ---------------------------------------------------------
    # MODEL FEATURE DETECTION
    # ---------------------------------------------------------

    def _get_model_columns(self):
        """
        Get columns expected by the trained
        ColumnTransformer.
        """

        if self.pipeline is None:
            return []

        try:

            # Milestone 3 uses a sklearn Pipeline.
            preprocessor = self.pipeline.named_steps[
                "preprocessor"
            ]

            columns = []

            for transformer in preprocessor.transformers_:

                transformer_name = transformer[0]
                transformer_columns = transformer[2]

                if transformer_name == "remainder":
                    continue

                if isinstance(
                    transformer_columns,
                    (list, tuple)
                ):

                    columns.extend(
                        transformer_columns
                    )

                elif isinstance(
                    transformer_columns,
                    np.ndarray
                ):

                    columns.extend(
                        transformer_columns.tolist()
                    )

            # Remove duplicates while preserving order.
            columns = list(
                dict.fromkeys(columns)
            )

            return columns

        except Exception as error:

            logger.warning(
                "Could not automatically determine "
                "model columns: %s",
                error
            )

            return []

    # ---------------------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------------------

    def _validate_input(self, X):
        """Validate incoming prediction data."""

        if not isinstance(
            X,
            pd.DataFrame
        ):

            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if X.empty:

            raise ValueError(
                "Input DataFrame is empty."
            )

        X_valid = X.copy()

        # -----------------------------------------------------
        # Financial strain compatibility
        # -----------------------------------------------------

        # If the assignment's financial_strain column is supplied,
        # calculate the actual feature used by our Milestone 2 model.
        if (
            self.ACTUAL_STRAIN_COLUMN
            not in X_valid.columns
            and "income" in X_valid.columns
            and "debt" in X_valid.columns
        ):

            income = pd.to_numeric(
                X_valid["income"],
                errors="coerce"
            )

            debt = pd.to_numeric(
                X_valid["debt"],
                errors="coerce"
            )

            safe_income = income.replace(
                0,
                np.nan
            )

            X_valid[
                self.ACTUAL_STRAIN_COLUMN
            ] = (
                debt / safe_income
            )

            X_valid[
                self.ACTUAL_STRAIN_COLUMN
            ] = (
                X_valid[
                    self.ACTUAL_STRAIN_COLUMN
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .fillna(0)
            )

        # If financial_strain_ratio was supplied,
        # also create the assignment-compatible name.
        if (
            "financial_strain"
            not in X_valid.columns
            and self.ACTUAL_STRAIN_COLUMN
            in X_valid.columns
        ):

            X_valid["financial_strain"] = X_valid[
                self.ACTUAL_STRAIN_COLUMN
            ]

        # -----------------------------------------------------
        # Required columns
        # -----------------------------------------------------

        missing_required = []

        for column in self.REQUIRED_COLUMNS:

            if column == "financial_strain":

                if (
                    "financial_strain"
                    not in X_valid.columns
                    and self.ACTUAL_STRAIN_COLUMN
                    not in X_valid.columns
                ):

                    missing_required.append(
                        column
                    )

            elif column not in X_valid.columns:

                missing_required.append(
                    column
                )

        if missing_required:

            raise ValueError(
                "Missing required input columns: "
                + ", ".join(missing_required)
            )

        # -----------------------------------------------------
        # Numeric conversion
        # -----------------------------------------------------

        X_valid["income"] = pd.to_numeric(
            X_valid["income"],
            errors="coerce"
        )

        X_valid["debt"] = pd.to_numeric(
            X_valid["debt"],
            errors="coerce"
        )

        X_valid["load_shedding_hours"] = pd.to_numeric(
            X_valid["load_shedding_hours"],
            errors="coerce"
        )

        # -----------------------------------------------------
        # High-income warning
        # -----------------------------------------------------

        high_income = (
            X_valid["income"] > 1_000_000
        )

        if high_income.any():

            logger.warning(
                "Income above R1,000,000 detected in %d record(s).",
                int(high_income.sum())
            )

        # -----------------------------------------------------
        # Unknown regions
        # -----------------------------------------------------

        known_regions = {
            "Cape Town",
            "Durban",
            "Gauteng",
            "Gqeberha",
            "Johannesburg",
            "Johburg",
            "Pretoria",
            "Western Cape"
        }

        unknown_regions = (
            ~X_valid["region"].isin(
                known_regions
            )
        )

        if unknown_regions.any():

            logger.warning(
                "Unknown region detected. "
                "Mapping unknown values to 'Other'."
            )

            X_valid.loc[
                unknown_regions,
                "region"
            ] = "Other"

        # -----------------------------------------------------
        # Invalid numeric values
        # -----------------------------------------------------

        numeric_columns = X_valid.select_dtypes(
            include=["number"]
        ).columns

        for column in numeric_columns:

            values = pd.to_numeric(
                X_valid[column],
                errors="coerce"
            )

            invalid_values = ~np.isfinite(
                values
            )

            if invalid_values.any():

                logger.warning(
                    "Invalid numeric values detected in '%s'.",
                    column
                )

                X_valid.loc[
                    invalid_values,
                    column
                ] = np.nan

        # -----------------------------------------------------
        # Model feature completion
        # -----------------------------------------------------

        model_columns = self._get_model_columns()

        for column in model_columns:

            if column not in X_valid.columns:

                logger.warning(
                    "Missing model feature '%s'. "
                    "Using safe default.",
                    column
                )

                if column in [
                    "loan_purpose",
                    "region"
                ]:

                    X_valid[column] = "Other"

                else:

                    X_valid[column] = 0.0

        # -----------------------------------------------------
        # Feature order
        # -----------------------------------------------------

        if model_columns:

            X_valid = X_valid[
                model_columns
            ]

        return X_valid

    # ---------------------------------------------------------
    # CIRCUIT BREAKER
    # ---------------------------------------------------------

    def _circuit_breaker_active(self):
        """Return True when prediction error rate exceeds 5%."""

        error_rate = (
            self.error_count
            / max(
                self.total_predictions,
                1
            )
        )

        return (
            error_rate
            > self.circuit_breaker_threshold
        )

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------

    def predict(self, X):
        """Predict churn with validation and safe defaults."""

        if isinstance(
            X,
            pd.DataFrame
        ):

            input_count = len(X)

        else:

            input_count = 1

        # Check circuit breaker before prediction.
        if self._circuit_breaker_active():

            logger.warning(
                "Circuit breaker triggered because "
                "prediction error rate exceeds 5%."
            )

            return np.zeros(
                input_count,
                dtype=int
            )

        self.total_predictions += input_count

        try:

            X_valid = self._validate_input(
                X
            )

            predictions = self.pipeline.predict(
                X_valid
            )

            # Successful prediction.
            self.consecutive_errors = 0

            return np.asarray(
                predictions
            )

        except Exception as error:

            self.error_count += input_count
            self.consecutive_errors += 1

            logger.error(
                "Prediction failed: %s",
                error
            )

            if self.consecutive_errors >= 10:

                logger.critical(
                    "More than 10 consecutive prediction "
                    "errors detected. Investigation required."
                )

            # Safe default.
            return np.zeros(
                input_count,
                dtype=int
            )

    # ---------------------------------------------------------
    # DRIFT MONITORING / PSI
    # ---------------------------------------------------------

    def monitor_drift(
        self,
        reference=None,
        current=None
    ):
        """
        Population Stability Index (PSI) monitoring hook.

        PSI interpretation:

        < 0.10       = stable
        0.10 - 0.25  = moderate drift
        > 0.25       = significant drift
        """

        if (
            reference is None
            or current is None
        ):

            return {
                "status": "placeholder",
                "method": "PSI",
                "message": (
                    "Provide reference and current "
                    "feature distributions to calculate PSI."
                )
            }

        reference = np.asarray(
            reference,
            dtype=float
        )

        current = np.asarray(
            current,
            dtype=float
        )

        if (
            len(reference) == 0
            or len(current) == 0
        ):

            raise ValueError(
                "Reference and current distributions "
                "cannot be empty."
            )

        # Create percentile bins from reference data.
        breakpoints = np.percentile(
            reference,
            np.arange(
                0,
                101,
                10
            )
        )

        breakpoints = np.unique(
            breakpoints
        )

        if len(breakpoints) < 2:

            return {
                "psi": 0.0,
                "status": "stable",
                "method": "PSI"
            }

        reference_counts, _ = np.histogram(
            reference,
            bins=breakpoints
        )

        current_counts, _ = np.histogram(
            current,
            bins=breakpoints
        )

        reference_total = max(
            reference_counts.sum(),
            1
        )

        current_total = max(
            current_counts.sum(),
            1
        )

        reference_pct = (
            reference_counts
            / reference_total
        )

        current_pct = (
            current_counts
            / current_total
        )

        # Avoid zero values.
        reference_pct = np.clip(
            reference_pct,
            0.0001,
            None
        )

        current_pct = np.clip(
            current_pct,
            0.0001,
            None
        )

        psi = np.sum(
            (
                current_pct
                - reference_pct
            )
            * np.log(
                current_pct
                / reference_pct
            )
        )

        if psi < 0.10:

            status = "stable"

        elif psi <= 0.25:

            status = "moderate_drift"

        else:

            status = "significant_drift"

        return {
            "psi": float(psi),
            "status": status,
            "method": "PSI"
        }

    # ---------------------------------------------------------
    # README GENERATION
    # ---------------------------------------------------------

    def generate_readme(self, output_path="README.md"):
        """Generate the deployment README."""

        version = self.metadata.get(
            "version",
            self.MODEL_VERSION
        )

        training_date = self.metadata.get(
            "training_date",
            self.TRAINING_DATE
        )

        data_hash = self.metadata.get(
            "data_hash",
            "N/A"
        )

        model_hash = self.metadata.get(
            "loaded_file_hash",
            "N/A"
        )

        lines = [
            "#  NEXUSLEND CHURN PREDICTION PIPELINE",
            "",
            "## Deployment Overview",
            "",
            "This repository contains the production handoff package "
            "for the NexusLend churn prediction solution.",
            "",
            "The deployment pipeline validates incoming data, handles "
            "unknown categories, provides safe defaults after failures, "
            "and includes a PSI monitoring hook.",
            "",
            f"**Pipeline Version:** {version}",
            "",
            f"**Training Date:** {training_date}",
            "",
            f"**Data Hash:** {data_hash}",
            "",
            f"**Model Hash:** {model_hash}",
            "",
            f"**Ethical Constraints:** "
            f"{len(self.ETHICAL_CONSTRAINTS)}",
            "",
            "##  Quick Start",
            "",
            "### 1. Install dependencies",
            "",
            "```bash",
            "pip install pandas numpy scikit-learn joblib matplotlib seaborn",
            "```",
            "",
            "### 2. Run the deployment pipeline",
            "",
            "```bash",
            "python deployment_pipeline.py",
            "```",
            "",
            "### 3. Load the packaged inference pipeline",
            "",
            "```python",
            "from deployment_pipeline import DeploymentPipeline",
            "",
            "pipe = DeploymentPipeline(",
            '    "models/inference_pipeline.pkl"',
            ").load_model()",
            "```",
            "",
            "##  Input Schema",
            "",
            "| Column | Type | Description |",
            "|---|---|---|",
            "| income | float | Monthly income in ZAR |",
            "| debt | float | Customer debt |",
            "| region | string | Customer region |",
            "| load_shedding_hours | float | Load-shedding exposure |",
            "| financial_strain | float | Financial strain indicator |",
            "",
            "Unknown regions are mapped to `Other`.",
            "",
            "The actual project model uses the Milestone 2 engineered "
            "feature `financial_strain_ratio`.",
            "",
            "## Input Validation",
            "",
            "- Checks that input is a pandas DataFrame.",
            "- Rejects empty input.",
            "- Checks required columns.",
            "- Converts numeric fields safely.",
            "- Detects invalid numeric values.",
            "- Warns about income above R1,000,000.",
            "- Maps unknown regions to `Other`.",
            "- Handles missing engineered features.",
            "- Preserves model feature order.",
            "",
            "## Circuit Breaker",
            "",
            "The circuit breaker activates when the prediction error "
            "rate exceeds **5%**.",
            "",
            "When activated, safe default predictions are returned "
            "instead of continuing with potentially unreliable "
            "model outputs.",
            "",
            "Ten consecutive prediction failures generate a critical "
            "operational warning.",
            "",
            "##  Critical Ethical Constraints",
            "",
            "- Human review required for debt_to_income > 0.6",
            "- Never deny service based solely on model output",
            "- Monthly fairness audits mandatory",
            f"- {ETHICAL_WARNING}",
            "- Do not exploit financially vulnerable customers.",
            "- Do not use the model for automatic loan denial.",
            "",
            "## 🔍 Monitoring Commands",
            "",
            "### Run the deployment pipeline",
            "",
            "```bash",
            "python deployment_pipeline.py",
            "```",
            "",
            "### Check the inference model",
            "",
            "```bash",
            'python -c "import os; print(os.path.exists(\'models/inference_pipeline.pkl\'))"',
            "```",
            "",
            "### Future PSI drift monitoring",
            "",
            "```bash",
            "python monitor_drift.py --pipeline models/inference_pipeline.pkl",
            "```",
            "",
            "PSI interpretation:",
            "",
            "- PSI < 0.10: stable",
            "- PSI 0.10–0.25: moderate drift",
            "- PSI > 0.25: significant drift",
            "",
            "##  Failure Modes",
            "",
            "### Unknown region",
            "",
            "Unknown regions are mapped to `Other` and a warning is logged.",
            "",
            "### Missing required input",
            "",
            "Missing required columns result in a safe default prediction.",
            "",
            "### Invalid numeric values",
            "",
            "Invalid numeric values are converted to missing values and "
            "handled by the model safeguards.",
            "",
            "### High prediction error rate",
            "",
            "If the prediction error rate exceeds 5%, the circuit breaker "
            "returns safe default predictions.",
            "",
            "### Repeated failures",
            "",
            "Ten consecutive failures generate a critical warning.",
            "",
            "## Model Metadata",
            "",
            f"- Version: {version}",
            f"- Training date: {training_date}",
            f"- Data hash: {data_hash}",
            f"- Model hash: {model_hash}",
            f"- Ethical constraints: {len(self.ETHICAL_CONSTRAINTS)}",
            "",
            "## Production Handoff",
            "",
            "Before production deployment:",
            "",
            "1. Validate the model artifact.",
            "2. Confirm the model version.",
            "3. Confirm the data hash.",
            "4. Confirm the model hash.",
            "5. Test input validation.",
            "6. Test unknown-category handling.",
            "7. Test the circuit breaker.",
            "8. Run fairness checks.",
            "9. Review drift monitoring.",
            "10. Confirm human oversight.",
            "",
            "## Responsible Use",
            "",
            "This pipeline is a decision-support system.",
            "",
            "Predictions must not be treated as facts about customers.",
            "",
            "Human oversight is required for high-impact decisions.",
            "",
            "The model must never be used to exploit customers "
            "experiencing financial hardship.",
            ""
        ]

        readme_content = "\n".join(lines)

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(readme_content)

        logger.info(
            "README generated: %s",
            output_path
        )

        return output_path

    def __str__(self):
        """Return deployment pipeline summary."""

        return (
            f"Pipeline v"
            f"{self.metadata.get('version', self.MODEL_VERSION)}"
            f" | Trained: "
            f"{self.metadata.get('training_date', self.TRAINING_DATE)}"
            f" | Ethical constraints: "
            f"{len(self.ETHICAL_CONSTRAINTS)}"
        )


# =============================================================
# INFERENCE ARTIFACT CREATION
# =============================================================

def create_inference_artifact(
    source_model_path,
    output_model_path
):
    """Package the Milestone 3 model for deployment."""

    if not os.path.exists(source_model_path):
        raise FileNotFoundError(
            f"Source model not found: {source_model_path}"
        )

    # Load the existing Milestone 3 model.
    pipeline = joblib.load(source_model_path)

    # Calculate source model hash.
    source_model_hash = hashlib.sha256()

    with open(source_model_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(8192),
            b""
        ):
            source_model_hash.update(chunk)

    source_model_hash = source_model_hash.hexdigest()

    # Calculate training data hash.
    data_path = (
        "data/processed/engineered_features.csv"
    )

    if os.path.exists(data_path):

        data = pd.read_csv(data_path)

        data_hash = hashlib.sha256(
            data.to_csv(
                index=False
            ).encode("utf-8")
        ).hexdigest()

    else:

        data_hash = "N/A"

    # Create deployment metadata.
    metadata = {
        "version": DeploymentPipeline.MODEL_VERSION,
        "training_date": DeploymentPipeline.TRAINING_DATE,
        "data_hash": data_hash,
        "model_hash": source_model_hash,
        "ethical_constraints": (
            DeploymentPipeline.ETHICAL_CONSTRAINTS
        )
    }

    # Package the model and metadata together.
    artifact = {
        "pipeline": pipeline,
        "metadata": metadata
    }

    output_directory = os.path.dirname(
        output_model_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    joblib.dump(
        artifact,
        output_model_path
    )

    logger.info(
        "Inference artifact created: %s",
        output_model_path
    )

    return output_model_path


# =============================================================
# MAIN
# =============================================================

def main():
    """Create the complete production handoff package."""

    source_model = (
        "models/churn_pipeline.pkl"
    )

    inference_model = (
        "models/inference_pipeline.pkl"
    )

    print(
        "Creating production inference package..."
    )

    # Create packaged inference model.
    create_inference_artifact(
        source_model,
        inference_model
    )

    print(
        "Inference model created successfully."
    )

    # Load the packaged model.
    deployment = DeploymentPipeline(
        inference_model
    )

    deployment.load_model()

    print(
        deployment
    )

    # Generate README.
    deployment.generate_readme(
        "README.md"
    )

    print(
        "Deployment package completed."
    )

    print(
        f"Inference model: {inference_model}"
    )

    print(
        "README: README.md"
    )

    print(
        "\nEthical constraints:"
    )

    for constraint in deployment.ETHICAL_CONSTRAINTS:
        print(
            f"- {constraint}"
        )

    print(
        f"- {ETHICAL_WARNING}"
    )


if __name__ == "__main__":
    main()   
