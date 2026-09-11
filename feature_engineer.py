# feature_engineer.py

import os
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Create business-intelligent features with ethical safeguards."""

    def __init__(self, df):
        self.df = df.copy()
        self.feature_metadata = {}
        self.created_features = []
        self.removed_features = []
        self.max_vif = 0.0

    def create_financial_strain_ratio(self):
        """Calculate debt-to-income ratio with safe zero handling."""

        if "debt" not in self.df.columns or "income" not in self.df.columns:
            raise ValueError("Required columns 'debt' and 'income' are missing.")

        self.df["financial_strain_ratio"] = np.where(
            self.df["income"] > 0,
            self.df["debt"] / self.df["income"],
            0
        )

        self.feature_metadata["financial_strain_ratio"] = (
            "Identifies customers at risk of default due to financial overcommitment."
        )

        self.created_features.append("financial_strain_ratio")

    def encode_load_shedding_impact(self):
        """Convert load shedding hours into cyclical features."""

        if "load_shedding_hours" not in self.df.columns:
            raise ValueError("Required column 'load_shedding_hours' is missing.")

        # Use 24 hours as the cycle because the variable represents hours.
        self.df["load_shedding_sin"] = np.sin(
            2 * np.pi * self.df["load_shedding_hours"] / 24
        )

        self.df["load_shedding_cos"] = np.cos(
            2 * np.pi * self.df["load_shedding_hours"] / 24
        )

        self.feature_metadata["load_shedding_sin"] = (
            "Captures the cyclical impact of load shedding hours on customer activity."
        )

        self.feature_metadata["load_shedding_cos"] = (
            "Preserves the circular relationship between load shedding time periods."
        )

        self.created_features.extend([
            "load_shedding_sin",
            "load_shedding_cos"
        ])

    def regional_benchmarks(self, train_stats=None):
        """Add region-level aggregates without using the churn target."""

        if "region" not in self.df.columns:
            raise ValueError("Required column 'region' is missing.")

        if train_stats is None:
            # Training mode: calculate benchmarks from the current data.
            train_stats = (
                self.df.groupby("region")
                .agg(
                    regional_median_income=("income", "median"),
                    regional_mean_debt=("debt", "mean"),
                    regional_customer_count=("customer_id", "count")
                )
                .reset_index()
            )

        # Merge the precomputed regional statistics.
        self.df = self.df.merge(
            train_stats,
            on="region",
            how="left"
        )

        self.feature_metadata["regional_median_income"] = (
            "Provides regional income context for comparing a customer "
            "with the economic conditions of their region."
        )

        self.feature_metadata["regional_mean_debt"] = (
            "Provides regional debt context without using the churn outcome."
        )

        self.feature_metadata["regional_customer_count"] = (
            "Shows the number of customers represented in each region."
        )

        self.created_features.extend([
            "regional_median_income",
            "regional_mean_debt",
            "regional_customer_count"
        ])

        # Flag regions with fewer than 50 customers.
        small_regions = train_stats.loc[
            train_stats["regional_customer_count"] < 50,
            "region"
        ].tolist()

        if small_regions:
            self.feature_metadata["small_region_warning"] = (
                f"Warning: regions with fewer than 50 samples: {small_regions}"
            )

    def handle_support_tickets(self):
        """Create two-part features for zero-inflated support tickets."""

        if "support_tickets" not in self.df.columns:
            raise ValueError(
                "Required column 'support_tickets' is missing."
            )

        # Part 1: whether the customer has any support tickets.
        self.df["has_support_tickets"] = (
            self.df["support_tickets"] > 0
        ).astype(int)

        # Part 2: log transformation for the number of tickets.
        self.df["support_tickets_log"] = np.log1p(
            self.df["support_tickets"].clip(lower=0)
        )

        self.feature_metadata["has_support_tickets"] = (
            "Identifies customers who have contacted support, "
            "capturing the difference between zero and non-zero support activity."
        )

        self.feature_metadata["support_tickets_log"] = (
            "Reduces the effect of unusually high support-ticket counts "
            "while retaining information about support activity."
        )

        self.created_features.extend([
            "has_support_tickets",
            "support_tickets_log"
        ])       

    def check_multicollinearity(self):
        """Remove engineered features with correlation above 0.85."""

        numeric_features = [
            feature for feature in self.created_features
            if feature in self.df.columns
            and pd.api.types.is_numeric_dtype(self.df[feature])
        ]

        if len(numeric_features) < 2:
            return

        correlation_matrix = self.df[numeric_features].corr().abs()

        features_to_remove = set()

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                feature_a = correlation_matrix.columns[i]
                feature_b = correlation_matrix.columns[j]

                correlation = correlation_matrix.iloc[i, j]

                if pd.notna(correlation) and correlation > 0.85:
                    # Keep the first feature and remove the second.
                    features_to_remove.add(feature_b)

        for feature in features_to_remove:
            if feature in self.df.columns:
                self.df.drop(columns=[feature], inplace=True)

            if feature in self.created_features:
                self.created_features.remove(feature)

            self.removed_features.append(feature)

        self.feature_metadata["multicollinearity_check"] = (
            "Features with absolute pairwise correlation above 0.85 "
            "were removed to reduce multicollinearity."
        )

    def check_vif(self):
        """Check VIF and remove features until the maximum VIF is 5 or less."""

        while True:
            numeric_features = [
                feature for feature in self.created_features
                if feature in self.df.columns
                and pd.api.types.is_numeric_dtype(self.df[feature])
            ]

            if len(numeric_features) < 2:
                self.max_vif = 1.0
                break

            data = self.df[numeric_features].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            if len(data) < len(numeric_features) + 2:
                self.max_vif = 1.0
                break

            vif_values = {}

            for feature in numeric_features:
                y = data[feature].to_numpy(dtype=float)

                other_features = [
                    f for f in numeric_features if f != feature
                ]

                X = data[other_features].to_numpy(dtype=float)

                X = np.column_stack([
                    np.ones(len(X)),
                    X
                ])

                try:
                    coefficients = np.linalg.lstsq(
                        X, y, rcond=None
                    )[0]

                    predictions = X @ coefficients

                    ss_res = np.sum((y - predictions) ** 2)
                    ss_total = np.sum((y - y.mean()) ** 2)

                    if ss_total == 0:
                        vif = 1.0
                    else:
                        r_squared = 1 - (ss_res / ss_total)

                        if r_squared >= 0.999999:
                            vif = np.inf
                        else:
                            vif = 1 / (1 - r_squared)

                except np.linalg.LinAlgError:
                    vif = np.inf

                vif_values[feature] = vif

            # Find the feature with the highest VIF.
            worst_feature = max(
                vif_values,
                key=vif_values.get
            )

            worst_vif = vif_values[worst_feature]

            # Stop when all remaining features satisfy VIF <= 5.
            if worst_vif <= 5.0:
                self.max_vif = worst_vif
                break

            # Remove the feature with the highest VIF.
            self.df.drop(
                columns=[worst_feature],
                inplace=True
            )

            if worst_feature in self.created_features:
                self.created_features.remove(worst_feature)

            self.removed_features.append(worst_feature)

        self.feature_metadata["vif_check"] = (
            "Features were evaluated using variance inflation factor. "
            "The highest-VIF feature was removed repeatedly until the "
            "remaining engineered features had VIF of 5.0 or lower."
        )

    def __str__(self):
        """Return a summary of the feature engineering process."""

        return (
            f"Created {len(self.created_features)} new features | "
            f"Max VIF: {self.max_vif:.1f}"
        ) 

    def run_full_engineering(self):
        """Execute all feature engineering steps."""

        try:
            self.create_financial_strain_ratio()
            self.encode_load_shedding_impact()
            self.regional_benchmarks()
            self.handle_support_tickets()
            self.check_multicollinearity()
            self.check_vif()

            return self.df

        except Exception as e:
            raise ValueError(
                f"Feature engineering failed: {str(e)}"
            ) from e       

def main():
    """Load cleaned data, engineer features and save the result."""

    input_path = "data/processed/cleaned_customers.csv"
    output_path = "data/processed/engineered_features.csv"

    try:
        # Load the cleaned dataset from Milestone 1.
        df = pd.read_csv(input_path)

        # Create the feature engineer.
        engineer = FeatureEngineer(df)

        # Run all feature engineering steps.
        engineer.run_full_engineering()

        # Save the engineered dataset.
        engineer.df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(engineer)

        print("\nCreated features:")
        for feature in engineer.created_features:
            print("-", feature)

        if engineer.removed_features:
            print("\nRemoved features:")
            for feature in engineer.removed_features:
                print("-", feature)

        print("\nFeature metadata:")
        for feature, rationale in engineer.feature_metadata.items():
            print(f"- {feature}: {rationale}")

        print(f"\nSaved engineered data to {output_path}")

    except Exception as e:
        raise ValueError(
            f"Feature engineering failed: {str(e)}"
        ) from e


if __name__ == "__main__":
    main()                   