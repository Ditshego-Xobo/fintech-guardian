# data_cleaner.py

import os
import pandas as pd
import numpy as np


class DataCleaningError(Exception):
    """Raised when data validation fails during cleaning."""
    pass


class DataCleaner:
    """Ethical data cleaning for South African fintech customer data."""

    def __init__(self, df):
        self.df = df.copy()
        self.ethical_notes = []
        self.cleaning_log = []
        self.region_variants_fixed = 0
        self.income_outliers_capped = 0
        self.negative_incomes_handled = 0

    def validate_income(self):
        """Convert income to numeric, handle negatives and cap outliers."""

        if "income" not in self.df.columns:
            raise DataCleaningError("Required column 'income' is missing.")

        # Convert income values such as R15,000 or R 15 000 to numbers
        income_clean = (
            self.df["income"]
            .astype("string")
            .str.replace("R", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.strip()
        )

        self.df["income"] = pd.to_numeric(
            income_clean,
            errors="coerce"
        )

        # Handle negative income values
        negative_mask = self.df["income"] < 0
        self.negative_incomes_handled = int(negative_mask.sum())

        if self.negative_incomes_handled > 0:
            self.df.loc[negative_mask, "income"] = np.nan
            self.cleaning_log.append(
                f"Handled {self.negative_incomes_handled} negative income values"
            )

        # Find the 99th percentile
        valid_income = self.df["income"].dropna()

        if valid_income.empty:
            raise DataCleaningError(
                "Income column contains no valid numeric values."
            )

        percentile_99 = valid_income.quantile(0.99)

        # Cap income values above the 99th percentile
        outlier_mask = self.df["income"] > percentile_99
        self.income_outliers_capped = int(outlier_mask.sum())

        self.df.loc[outlier_mask, "income"] = percentile_99

        self.cleaning_log.append(
            f"Capped {self.income_outliers_capped} income outliers "
            f"at the 99th percentile ({percentile_99:.2f})"
        )

        # Check whether township applicants have higher income missingness
        if "township_flag" in self.df.columns:

            township_values = (
                self.df["township_flag"]
                .astype("string")
                .str.strip()
                .str.lower()
            )

            township_mask = township_values.isin(
                ["1", "true", "yes", "y"]
            )

            township_missing_rate = self.df.loc[
                township_mask, "income"
            ].isna().mean()

            non_township_missing_rate = self.df.loc[
                ~township_mask, "income"
            ].isna().mean()

            if (
                pd.notna(township_missing_rate)
                and pd.notna(non_township_missing_rate)
                and non_township_missing_rate > 0
            ):
                difference = (
                    (township_missing_rate - non_township_missing_rate)
                    / non_township_missing_rate
                ) * 100

                if difference > 0:
                    self.ethical_notes.append(
                        f"WARNING: Township applicants show "
                        f"{difference:.1f}% higher income missingness "
                        f"than non-township applicants. This may indicate "
                        f"form accessibility or representation issues."
                    )

        self.cleaning_log.append("Income validation completed")

    def standardize_regions(self):
        """Map region variants to standardised names."""

        if "region" not in self.df.columns:
            raise DataCleaningError("Required column 'region' is missing.")

        region_map = {
            "JHB": "Johannesburg",
            "Joburg": "Johannesburg",
            "Jozi": "Johannesburg",
            "JHB CBD": "Johannesburg",
            "Greater Johannesburg": "Johannesburg",

            "CPT": "Cape Town",
            "CapeTown": "Cape Town",
            "Capetown": "Cape Town",
            "CT": "Cape Town",

            "DBN": "Durban",
            "Durbs": "Durban",
            "eThekwini": "Durban",
            "Durban Metro": "Durban",

            "PTA": "Pretoria",
            "Pta": "Pretoria",
            "Tshwane": "Pretoria",

            "PE": "Gqeberha",
            "Gqeberha": "Gqeberha",
            "Port Elizabeth": "Gqeberha",
            "Nelson Mandela Bay": "Gqeberha",

            "BFN": "Bloemfontein",
            "Bloem": "Bloemfontein",

            "EL": "East London",

            "PMB": "Pietermaritzburg",

            "Nelspruit": "Mbombela",
            "Mbombela": "Mbombela"
        }

        original_region = self.df["region"].copy()

        # Remove unnecessary spaces
        cleaned_region = (
            self.df["region"]
            .astype("string")
            .str.strip()
        )

        # Case-insensitive mapping
        lookup = {
            key.lower(): value
            for key, value in region_map.items()
        }

        self.df["region"] = cleaned_region.apply(
            lambda value: (
                lookup.get(str(value).lower(), value)
                if pd.notna(value)
                else np.nan
            )
        )

        # Count region values that were changed
        original_clean = (
            original_region
            .astype("string")
            .str.strip()
        )

        changed = (
            original_clean.fillna("<MISSING>")
            != self.df["region"].astype("string").fillna("<MISSING>")
        )

        self.region_variants_fixed = int(changed.sum())

        self.cleaning_log.append(
            f"Fixed {self.region_variants_fixed} region variants"
        )

    def create_missing_indicators(self):
        """Create binary indicators for missing critical values."""

        if "income" not in self.df.columns:
            raise DataCleaningError(
                "Required column 'income' is missing."
            )

        if "township_flag" not in self.df.columns:
            raise DataCleaningError(
                "Required column 'township_flag' is missing."
            )

        self.df["income_missing"] = (
            self.df["income"].isna().astype(int)
        )

        self.df["township_flag_missing"] = (
            self.df["township_flag"].isna().astype(int)
        )

        self.cleaning_log.append(
            "Created income_missing and township_flag_missing indicators"
        )

    def to_csv(self, output_path):
        """Save cleaned data using UTF-8 encoding and date formatting."""

        output_directory = os.path.dirname(output_path)

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        self.df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d %H:%M:%S"
        )

        self.cleaning_log.append(
            f"Saved cleaned data to {output_path}"
        )

    def __str__(self):
        """Return a summary of the cleaning process."""

        return (
            f"Cleaned {len(self.df):,} rows | "
            f"Fixed {self.region_variants_fixed:,} region variants | "
            f"Capped {self.income_outliers_capped:,} income outliers"
        )

    def run_full_cleaning(self):
        """Run all cleaning steps in the required ethical sequence."""

        try:
            self.validate_income()
            self.standardize_regions()
            self.create_missing_indicators()

            self.cleaning_log.append(
                "Full cleaning completed"
            )

            return self.df

        except DataCleaningError:
            raise

        except Exception as e:
            raise DataCleaningError(
                f"Cleaning failed: {str(e)}"
            ) from e


def main():
    """Load, clean and export the customer dataset."""

    input_path = "data/raw/customer_loans_q1_2024.csv"
    output_path = "data/processed/cleaned_customers.csv"

    try:
        df = pd.read_csv(input_path)

        cleaner = DataCleaner(df)

        cleaner.run_full_cleaning()

        cleaner.to_csv(output_path)

        print(cleaner)

        print("\nCleaning log:")
        for entry in cleaner.cleaning_log:
            print("-", entry)

        if cleaner.ethical_notes:
            print("\nEthical notes:")
            for note in cleaner.ethical_notes:
                print("-", note)
        else:
            print("\nEthical notes:")
            print("- Income missingness was measured separately for township and non-township applicants.")
            print("- No higher township income missingness was detected in this dataset.")

    except Exception as e:
        raise DataCleaningError(
            f"Data cleaning failed: {str(e)}"
        ) from e


if __name__ == "__main__":
    main()