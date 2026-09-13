# report_generator.py

import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Required ethical warning from the assignment.
ETHICAL_WARNING = (
    "Do NOT target high-strain customers with predatory offers"
)

CFO_QUOTE = (
    "Per CFO: 'Every 1% churn reduction = R420k saved'"
)


class ReportGenerator:
    """Transform model outputs into boardroom-ready business actions."""

    def __init__(
        self,
        model,
        X,
        region_col="region",
        strain_col="financial_strain_ratio"
    ):
        self.model = model
        self.X = X.copy()
        self.region_col = region_col
        self.strain_col = strain_col

        self.interventions = {}
        self.heatmap_data = None

    def calculate_intervention_roi(self):
        """Calculate financial and inclusion impact for three interventions."""

        # Estimate the high-strain population using the upper quartile.
        high_strain_threshold = self.X[
            self.strain_col
        ].quantile(0.75)

        high_strain_customers = self.X[
            self.X[self.strain_col] >= high_strain_threshold
        ].copy()

        eligible_customers = len(
            high_strain_customers
        )

        # Estimate the proportion of eligible customers
        # from township locations.
        if "township_flag" in high_strain_customers.columns:
            township_share = (
                pd.to_numeric(
                    high_strain_customers["township_flag"],
                    errors="coerce"
                )
                .fillna(0)
                .mean()
            )
        else:
            township_share = 0.0

        # The intervention assumptions are explicit so that
        # financial impact can be audited and changed later.
        interventions = {
            "Fee Waiver for High-Strain": {
                "cost": 185000,
                "saved_customers": 320,
                "lifetime_value": 6500,
                "description": (
                    "Temporary fee relief for financially strained "
                    "customers to reduce avoidable churn."
                )
            },
            "Load-Shedding Support Package": {
                "cost": 160000,
                "saved_customers": 250,
                "lifetime_value": 6000,
                "description": (
                    "Targeted service support for customers experiencing "
                    "higher load-shedding exposure."
                )
            },
            "Flexible Payment Support": {
                "cost": 140000,
                "saved_customers": 220,
                "lifetime_value": 5500,
                "description": (
                    "Flexible payment arrangements designed to support "
                    "customers experiencing temporary financial pressure."
                )
            }
        }

        for name, details in interventions.items():

            gross_value = (
                details["saved_customers"]
                * details["lifetime_value"]
            )

            net_impact = (
                gross_value
                - details["cost"]
            )

            roi = (
                net_impact
                / details["cost"]
            )

            inclusion_customers = round(
                details["saved_customers"]
                * township_share
            )

            inclusion_rate = (
                inclusion_customers
                / details["saved_customers"]
                if details["saved_customers"] > 0
                else 0
            )

            details["eligible_customers"] = (
                eligible_customers
            )

            details["gross_value"] = (
                gross_value
            )

            details["net_impact"] = (
                net_impact
            )

            details["roi"] = (
                roi
            )

            details["township_customers_supported"] = (
                inclusion_customers
            )

            details["inclusion_rate"] = (
                inclusion_rate
            )

            details["high_strain_threshold"] = (
                float(high_strain_threshold)
            )

        # Sort by net financial impact.
        self.interventions = dict(
            sorted(
                interventions.items(),
                key=lambda item: item[1]["net_impact"],
                reverse=True
            )
        )

        return self.interventions

    def generate_churn_heatmap(
        self,
        output_path="reports/churn_heatmap.png"
    ):
        """Create a regional churn-risk heatmap by financial-strain quartile."""

        if self.strain_col not in self.X.columns:
            raise ValueError(
                f"Required strain column '{self.strain_col}' is missing."
            )

        if self.region_col not in self.X.columns:
            raise ValueError(
                f"Required region column '{self.region_col}' is missing."
            )

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        # Generate churn probabilities from the saved model.
        probabilities = self.model.predict_proba(
            self.X
        )[:, 1]

        heatmap_data = self.X[
            [
                self.region_col,
                self.strain_col
            ]
        ].copy()

        heatmap_data["churn_probability"] = (
            probabilities
        )

        # Create four financial-strain groups.
        try:
            heatmap_data["strain_quartile"] = pd.qcut(
                heatmap_data[self.strain_col],
                q=4,
                labels=[
                    "Q1 - Lowest",
                    "Q2",
                    "Q3",
                    "Q4 - Highest"
                ],
                duplicates="drop"
            )
        except ValueError:
            heatmap_data["strain_quartile"] = pd.cut(
                heatmap_data[self.strain_col],
                bins=4,
                labels=[
                    "Q1 - Lowest",
                    "Q2",
                    "Q3",
                    "Q4 - Highest"
                ]
            )

        # Calculate mean predicted churn risk.
        pivot_data = heatmap_data.pivot_table(
            index=self.region_col,
            columns="strain_quartile",
            values="churn_probability",
            aggfunc="mean"
        )

        # Keep strain groups in logical order.
        desired_columns = [
            "Q1 - Lowest",
            "Q2",
            "Q3",
            "Q4 - Highest"
        ]

        existing_columns = [
            column
            for column in desired_columns
            if column in pivot_data.columns
        ]

        pivot_data = pivot_data[
            existing_columns
        ]

        # Store the calculated data for possible reuse.
        self.heatmap_data = pivot_data

        # Create the heatmap.
        plt.figure(
            figsize=(11, 7)
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".1%",
            cmap="YlOrRd",
            linewidths=0.5,
            cbar_kws={
                "label": "Mean Predicted Churn Risk"
            }
        )

        plt.title(
            "Regional Churn Risk by Financial-Strain Quartile"
        )

        plt.xlabel(
            "Financial-Strain Quartile"
        )

        plt.ylabel(
            "Region"
        )

        # Ethical disclaimer on the visual itself.
        plt.figtext(
            0.5,
            0.01,
            "Ethical disclaimer: "
            "Do NOT target high-strain customers with predatory offers. "
            "Use risk insights only to provide supportive interventions.",
            ha="center",
            fontsize=9
        )

        plt.tight_layout(
            rect=[0, 0.04, 1, 1]
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        return output_path

    def generate_executive_summary(
        self,
        output_path="reports/executive_summary.md"
    ):
        """Create a boardroom-ready executive summary."""

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        interventions = (
            self.calculate_intervention_roi()
        )

        if not interventions:
            raise ValueError(
                "No intervention calculations were generated."
            )

        top_intervention = next(
            iter(interventions.items())
        )

        top_name = top_intervention[0]
        top_details = top_intervention[1]

        total_net_impact = sum(
            details["net_impact"]
            for details in interventions.values()
        )

        total_cost = sum(
            details["cost"]
            for details in interventions.values()
        )

        total_saved_customers = sum(
            details["saved_customers"]
            for details in interventions.values()
        )

        total_township_support = sum(
            details["township_customers_supported"]
            for details in interventions.values()
        )

        overall_roi = (
            total_net_impact / total_cost
            if total_cost > 0
            else 0
        )

        # Format intervention details.
        intervention_lines = []

        for name, details in interventions.items():

            intervention_lines.append(
                f"""### {name}

- Investment: R{details['cost']:,.0f}
- Customers potentially retained: {details['saved_customers']:,}
- Estimated lifetime value: R{details['lifetime_value']:,.0f} per customer
- Gross value: R{details['gross_value']:,.0f}
- Net financial impact: R{details['net_impact']:,.0f}
- ROI: {details['roi']:.0%}
- Estimated township customers supported: {details['township_customers_supported']:,}
- Inclusion rate: {details['inclusion_rate']:.1%}
- Rationale: {details['description']}
"""
            )

        intervention_section = "\n".join(
            intervention_lines
        )

        summary = f"""# NEXUSLEND CHURN INTERVENTION PLAN

## Problem

The churn model identifies customers who may be at higher risk of
leaving the service. Financial strain and operational conditions such
as load-shedding can affect customer experience and financial stability.

The model should therefore be used to identify opportunities for
support rather than to punish customers experiencing financial pressure.

## Solution

The recommended approach is to use churn-risk insights to design
supportive interventions:

1. Fee relief for customers experiencing high financial strain.
2. Load-shedding support for customers exposed to service disruption.
3. Flexible payment support for customers experiencing temporary
   financial pressure.

### Recommended Priority

The highest-net-impact intervention is **{top_name}**.

- Investment: R{top_details['cost']:,.0f}
- Potential customers retained: {top_details['saved_customers']:,}
- Estimated gross value: R{top_details['gross_value']:,.0f}
- Estimated net impact: R{top_details['net_impact']:,.0f}
- ROI: {top_details['roi']:.0%}

## ROI

Three interventions were evaluated using explicit cost, customer
retention and lifetime-value assumptions.

{intervention_section}

### Portfolio Impact

- Total investment: R{total_cost:,.0f}
- Potential customers retained: {total_saved_customers:,}
- Estimated township customers supported: {total_township_support:,}
- Total projected net financial impact: R{total_net_impact:,.0f}
- Portfolio ROI: {overall_roi:.0%}
- Projected ROI: R{total_net_impact:,.0f}

{CFO_QUOTE}

## Ethical Guardrails

**{ETHICAL_WARNING}**

The model must be used to identify opportunities for customer support,
not opportunities to exploit customers experiencing financial
difficulty.

High-strain customers should receive transparent, affordable and
voluntary support options. Recommendations must not increase
financial hardship.

The model should not be used for automatic loan denial or other
high-impact decisions without appropriate human review.

## Inclusion Impact

Inclusion should be measured alongside financial return.

The interventions estimate how many customers from township locations
could benefit from the proposed support. This metric should be
monitored alongside ROI to ensure that commercial outcomes do not
come at the expense of vulnerable customer groups.

## Next Steps

1. Pilot the highest-value supportive intervention with human oversight.
2. Measure actual churn reduction against the model's expected impact.
3. Compare intervention uptake across regions and customer groups.
4. Monitor financial and inclusion outcomes monthly.
5. Review customer feedback before expanding an intervention.
6. Update assumptions when observed retention and lifetime-value
   outcomes differ materially from the estimates.

## Monitoring Triggers

- [ ] Retrain if township recall drops >10%
- [ ] Investigate if regional churn-risk patterns change substantially.
- [ ] Review intervention outcomes monthly.
- [ ] Monitor calibration and model performance.
- [ ] Review inclusion metrics for underserved regions.
- [ ] Stop or redesign an intervention if it increases customer
      financial hardship.

## Stakeholder Perspective

> {CFO_QUOTE}

Business performance should be balanced with customer wellbeing,
financial inclusion and responsible use of predictive analytics.

## Conclusion

The objective is not simply to reduce churn. The objective is to
understand why customers may leave and respond with fair, transparent
and supportive interventions.

Data science should serve people as well as business performance.
"""

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(summary)

        return output_path

    def __str__(self):
        """Return a concise intervention summary."""

        if not self.interventions:
            self.calculate_intervention_roi()

        total_net_impact = sum(
            details["net_impact"]
            for details in self.interventions.values()
        )

        return (
            f"Generated {len(self.interventions)} interventions | "
            f"Projected ROI: R{total_net_impact:,.0f}"
        )


def main():
    """Load model and engineered data, then generate all reports."""

    data_path = (
        "data/processed/engineered_features.csv"
    )

    model_path = (
        "models/churn_pipeline.pkl"
    )

    heatmap_path = (
        "reports/churn_heatmap.png"
    )

    summary_path = (
        "reports/executive_summary.md"
    )

    # Load engineered data.
    df = pd.read_csv(
        data_path
    )

    # The model was trained without the target column.
    if "churned" in df.columns:
        X = df.drop(
            columns=["churned"]
        )
    else:
        X = df.copy()

    # Load Milestone 3 model.
    with open(
        model_path,
        "rb"
    ) as file:
        model = pickle.load(file)

    # Validate required columns.
    required_columns = [
        "region",
        "financial_strain_ratio"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing_columns)
        )

    generator = ReportGenerator(
        model=model,
        X=X,
        region_col="region",
        strain_col="financial_strain_ratio"
    )

    # Calculate ROI.
    generator.calculate_intervention_roi()

    # Generate heatmap.
    generator.generate_churn_heatmap(
        heatmap_path
    )

    # Generate executive summary.
    generator.generate_executive_summary(
        summary_path
    )

    print(generator)

    print(
        f"\nSaved heatmap to {heatmap_path}"
    )

    print(
        f"Saved executive summary to {summary_path}"
    )

    print(
        "\nEthical warning:"
    )

    print(
        ETHICAL_WARNING
    )

    print(
        "\nStakeholder quote:"
    )

    print(
        CFO_QUOTE
    )


if __name__ == "__main__":
    main()