# 🛡️ NEXUSLEND CHURN PREDICTION PIPELINE

## Deployment Overview

This repository contains the production handoff package for the NexusLend churn prediction solution.

The deployment pipeline validates incoming data, handles unknown categories, provides safe defaults after failures, and includes a PSI monitoring hook.

**Pipeline Version:** 1.2

**Training Date:** 2024-03-09

**Data Hash:** 97c925c876e5f2d37c4bcb5f3959ed54e9d39851b2432a03ae2c5357c6063a60

**Model Hash:** 57891d295d94a0f9b6618f7eea8bdf9ad1116a53c87762966f14888da4454331

**Ethical Constraints:** 3

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

### 2. Run the deployment pipeline

```bash
python deployment_pipeline.py
```

### 3. Load the packaged inference pipeline

```python
from deployment_pipeline import DeploymentPipeline

pipe = DeploymentPipeline(
    "models/inference_pipeline.pkl"
).load_model()
```

## 📋 Input Schema

| Column | Type | Description |
|---|---|---|
| income | float | Monthly income in ZAR |
| debt | float | Customer debt |
| region | string | Customer region |
| load_shedding_hours | float | Load-shedding exposure |
| financial_strain | float | Financial strain indicator |

Unknown regions are mapped to `Other`.

The actual project model uses the Milestone 2 engineered feature `financial_strain_ratio`.

## Input Validation

- Checks that input is a pandas DataFrame.
- Rejects empty input.
- Checks required columns.
- Converts numeric fields safely.
- Detects invalid numeric values.
- Warns about income above R1,000,000.
- Maps unknown regions to `Other`.
- Handles missing engineered features.
- Preserves model feature order.

## Circuit Breaker

The circuit breaker activates when the prediction error rate exceeds **5%**.

When activated, safe default predictions are returned instead of continuing with potentially unreliable model outputs.

Ten consecutive prediction failures generate a critical operational warning.

## ⚠️ Critical Ethical Constraints

- Human review required for debt_to_income > 0.6
- Never deny service based solely on model output
- Monthly fairness audits mandatory
- Do NOT target high-strain customers with predatory offers
- Do not exploit financially vulnerable customers.
- Do not use the model for automatic loan denial.

## 🔍 Monitoring Commands

### Run the deployment pipeline

```bash
python deployment_pipeline.py
```

### Check the inference model

```bash
python -c "import os; print(os.path.exists('models/inference_pipeline.pkl'))"
```

### Future PSI drift monitoring

```bash
python monitor_drift.py --pipeline models/inference_pipeline.pkl
```

PSI interpretation:

- PSI < 0.10: stable
- PSI 0.10–0.25: moderate drift
- PSI > 0.25: significant drift

## 🆘 Failure Modes

### Unknown region

Unknown regions are mapped to `Other` and a warning is logged.

### Missing required input

Missing required columns result in a safe default prediction.

### Invalid numeric values

Invalid numeric values are converted to missing values and handled by the model safeguards.

### High prediction error rate

If the prediction error rate exceeds 5%, the circuit breaker returns safe default predictions.

### Repeated failures

Ten consecutive failures generate a critical warning.

## Model Metadata

- Version: 1.2
- Training date: 2024-03-09
- Data hash: 97c925c876e5f2d37c4bcb5f3959ed54e9d39851b2432a03ae2c5357c6063a60
- Model hash: 57891d295d94a0f9b6618f7eea8bdf9ad1116a53c87762966f14888da4454331
- Ethical constraints: 3

## Production Handoff

Before production deployment:

1. Validate the model artifact.
2. Confirm the model version.
3. Confirm the data hash.
4. Confirm the model hash.
5. Test input validation.
6. Test unknown-category handling.
7. Test the circuit breaker.
8. Run fairness checks.
9. Review drift monitoring.
10. Confirm human oversight.

## Responsible Use

This pipeline is a decision-support system.

Predictions must not be treated as facts about customers.

Human oversight is required for high-impact decisions.

The model must never be used to exploit customers experiencing financial hardship.
