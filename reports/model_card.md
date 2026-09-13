# Churn Prediction Model Card

## Model Purpose

This model predicts the likelihood that a fintech customer may churn.
It is intended to support human decision-making and customer retention
strategies, not to make automatic lending decisions.

## Performance

- Overall Recall: 98.9%
- Maximum Regional Recall Disparity: 2.8%
- Decision Threshold: 0.10

### Regional Recall

- Cape Town: 99.0%
- Durban: 99.8%
- Gauteng: 99.3%
- Gqeberha: 100.0%
- Johannesburg: 97.2%
- Johburg: 98.6%
- Pretoria: 98.5%
- Western Cape: 99.3%

## Interpretability

The model uses feature-level explanations to help identify important
drivers of predicted churn.

### Top Feature Insights

- **numeric__township_flag**: numeric__township_flag contributes to the model's churn-risk prediction and should be reviewed alongside customer context.
- **numeric__load_shedding_hours**: numeric__load_shedding_hours contributes to the model's churn-risk prediction and should be reviewed alongside customer context.
- **numeric__load_shedding_cos**: numeric__load_shedding_cos contributes to the model's churn-risk prediction and should be reviewed alongside customer context.

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
