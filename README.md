# Loan Default Risk Modeling

## Objective
Build and compare a heuristic and logistic regression model
to assess loan default risk.

## Project Structure
- notebooks/ : EDA, heuristic model, logistic regression
- data/      : input dataset
- outputs/   : model outputs for BI analysis

## Key Results
- Heuristic model: KS ≈ 13%, Lift ≈ 1.23
- Logistic regression: KS ≈ 20%, Top-decile Lift ≈ 2.0

## Power BI Dashboard
An interactive dashboard built on logistic regression outputs to visualize:
- Risk deciles and monotonic default rates
- Lift concentration in top deciles
- Approval vs rejection distribution
- Predicted PD distribution
📁 File: `Loan_Default_Risk_Dashboard.pbix`
