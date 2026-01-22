#Credit Risk Modeling: Heuristic Baseline vs Logistic Regression

##Problem Statement
This project demonstrates a credit risk segmentation and default prediction workflow using a rule-based heuristic baseline and a logistic regression model, evaluated through risk deciles, KS, lift, and business requirement.

##Baseline Approach: Rule-Based Heuristic Risk Score
A rule-based risk score was built to serve as a baseline, reflecting how many lending institutions initially segment credit risk.
-Designed for interpretability and quick decision-making
-Produces stable risk deciles and monotonic risk ordering but limited discriminatory power
-Baseline performance: KS = 13%, Lift = 1.25 (expected for heuristic methods)
This baseline is used strictly as a reference point.

##Logistic Regression Enhancement
A logistic regression model was implemented using the same feature set to ensure a fair comparison.
-Feature standardization and multicollinearity checks applied
-Coefficients interpreted to identify key directional risk drivers
-Improved separation in higher-risk deciles compared to the baseline: KS = 20%, Lift = 2.0

##Model Evaluation
Models were evaluated using industry-relevant metrics:
-KS statistic
-AUC
-Lift and default rate by risk decile
-Monotonicity of risk ordering
A business-aligned cutoff was selected to assess portfolio-level trade-offs.

##Business Implications
Model outputs were translated into actionable insights using a Power BI dashboard.
-Comparison of heuristic vs logistic models across risk deciles
-Impact of rejecting the top 20% highest-risk loans on default rate and approval volume
-Demonstrates how model performance supports practical credit risk decisions

##Limitations & Next Steps
-Limited feature depth and linear decision boundary
-Future work may explore non-linear models (e.g., gradient boosting) for incremental gains
