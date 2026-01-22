# -*- coding: utf-8 -*-
"""
@author: akshita
"""
#%%
#Stage: Importing packages and reading data

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix)

from scipy.stats import ks_2samp

df = pd.read_csv("D:\AKSHITA ONLY\loan_default_project\loan_data.csv")

df.shape

#%%
#Stage: Feature selection

# Target
target = 'Default'

# Selected features
num_features = [
    'CreditScore',
    'DTIRatio',
    'Income',
    'MonthsEmployed']

cat_features = [
    'Education',
    'EmploymentType',
    'LoanPurpose']

features = num_features + cat_features

df_model = df[features + [target]].copy()
df_model.shape
df_model.head()

#%%
#Stage: Train-test split

x = df_model.drop(columns=target)
y = df_model[target]

temp = train_test_split(
    x,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42)

x_train = temp[0]
x_test = temp[1]
y_train = temp[2]
y_test = temp[3]

x_train.shape, x_test.shape

#%%
#Stage: Encoding categoricals

x_train_enc = pd.get_dummies(
    x_train,
    columns=cat_features,
    drop_first=True)

x_test_enc = pd.get_dummies(
    x_test,
    columns=cat_features,
    drop_first=True)

# Align columns
x_train_enc, x_test_enc = x_train_enc.align(
    x_test_enc,
    join="left",
    axis=1,
    fill_value=0)

#%%
#Stage: Scaling numeric features

scaler = StandardScaler()

x_train_enc[num_features] = scaler.fit_transform(
    x_train_enc[num_features])

x_test_enc[num_features] = scaler.transform(
    x_test_enc[num_features])

#%%
#Stage: Multicollinearity check

plt.figure(figsize=(10, 6))
sns.heatmap(
    x_train_enc[num_features].corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2g')
plt.title('Correlation Matrix – Numeric Features')
plt.show()

#%%
#Stage: Logisitic Regression Model Fitting

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42)

lr_model.fit(x_train_enc, y_train)

#%%
#Stage: Coefficient Interpretation

coefficients = lr_model.coef_[0]
feature_names = x_train_enc.columns

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients})

coef_df['Odds_Ratio'] = np.exp(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Coefficient', ascending=False)

coef_df

#Comment:
'''Unstable employment categories create higher default risk.
Higher income and longer employment term lower default risk.'''
#%%
#Stage: Model Evaluation

# Predicted probabilities

train_probabilities = lr_model.predict_proba(x_train_enc)
test_probabilities  = lr_model.predict_proba(x_test_enc)

y_train_pred = train_probabilities[:, 1]
y_test_pred  = test_probabilities[:, 1]


# AUC

train_auc = roc_auc_score(y_train, y_train_pred)
test_auc = roc_auc_score(y_test, y_test_pred)

print(train_auc, test_auc)


# KS Statistic

ks_stat = ks_2samp(
    y_test_pred[y_test == 1],
    y_test_pred[y_test == 0]).statistic

print(ks_stat)


# Risk deciles and Lift

eval_df = x_test.copy()
eval_df['ActualDefault'] = y_test.values
eval_df['PredictedPD'] = y_test_pred

eval_df['LR_RiskDecile'] = (
    pd.qcut(eval_df['PredictedPD'], 10, labels=False) + 1)

decile_summary = (
    eval_df
    .groupby('LR_RiskDecile')
    .agg(
        Count=('ActualDefault', 'count'),
        DefaultRate=('ActualDefault', 'mean'))
    .reset_index())

overall_default = eval_df['ActualDefault'].mean()
decile_summary['Lift'] = decile_summary['DefaultRate'] / overall_default

print(decile_summary)


#Comment:
'''Train AUC and test AUC values of roughly 0.63 indicate
moderate discrimination power.
Train and test AUC are almost identical, which indicates 
no overfitting.
KS value of 19.9% indicates moderate separation power 
and is a clear improvement over the heuristic baseline.
DefaultRate increases monotonically from decile 1 to 10.
Lift increases steadily, peaking in the top decile.'''
#%%
#Stage: Business cutoff

cutoff = eval_df['PredictedPD'].quantile(0.8)

eval_df['Approved'] = (eval_df['PredictedPD'] < cutoff).astype(int)

approval_rate = eval_df['Approved'].mean()

post_default_rate = eval_df.loc[
    eval_df['Approved'] == 1, 'ActualDefault'].mean()

print(approval_rate, post_default_rate)

#%%
#Stage: Outputs for Power BI dashboard

# Loan-level output
loan_level_output = eval_df.copy()

loan_level_output.to_csv(
    'lr_loan_level_output.csv',
    index=False)

# Decile-level output
decile_summary.to_csv(
    'lr_decile_summary.csv',
    index=False)

#%%
#Stage: Conclusion

'''Logistic regression improves upon the heuristic baseline 
by providing better rank-ordering of risk and probabilistic outputs. 
The model demonstrates modest but meaningful gains in KS and lift, 
validating the value of statistical modeling over 
purely rule-based approaches.'''


