"""
@author: akshita
"""
#%%
#Stage: Input variables

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("D:\AKSHITA ONLY\loan_default_project\loan_data.csv")

df.dtypes
heur_vars = [
    'CreditScore',
    'DTIRatio',
    'Income',
    'MonthsEmployed',
    'Education',
    'EmploymentType',
    'LoanPurpose']

#%%
#Stage: Binning continuous variables

df['CreditScore'].describe()
df['Band_CreditScore'] = pd.cut(
    df['CreditScore'], 
    bins= [300, 580, 670, 740, 900],
    labels= ['Poor', 'Fair', 'Good', 'Excellent'],
    include_lowest=True)

df['DTIRatio'].describe()
df['Band_DTI'] = pd.cut(
    df['DTIRatio'], 
    bins= [0.1, 0.3, 0.5, 0.9],
    labels= ['Low', 'Medium', 'High'],
    include_lowest=True)

df['Income'].describe()
df['Band_Income'] = pd.qcut(
    df['Income'], 
    q=3, 
    labels= ['Low', 'Medium', 'High'])

df['MonthsEmployed'].describe()
df['Band_MonthsEmployed'] = pd.cut(
    df['MonthsEmployed'], 
    bins= [0, 12, 36, 120], 
    labels= ['Short', 'Medium', 'Long'], 
    include_lowest=True)

#%%
#Stage: Assigning risk points to bands

df['RuleBasedRiskScore'] = 0
df['RuleBasedRiskScore'] = df['RuleBasedRiskScore'].astype(int)

#Band_CreditScore
credit_points = {
    'Poor': 3,
    'Fair': 2,
    'Good': 1,
    'Excellent': 0}
df['RuleBasedRiskScore'] += (df['Band_CreditScore']
                             .map(credit_points)
                             .fillna(2)
                             .astype(int))

#Band_DTI
dti_points = {
    'Low': 0,
    'Medium': 1,
    'High': 3}
df['RuleBasedRiskScore'] += (df['Band_DTI']
                             .map(dti_points)
                             .fillna(1)
                             .astype(int))

#Band_Income
income_points = {
    'Low': 2,
    'Medium': 1,
    'High': 0}
df['RuleBasedRiskScore'] += (df['Band_Income']
                             .map(income_points)
                             .fillna(1)
                             .astype(int))

#Band_MonthsEmployed
emp_len_points = {
    'Short': 2,
    'Medium': 1,
    'Long': 0}
df['RuleBasedRiskScore'] += (df['Band_MonthsEmployed']
                             .map(emp_len_points)
                             .fillna(1)
                             .astype(int))

#%%
#Stage: Assigning points for categorical variables

#Education
df['Education'].value_counts(dropna=False)
df.groupby(['Education']).agg({'Default' : 'mean'}).mul(100).round(2)

edu_points = {
    "High School": 2,
    "Bachelor's": 1,
    "Master's": 0,
    "PhD": 0}

df['RuleBasedRiskScore'] += (
    df['Education']
    .map(edu_points)
    .fillna(1))


#EmploymentType
df['EmploymentType'].value_counts(dropna=False)
df.groupby(['EmploymentType']).agg({'Default' : 'mean'}).mul(100).round(2)

emp_type_points = {
    'Full-time': 0,
    'Self-employed': 1,
    'Part-time': 1,
    'Unemployed': 3}

df['RuleBasedRiskScore'] += (
    df['EmploymentType']
    .map(emp_type_points)
    .fillna(1))


#LoanPurpose
df['LoanPurpose'].value_counts(dropna=False)
df.groupby(['LoanPurpose']).agg({'Default' : 'mean'}).mul(100).round(2)

ln_purpose_points = {
    'Home': 0,
    'Education': 1,
    'Auto': 1,
    'Other': 1,
    'Business': 2}

df['RuleBasedRiskScore'] += (
    df['LoanPurpose']
    .map(ln_purpose_points)
    .fillna(1))

#%%
#Stage: Creating Risk Decile

df['RiskDecile'] = (pd.qcut(
    df['RuleBasedRiskScore'],
    q=10,
    labels=False,
    duplicates='drop' 
    ) + 1)

#%%
#Stage: Model Evaluation

#Default rate by RiskDecile
decile_default = (df.groupby(['RiskDecile']).
agg({'Default' : 'mean'})
.reset_index()
.sort_values('RiskDecile'))

print(decile_default)


#Adjustments
df['RiskBand'] = df['RiskDecile']
#Comment:
'''Due to the discrete and intentionally coarse nature of the 
rule-based score, only six distinct risk bands were formed 
despite attempting decile binning. 
This reflects the limited scope of heuristic models and 
is expected behavior.'''


#Monotonicity check
decile_default.plot(
    x='RiskDecile',
    y='Default',
    title='Default Rate by Rule-Based Risk Decile')
plt.show()


#KS statistic
from scipy.stats import ks_2samp

ks = ks_2samp(
    df.loc[df['Default'] == 1, 'RuleBasedRiskScore'],
    df.loc[df['Default'] == 0, 'RuleBasedRiskScore']
).statistic

print(ks)


#Lift(top band vs overall)
overall_default = df['Default'].mean()
 
top_band_default = df[df['RiskBand'] >= 5]['Default'].mean()

lift = top_band_default / overall_default

print(lift)

#%%
#Stage: Summary
'''A rule-based risk score was constructed using a 
limited set of borrower characteristics identified during EDA
and the variables were discretized into intuitive bands. 
The resulting risk bands show monotonic separation, 
with KS of 13% and lift value of 1.25, as expected for a heuristic. 
These results establish a baseline against which a 
statistical model can be evaluated.'''
#%%




