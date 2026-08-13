from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

'''
Purpose:
    Provide an example of ordinary least squares (OLS) regression via the
    scikit-learn package. Resources aided with the tutorial, but some
    variations exist.
    
    Techniques outlined here are for: 1) Single variable linear regression
    
    Assumptions:
        1) Utilizing entire dataset for model predictions, and not
    splitting into training and testing groups
    
    The target (y) variable is the median house value for California districts,
expressed in hundreds of thousands of dollars ($100,000).

Resources:
    https://www.datacamp.com/tutorial/sklearn-linear-regression
    https://www.datacamp.com/blog/a-beginner-s-guide-to-the-machine-learning-workflow
    https://www.statsmodels.org/0.6.1/examples/notebooks/generated/ols.html
    https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html
'''

# Step 1: Load in sample dataset
housing_all = fetch_california_housing()
descriptions = housing_all.DESCR
housing_data = housing_all.data
housing_data_df = pd.DataFrame(data=housing_data,
                               columns=housing_all.feature_names)
housing_data_df.head()
housing_data_df.columns

# Selected one
X_slr = housing_data_df.MedInc.to_frame()

# Want to predict 
y_observed = housing_all.target
    
# Step X: Summarize variables individually 

# Step X: Initialize linear regression
model = LinearRegression()
# LinearRegression? help

# Fit the model to the training data.
# model.fit? Needs to be arrays
model.fit(X_slr, y_observed)

# Make predictions on the testing data.
y_predict = model.predict(X_slr)

# Step X: Evaulate model performance in terms of OLS assumptions, such as:
# Heteroskedasticity:
residuals = y_predict - y_observed
#fitted_residuals = pd.DataFrame(data=[y_pred,residuals]).T

# Create plots.
plt.figure(figsize=(12,5))
# Plot 1: Residuals Distribution.
sns.scatterplot(x=y_predict, y=residuals)
'''
Heavy presence of a relationship b/w residuals and fitted, indicating
likely heteroskedasticity.
Also supported by a separate analysis:
    https://www.kaggle.com/code/teesh841/california-housing-prices-regression-analysis#2.-Univariate-Analysis
    Suggesting spatial autocorrelation
Therefore, multiple linear regression wont be useful if its
present in the single linear regression case.
'''
    
# Multicollinearity:
    
# Error distribution (i.e., should be 'normal')
    
# Calculate and print R^2 score.
X2 = sm.add_constant(X_slr)
mod = sm.OLS(y_observed,X2).fit()
ols_slr_r2 = mod.rsquared
print(f"R-squared: {ols_slr_r2:.2f}") 

# Alternate way to perform OLS as a means of code validation
r2 = r2_score(y_observed, y_predict)
print(f"R-squared: {r2:.2f}")

mod.summary()
