import pandas as pd
import statsmodels.formula.api as smf
from cfg import CFG
import numpy as np

data = pd.read_csv(f'../data/bigbird_bs{CFG.batch_size}_lr{CFG.learning_rate}.csv')
Ｘ = data.iloc[:, 3:7]
y = data.iloc[:, 2:3]
model = smf.ols('y~X', data=data).fit()
print(model.summary())


# Min–max normalization
# def min_max_y(raw_data):
#     min_max_data = []
#
#     # Min–max normalization
#     for d in raw_data:
#         min_max_data.append((d - min(raw_data)) / (max(raw_data) - min(raw_data)))
#     return min_max_data
#
#
# avg_word_embedding = np.array(data['avg_word_embedding'])
# a = min_max_y(avg_word_embedding)
# print(np.array(a))










'''
bigbird bs16 lr1e-4
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.001
Model:                            OLS   Adj. R-squared:                  0.001
Method:                 Least Squares   F-statistic:                     9.312
Date:                Wed, 06 Sep 2023   Prob (F-statistic):           1.62e-07
Time:                        10:23:46   Log-Likelihood:                -15927.
No. Observations:               28470   AIC:                         3.186e+04
Df Residuals:                   28465   BIC:                         3.190e+04
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      1.6478      0.012    138.733      0.000       1.625       1.671
X[0]          -0.0713      0.015     -4.622      0.000      -0.102      -0.041
X[1]           0.1521      0.104      1.456      0.145      -0.053       0.357
X[2]          -0.0330      0.013     -2.600      0.009      -0.058      -0.008
X[3]           0.0250      0.007      3.438      0.001       0.011       0.039
==============================================================================
Omnibus:                    10983.247   Durbin-Watson:                   0.975
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            35740.248
Skew:                          -2.034   Prob(JB):                         0.00
Kurtosis:                       6.685   Cond. No.                         58.0
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Process finished with exit code 0
'''

'''
bigbird bs16 lr1e-5
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.031
Model:                            OLS   Adj. R-squared:                  0.030
Method:                 Least Squares   F-statistic:                     224.8
Date:                Wed, 06 Sep 2023   Prob (F-statistic):          2.50e-190
Time:                        10:32:09   Log-Likelihood:                -24291.
No. Observations:               28470   AIC:                         4.859e+04
Df Residuals:                   28465   BIC:                         4.863e+04
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.3988      0.016     25.027      0.000       0.368       0.430
X[0]           0.1441      0.021      6.964      0.000       0.104       0.185
X[1]           3.3384      0.140     23.815      0.000       3.064       3.613
X[2]           0.1483      0.017      8.719      0.000       0.115       0.182
X[3]           0.1098      0.010     11.269      0.000       0.091       0.129
==============================================================================
Omnibus:                     5245.305   Durbin-Watson:                   1.916
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2484.745
Skew:                           0.567   Prob(JB):                         0.00
Kurtosis:                       2.100   Cond. No.                         58.0
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Process finished with exit code 0

'''

'''
bigbird bs16 lr3e-5
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.006
Model:                            OLS   Adj. R-squared:                  0.006
Method:                 Least Squares   F-statistic:                     41.97
Date:                Wed, 06 Sep 2023   Prob (F-statistic):           3.82e-35
Time:                        10:42:44   Log-Likelihood:                -30213.
No. Observations:               28470   AIC:                         6.044e+04
Df Residuals:                   28465   BIC:                         6.048e+04
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      1.0299      0.020     52.497      0.000       0.991       1.068
X[0]           0.0198      0.025      0.775      0.438      -0.030       0.070
X[1]           1.7203      0.173      9.967      0.000       1.382       2.059
X[2]           0.0822      0.021      3.928      0.000       0.041       0.123
X[3]           0.0863      0.012      7.192      0.000       0.063       0.110
==============================================================================
Omnibus:                   246180.843   Durbin-Watson:                   0.834
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3037.114
Skew:                          -0.377   Prob(JB):                         0.00
Kurtosis:                       1.589   Cond. No.                         58.0
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Process finished with exit code 0
'''


















