import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import statsmodels.stats.api as sms

df_control_ = pd.read_excel("dsmlbc4/datasets/ab_testing_data.xlsx", sheet_name="Control Group")
df_control=df_control_.copy()
df_control.head()

df_test_ = pd.read_excel("dsmlbc4/datasets/ab_testing_data.xlsx", sheet_name="Test Group")
df_test=df_test_.copy()
df_test.head()

#Return Rate
df_control["Return Rate"]=df_control["Purchase"]/df_control["Impression"]
df_control.head()

df_test["Return Rate"]=df_test["Purchase"]/df_test["Impression"]
df_test.head()


############################
# AB Testing (Independent Sample T Test)
############################

# Used when two group means need to be compared.

# Q: Is there a difference between Return Rates of Control and Test Groups



############################
# 1. Assumption Check
############################

# 1.1 Assumption of Normality
# 1.2 Variance Homogeneity

############################
# 1.1 Assumption of Normality
############################

# H0: Assumption of Normality is fulfilled.
# H1:..not fulfilled.

from scipy.stats import shapiro
test_stat, pvalue = shapiro(df_control["Return Rate"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < 0.05  HO DENIED.
# p-value > 0.05 H0 Can not be Denied.
# Assumption is  valid for control group.

test_stat, pvalue = shapiro(df_test["Return Rate"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value < 0.05  HO DENIED.
# p-value > 0.05 H0 Can not be Denied.
# Assumption is  valid for test group.




test_stat, pvalue = stats.mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                              df.loc[df["smoker"] == "No", "total_bill"])


print('Test Stat = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
############################
# 1.2 Assumption of Variance Homogeneity
############################

# H0: Variances are Homogeneous
# H1: Variances are not Homogeneous

from scipy import stats
test_stat,pvalue=stats.levene(df_control["Return Rate"],\
             df_test["Return Rate"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < 0.05 HO DENIED.
# p-value > 0.05 H0 CANNOT BE DENIED.
# Variances are  Homogeneous.



############################
# 2. Application of Hypothesis
############################



# H0: M1 = M2 (... no statistical difference between two groups.)
# H1: M1 != M2 (...difference exists)



# 1.1 If asssumption are fulfilled Independent T-test(parametric test)
test_stat, pvalue = stats.ttest_ind(df_control["Return Rate"],\
                                    df_test["Return Rate"],\
                                    equal_var=True)

print('Test stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value <  0.05 HO DENIED.
# p-value > 0.05 H0 CANNOT BE DENIED.

# There is no significant statistical difference between Return Rates of Control and Test Groups
















