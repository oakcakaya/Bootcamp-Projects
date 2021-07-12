import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import statsmodels.stats.api as sms



df_ = pd.read_csv("dsmlbc4/datasets/pricing.csv", sep=';')
df=df_.copy()
df.head()

df.describe().T

df.isnull().value_counts()

df.hist(column="price", bins=50,)
plt.show()

df.sort_values("price",ascending=False).head(100)
df["price"].nunique()
df.describe(percentiles=[.1,.25,.50,.75,.85,.90,.95,.96,.97,.98,.99])


# Since the price values goes abrupt  at 95% (>108), this percentile is selected as threshold value
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

outlier_thresholds(df, "price")
replace_with_thresholds(df, "price")

df.describe(percentiles=[.1,.25,.50,.75,.85,.90,.95,.96,.97,.98,.99])


df["category_id"].nunique()
df["category_id"].value_counts()

df_cat=df.groupby("category_id").agg({"price":"mean"})
df_cat["counts"]=df["category_id"].value_counts()
df_cat

#There are 6 distinct categories
#Average prices for each category need to be compared by Independent Samples T test
df_cat.columns=["avg_price","counts"]
df_cat

############################
# 1. Independent Samples T test
############################


############################
# 1.1 Assumption Check
############################

# 1.1.1 Assumption of Normality
# 1.2.2 Variance Homogeneity

############################
# 1.1 Assumption of Normality
############################

# H0: Assumption of Normality is fulfilled.
# H1:..not fulfilled.

from scipy.stats import shapiro
for i in df["category_id"].unique():
    test_stat, pvalue = shapiro(df.loc[df["category_id"] ==  i, "price"])
    print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < 0.05  HO Rejected .
# p-value > 0.05 H0 Can not be Rejected.
# Assumption of normality is violated for any of the categories
# Since the Assumption of normality is violated, non parametric test will be conducted
# mannwhitneyu test is will be conducted among the categories

############################
# 2.1 Hypothesis Testing
############################

# H0: M1 = M2 (... There is no significant statistical difference between two groups.
# H1: M1 != M2 (...difference exists)

from scipy import stats
import itertools

combinations = []

for combination in itertools.combinations(df["category_id"].unique(),2):
    combinations.append(combination)
combinations

import warnings
warnings.filterwarnings("ignore")

for combination in combinations:
    test_stat,pvalue = stats.mannwhitneyu(df.loc[df["category_id"] ==  combination[0],"price"],df.loc[df["category_id"] ==  combination[1],"price"] )
        print("{0} - {1} -- ".format(combination[0],combination[1]),'Test stat = %.4f, p-Value = %.4f' % (test_stat, pvalue))



"""
489756 - 361254 --  Test statistic = 380060.0000, p-Value = 0.0000 REJECTED
489756 - 874521 --  Test statistic = 519398.0000, p-Value = 0.0000 REJECTED
489756 - 326584 --  Test statistic = 69998.5000, p-Value = 0.0000 REJECTED
489756 - 675201 --  Test statistic = 86723.5000, p-Value = 0.0000 REJECTED
489756 - 201436 --  Test statistic = 60158.0000, p-Value = 0.0000 REJECTED
361254 - 874521 --  Test statistic = 218106.0000, p-Value = 0.0241 REJECTED
361254 - 326584 --  Test statistic = 33158.5000, p-Value = 0.0000 REJECTED
361254 - 675201 --  Test statistic = 39586.0000, p-Value = 0.3249
361254 - 201436 --  Test statistic = 30006.0000, p-Value = 0.4866
874521 - 326584 --  Test statistic = 38748.0000, p-Value = 0.0000 REJECTED
874521 - 675201 --  Test statistic = 47522.0000, p-Value = 0.2752
874521 - 201436 --  Test statistic = 34006.0000, p-Value = 0.1478
326584 - 675201 --  Test statistic = 6963.5000, p-Value = 0.0001 REJECTED
326584 - 201436 --  Test statistic = 5301.0000, p-Value = 0.0005 REJECTED
675201 - 201436 --  Test statistic = 6121.0000, p-Value = 0.3185
"""
"""
Within following groups there is no significant statistical difference so then can be interpreted together
361254 - 675201 --  Test statistic = 39586.0000, p-Value = 0.3249
361254 - 201436 --  Test statistic = 30006.0000, p-Value = 0.4866
874521 - 675201 --  Test statistic = 47522.0000, p-Value = 0.2752
874521 - 201436 --  Test statistic = 34006.0000, p-Value = 0.1478
675201 - 201436 --  Test statistic = 6121.0000, p-Value = 0.3185

201436-361254-675201-874521
"""
df_cat.dtypes
df_cat.reset_index(inplace=True)
df_cat["cats"]=""
cats=[201436,361254,675201,874521]
for i in df_cat:
    if df_cat["category_id"].isin(cats):
        df_cat["cats"] = "A"
    elif loc.[df_cat["category_id"]] == 326584:
        df_cat["cats"] = "B"
    else:
        df_cat["cats"] = "C"


num_cols = [col for col in df.columns if df[col].dtype != "O"]
# H0: M1 = M2 (... There is no significant statistical difference between two groups.
# H1: M1 != M2 (...difference exists)
# p-value < 0.05 H0 REJECTED.
# p-value > 0.05 H0 CANNOT BE REJECTED.

# Assumption is  valid for control group.

test_stat, pvalue = shapiro(df_test["Return Rate"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value < 0.05  HO DENIED.
# p-value > 0.05 H0 Can not be Denied.
# Assumption is  valid for test group.


def normality_check(dataframe, column):
    test_stat, pvalue = shapiro(dataframe[column])
    print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
    if pvalue < 0.05:
        print("HO DENIED, Normality Assumption is not valid")
    else:
        print("HO Cannot be DENIED, Normality Assumption valid")

normality_check( df[df["category_id"]==201436], "price")


df[df["category_id"]==201436]



from scipy.stats import shapiro
test_stat, pvalue = shapiro(df[df["category_id"]==201436]["Return Rate"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



import seaborn as sns
sns.set_theme(style="whitegrid")

ax = sns.boxplot(x=df[df["category_id"]==201436]["price"])
plt.show()
ax = sns.boxplot(x="category_id", y="price", data=df[df["category_id"]==201436])
plt.show()





































