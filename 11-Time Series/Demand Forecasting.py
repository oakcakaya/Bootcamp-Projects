#####################################################
# Store Item Demand Forecasting
#####################################################

# 3-months item level sale forecasting for different stores.
# 10 separate stores and 50 unique items in 5-years dataset.
# forecast for next 3 months for each store.
# hierarchical forecast ya da or non-hierarchical


#####################################################
# Libraries
#####################################################

import time
import numpy as np
import pandas as pd
#pip install lightgbm
import lightgbm as lgb
import warnings
from dsmlbc4.helpers.eda import *
from dsmlbc4.helpers.data_prep import *

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


#####################################################
# Loading the data
#####################################################

train = pd.read_csv('dsmlbc4/datasets/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('dsmlbc4/datasets/demand_forecasting/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('dsmlbc4/datasets/demand_forecasting/sample_submission.csv')
df = pd.concat([train, test], sort=False)

#####################################################
# EDA
#####################################################

df["date"].min(), df["date"].max()
# Min: 2013-01-01 Max:2018-03-31

check_df(train)
# (913000, 4)

check_df(test)
# (45000, 4)

check_df(sample_sub)

check_outlier(df, "sales")
missing_values_table(df)

# Distribution of sales
df[["sales"]].describe().T

# Total number of stores
df[["store"]].nunique()

# Number of unique items
df[["item"]].nunique()

# Total number of unique item in each store
df.groupby(["store"])["item"].nunique()

# Sales in each store
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# sale statistics for store&item
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

#####################################################
# FEATURE ENGINEERING
#####################################################

#####################################################
# Date Features
#####################################################

df.head()
df.shape

# deriving date variables from "date"
# df['month'] = df.date.dt.month : month
# df['day_of_month'] = df.date.dt.day : day
# df['day_of_year'] = df.date.dt.dayofyear : day of the year
# df['week_of_year'] = df.date.dt.weekofyear : weak of the year
# f['day_of_week'] = df.date.dt.dayofweek + 1 : first day of the year is tuesday
# df['year'] = df.date.dt.year : year
# df["is_wknd"] = df.date.dt.weekday // 4 : weekend days (friday-sunday) monday==0
# df['is_month_start'] = df.date.dt.is_month_start.astype(int)
# df['is_month_end'] = df.date.dt.is_month_end.astype(int)

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek + 1
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)
df.head(20)


df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})

#####################################################
# Random Noise
#####################################################
# To tackle overfitting, random noise is added to the data
# To disrupt the pattern , random noise is added
# mean 0, standard deviation:1.6, 1.6 comes from kaggle notebook
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


#####################################################
# Lag/Shifted Features
#####################################################

# Order in the dataset may not be present, because of the recording system. To avoid confusion, values sorted as follows
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

check_df(df)
df["sales"].head(10)

# previous sales data as an array
df["sales"].shift(1).values[0:10]


# adding previous  5 sales values to the dataframe
pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})


df.groupby(["store", "item"])['sales'].head()

# calculating lag in more functional way by using transform
df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1))


# deriving lag features
def lag_features(dataframe, lags):
    dataframe = dataframe.copy()
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

# lag values starting from 3 & 6 months and increasing weekly
df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

df.head(100)

df[df["sales"].isnull()]

df[df["date"] == "2017-10-02"]

# pd.to_datetime("2018-01-01") - pd.DateOffset(91)

#####################################################
# Rolling Mean Features
#####################################################

# Moving Average
# window size: number of previous values
df["sales"].head(10)
df["sales"].rolling(window=2).mean().values[0:10]
df["sales"].rolling(window=3).mean().values[0:10]
df["sales"].rolling(window=5).mean().values[0:10]

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})

# To exclude self value from the moving average, shift(1) is added,
# By this way only the effect of previous values are taken into account. Trend can be respresented clearly.

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})

def roll_mean_features(dataframe, windows):
    dataframe = dataframe.copy()
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(dataframe)
    return dataframe

df = roll_mean_features(df, [365, 546])

df.head()


#####################################################
# Exponentially Weighted Mean Features
#####################################################

# shift eliminates the trap
# alpha= parameter that decides the weight of previous values. alpha goes up weight of the most recent value increases.
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm01": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})



def ewm_features(dataframe, alphas, lags):
    dataframe = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales']. \
                    transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

check_df(df)
df.columns


#####################################################
# One-Hot Encoding
#####################################################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])
check_df(df)

#####################################################
# Converting sales to log(1+sales)
#####################################################
# since log(0) is undefined, to eliminate this issue
# log conversion makes calculation easier

df['sales'] = np.log1p(df["sales"].values)

#####################################################
# Custom Cost Function
#####################################################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


#####################################################
# MODEL VALIDATION
#####################################################

# Light GBM: optimization should be considered two-fold.
# hyper parameters optimization, iteration will be fixed
# iteration times, hyper parameter will be fixed
# best iteration count wil lbe optimized

#####################################################
# Time-Based Validation Sets
#####################################################

# Kaggle test set values to be predicted: first 3 months of 2018.

test["date"].min(), test["date"].max()
train["date"].min(), train["date"].max()

# train set is from start to end of 2016 ( start of 2017)
train = df.loc[(df["date"] < "2017-01-01"), :]
train["date"].min(), train["date"].max()

# Validation set: first 3 months of 2017.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

df.columns

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape


#####################################################
# LightGBM Model
#####################################################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 10000,
              'early_stopping_rounds': 200,
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error
# learning_rate: shrinkage_rate, eta
# num_boost_round: n_estimators, number of boosting iterations.
# nthread: num_thread, nthread, nthreads, n_jobs
# early stopping rounds: if the error does not decrease, stops iteration. Chekcs every 200 iterations,
#   This decreases train time validation


# lgm asl parameter in it own way.
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)


y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
smape(np.expm1(y_pred_val), np.expm1(Y_val))



##########################################
# Feature Importance
##########################################

def plot_lgb_importances(model, plot=False, num=10):
    from matplotlib import pyplot as plt
    import seaborn as sns
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30)

"""
split: shows the number of split tha a feature encounter in the model 
gain: entropy change before and after split. entropy is complexity and if you decrease complexity you gain info.
 feature  split       gain
17           sales_roll_mean_546   7076  53.996709
13                 sales_lag_364   5897  13.220986
16           sales_roll_mean_365   5304   9.790065
60    sales_ewm_alpha_05_lag_365   1775   4.828710
18    sales_ewm_alpha_095_lag_91   1075   2.212179
1                    day_of_year   5205   2.116706
54     sales_ewm_alpha_05_lag_91   1090   1.864621
3                        is_wknd    796   1.190595
123                day_of_week_1    661   1.171935
141                     month_12    986   1.169769
27     sales_ewm_alpha_09_lag_91    649   1.050540
36     sales_ewm_alpha_08_lag_91    601   0.923674
2                   week_of_year   1543   0.911584"""


plot_lgb_importances(model, plot=True, num=30)



lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()


##########################################
# Final Model
##########################################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}


# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

# Create submission
submission_df = test.loc[:, ['id', 'sales']]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)
submission_df.to_csv('submission.csv', index=False)
