# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables.
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from dsmlbc4.helpers.eda import *
from dsmlbc4.helpers.data_prep import *

warnings.simplefilter (action='ignore', category=FutureWarning)
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Timer used as decorator to measure process time for each computation
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Display importance Export Feature Importance png
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :100].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    print(best_features)
    plt.figure(figsize=(15, 20))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

# Gets categorical columns
def get_categoric_columns(df):
    cols = df.select_dtypes(include=['object', 'category']).columns
    return cols


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, ohe_columns=None, nan_as_category=True, drop_first=False):
    # if there exists nan values in categorical variables  to store thiese values in another variable
    # nan_as_category should be set as true
    original_columns = list(df.columns)
    if ohe_columns:
        df = pd.get_dummies(df, columns=ohe_columns, drop_first=drop_first)
    else:
        categorical_columns = [col for col in df.columns if
                               df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_columns,
                            dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows=None, nan_as_category=False):
    df = pd.read_csv('D:/Data Science/dsmlbc4/datasets/home_credit/application_train.csv', nrows=num_rows)
    test_df = pd.read_csv('D:/Data Science/dsmlbc4/datasets/home_credit/application_test.csv', nrows=num_rows)

    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))

    df = df.append(test_df).reset_index()

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # During application age of the customer noted as days passed. To  transform the age as normal:
    df["NEW_APP_AGE"] = round(df["DAYS_BIRTH"] * -1 / 365)
    df.drop("DAYS_BIRTH", axis=1, inplace=True)

    df.head()

    # Drop out variables having more than 40% of null values:

    cols_dropped = [col for col in df.columns if
                    (col.endswith("AVG")) | (col.endswith("MODE")) | (col.endswith("MEDI"))]

    df.drop(cols_dropped, axis=1, inplace=True)


    # Aggregating Flag Document  variables:

    flag_doc_cols = [col for col in df.columns if col.startswith("FLAG_DOC")]

    df["NEW_FLAG_DOC"] = df[flag_doc_cols].sum(axis=1) / len(flag_doc_cols)
    df.drop(flag_doc_cols, axis=1, inplace=True)

    # EDA on DAYS_ variables:
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    days_cols = [col for col in df.columns if col.startswith("DAYS_")]

    # Forming new day variables using DAYS_:
    for col in days_cols:
        df[col + "_LABEL"] = pd.qcut(df[col], 5, labels=[1, 2, 3, 4, 5]).to_numpy()

    label_days_cols = [col for col in df.columns if col.endswith("LABEL")]

    df[label_days_cols].corrwith(df["TARGET"])

    label_cols = df[label_days_cols].corr()

    df.head()

    df[["DAYS_LAST_PHONE_CHANGE_LABEL", "TARGET"]].corr()


    # Feature Enginnering:

    # Ratio of Days Employed per Age:
    df['DAYS_EMPLOYED_PERC'] = (df['DAYS_EMPLOYED'] * (-1)) / (df["NEW_APP_AGE"] * 365)

    # Income per Loan:
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

    # Income per family member:
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # Anuual Loan Payment per Income:
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    # Annaul Payment per Loan:
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # Loan per Goods Price:
    df["NEW_GOODSPRICE_CREDIT"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]

    # Employment Status:
    df["WORK_NOTWORK"] = df["DAYS_EMPLOYED"]

    # Unemployed
    df.loc[(df["WORK_NOTWORK"] == 0), "WORK_NOTWORK"] = 0
    # Employed
    df.loc[(df["WORK_NOTWORK"] != 0), "WORK_NOTWORK"] = 1


    ## Credit checks in recent time (1 year) and older time:

    # recent credit checks
    df.loc[(df["AMT_REQ_CREDIT_BUREAU_YEAR"] < 1), "NEW_REQ"] = "yakın zaman"

    # older credit checks
    df.loc[(df["AMT_REQ_CREDIT_BUREAU_YEAR"] >= 1), "NEW_REQ"] = "uzak zaman"

    # no credit checks
    df.loc[(pd.isna(df["NEW_REQ"])), "NEW_REQ"] = "soruşturma yok"

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, None, nan_as_category)

    # New features based on External sources
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3
    np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
        feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
        df[feature_name] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    del test_df
    gc.collect()
    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('D:/Data Science/dsmlbc4/datasets/home_credit/bureau.csv', nrows=num_rows)
    bb = pd.read_csv('D:/Data Science/dsmlbc4/datasets/home_credit/bureau_balance.csv', nrows=num_rows)

    # Status Feature
    status_val = ['C', 'X', 'O']
    bb["NEW_STATUS"] = bb["STATUS"].apply(lambda x: 0 if x in status_val or 0 else 1)

    # One hot encoding of categorical variables in both dataframes
    bb, bb_cat = one_hot_encoder(bb)
    bureau, bureau_cat = one_hot_encoder(bureau)

    # calculation of min,max & size of months balance, mean & sum of one hot encoded status ,
    # By "mean" the difference between payment rates is made
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean', "sum"]

    # Deduplication on SK_ID_BUREAU
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)

    # Two layered variable names deduplicated (first latyer  months balance and status, other min,max,size and  mean, sum')
    # Variable names changed dynamically
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])

    # Table merged
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')

    # SK_ID_BUREAU dropped
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)

    # Temp df's deleted from RAM
    del bb, bb_agg
    gc.collect()

    # CREDIT_CURRENCY_FX (foreign exchange)
    # How many loans are taken other than th dominant Currenyc1
    bureau["CREDIT_CURRENCY_FX"] = bureau["CREDIT_CURRENCY_currency 2"] + bureau["CREDIT_CURRENCY_currency 3"] \
                                   + bureau["CREDIT_CURRENCY_currency 4"]

    # Possible NPL credits:
    # Sum and mean of loans having status of 4&5 (not paid more than 90 days)
    # Calculation of non-performing loans of each customer using the values above
    bureau["STATUS_NPL_SUM"] = bureau.fillna(0)["STATUS_4_SUM"] + bureau.fillna(0)["STATUS_5_SUM"]
    bureau["STATUS_NPL_MEAN"] = bureau.fillna(0)["STATUS_4_MEAN"] + bureau.fillna(0)["STATUS_5_MEAN"]

    # Posibble PL credits:
    # Sum and mean of loans having status of 1,2 vand 3 olan (not paid less than 90 days)

    bureau["STATUS_PL_SUM"] = bureau.fillna(0)["STATUS_3_SUM"] + bureau.fillna(0)["STATUS_2_SUM"] \
                              + bureau.fillna(0)["STATUS_1_SUM"]
    bureau["STATUS_PL_MEAN"] = bureau.fillna(0)["STATUS_3_MEAN"] + bureau.fillna(0)["STATUS_2_MEAN"] \
                               + bureau.fillna(0)["STATUS_1_MEAN"]

    # Rate of loan payment to total possible loan amount
    bureau['LOAN_RATE'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']

    # Aggregation operations for numrical variables of both tables
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'STATUS_PL_SUM': ["sum"],
        'STATUS_PL_MEAN': ['mean'],
        'STATUS_NPL_SUM': ["sum"],
        'STATUS_NPL_MEAN': ['mean'],
        'CREDIT_CURRENCY_FX': ["sum"],
        'LOAN_RATE': ['min', 'max', 'mean']}

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}

    # Mean of all categorical variables
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']

    # Adding  "mean" to varable manes
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})

    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    bureau.head()

    # Bureau: Active credits - using only numerical aggregations
    # Adding active loans to another df
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]

    # Calculation of  max, mean, sum numerical values
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.head()

    # Fixing multi index problem, adding  "active" on new variable names
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    active_agg.head()

    # Merging active loans with bureau
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')

    # temp df'ler gecici bellekten siliniyor
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations

    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')

    # Temp df's deleted from RAM
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg



# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('D:/Data Science/dsmlbc4/datasets/home_credit/POS_CASH_balance.csv',  nrows=num_rows)

    # Separation Deferred loans from Default loans
    pos["SK_DPD_NOT_IMP"] = pos["SK_DPD"] - pos["SK_DPD_DEF"]
    pos.drop("SK_DPD", axis=1, inplace=True)

    # Dividing into categories with respect to DPD spans(days))
    pos.loc[pos["SK_DPD_NOT_IMP"] == 0, "DPD_NOT_IMP_CAT"] = "0"
    pos.loc[pos["SK_DPD_NOT_IMP"] >= 60, "DPD_NOT_IMP_CAT"] = "60+"
    pos.loc[(pos["SK_DPD_NOT_IMP"] >= 30) & (pos["SK_DPD_NOT_IMP"] < 60), "DPD_NOT_IMP_CAT"] = "30_59"
    pos.loc[(pos["SK_DPD_NOT_IMP"] > 0) & (pos["SK_DPD_NOT_IMP"] < 30), "DPD_NOT_IMP_CAT"] = "1_29"

    pos.loc[pos["SK_DPD_DEF"] == 0, "DPD_DEF_CAT"] = "0"
    pos.loc[pos["SK_DPD_DEF"] >= 60, "DPD_DEF_CAT"] = "60+"
    pos.loc[(pos["SK_DPD_DEF"] >= 30) & (pos["SK_DPD_DEF"] < 60), "DPD_DEF_CAT"] = "30_59"
    pos.loc[(pos["SK_DPD_DEF"] > 0) & (pos["SK_DPD_DEF"] < 30), "DPD_DEF_CAT"] = "1_29"

    # Deferral count per loan
    pos.loc[pos["SK_DPD_DEF"] > 0, "COUNT_SK_DPD_DEF"] = 1
    pos.loc[pos["SK_DPD_DEF"] == 0, "COUNT_SK_DPD_DEF"] = 0
    pos.loc[pos["SK_DPD_NOT_IMP"] > 0, "COUNT_SK_DPD_NOT_IMP"] = 1
    pos.loc[pos["SK_DPD_NOT_IMP"] == 0, "COUNT_SK_DPD_NOT_IMP"] = 0



    # One-Hot Encoding
    #pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    ohe_df = pd.get_dummies (pos, dummy_na=nan_as_category)

    # Aggregation Operations
    agg_dict = {}
    for col in ohe_df.columns:
        if (ohe_df[col].nunique() == 2) & (col.startswith("NAME")):
            agg_dict[col] = ["sum"]
        elif (ohe_df[col].nunique() == 2) & (col.startswith("DPD")):
            agg_dict[col] = ["mean"]
        elif col == "SK_ID_PREV":
            agg_dict[col] = ["nunique", "count"]
        elif col.startswith("COUNT"):
            agg_dict[col] = ["mean"]
        elif (col != "SK_ID_CURR") & (ohe_df[col].nunique() > 2):
            agg_dict[col] = ["max", "mean"]

    pos_agg = ohe_df.groupby("SK_ID_CURR").agg(agg_dict)

    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # Dropped because having the same value as POS_CNT_INSTALMENT
    pos_agg.drop("POS_CNT_INSTALMENT_FUTURE_MAX", axis=1, inplace=True)

    # Rate of Active Loans per Completed Loans.
    pos_agg["POS_NAME_CONTRACT_STATUS_Active_SUM"] = pos_agg["POS_SK_ID_PREV_NUNIQUE"] - pos_agg[
        "POS_NAME_CONTRACT_STATUS_Completed_SUM"]

    # Rate of Completed & Active loans over total loans.
    name_contract_cols = ["POS_NAME_CONTRACT_STATUS_Completed_SUM", "POS_NAME_CONTRACT_STATUS_Active_SUM"]

    for col in name_contract_cols:
        pos_agg[col] = pos_agg[col] / pos_agg["POS_SK_ID_PREV_NUNIQUE"]

    # Assigment of new variable names.
    pos_agg.rename(columns={"POS_NAME_CONTRACT_STATUS_Completed_SUM": "POS_NAME_CONTRACT_STATUS_Completed_RATIO",
                               "POS_NAME_CONTRACT_STATUS_Active_SUM": "POS_NAME_CONTRACT_STATUS_Active_RATIO"},
                      inplace=True)

    # 1010 observations having value more than 1 will be assigned as 1
    pos_agg.loc[
        pos_agg["POS_NAME_CONTRACT_STATUS_Completed_RATIO"] > 1, "POS_NAME_CONTRACT_STATUS_Completed_RATIO"] = 1

    # 1010 observations having value less than 1 will be assigned as 0.

    pos_agg.loc[pos_agg["POS_NAME_CONTRACT_STATUS_Active_RATIO"] < 0, "POS_NAME_CONTRACT_STATUS_Active_RATIO"] = 0

    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None):
    cc = pd.read_csv('D:/Data Science/dsmlbc4/datasets/home_credit/credit_card_balance.csv', nrows=num_rows)

    # ATM withdrawals over withdrawal count
    cc["NEW_ATM_CURRENT"] = cc["AMT_DRAWINGS_ATM_CURRENT"] / \
        cc["CNT_DRAWINGS_ATM_CURRENT"]

    # POS  operations over POS  operation count
    cc["NEW_POS_CURRENT"] = cc["AMT_DRAWINGS_POS_CURRENT"] / \
        cc["CNT_DRAWINGS_POS_CURRENT"]

    # Total  operation over Tottal operation count
    cc["NEW_DRAWINGS_CURRENT"] = cc["AMT_DRAWINGS_CURRENT"] / \
        cc["CNT_DRAWINGS_CURRENT"]

    # Rate of ATM operation to total operations
    cc["NEW_ATM_RATE"] = cc["AMT_DRAWINGS_ATM_CURRENT"] / \
        cc["AMT_DRAWINGS_CURRENT"]

    # Rate of POS operation to total operations
    cc["NEW_POS_RATE"] = cc["AMT_DRAWINGS_POS_CURRENT"] / \
        cc["AMT_DRAWINGS_CURRENT"]

    # Rate of Minimum Installment to monthly payment
    cc["NEW_PAYMENT_RATE"] = cc["AMT_INST_MIN_REGULARITY"] / \
        cc["AMT_PAYMENT_CURRENT"]

    # Rate of Total Payment to Total Receivable
    cc["NEW_TOTAL_PAYMENT_RATE"] = cc["AMT_PAYMENT_TOTAL_CURRENT"] / \
        cc["AMT_TOTAL_RECEIVABLE"]

    # Separation Deferred loans from Default loans
    cc["SK_DPD_N_IMP"] = cc["SK_DPD"] - cc["SK_DPD_DEF"]

    # Categorization according to DPD
    cc.loc[cc["SK_DPD_N_IMP"] == 0, "DPD_N_IMP_CAT"] = "0"
    cc.loc[cc["SK_DPD_N_IMP"] >= 60, "DPD_N_IMP_CAT"] = "60+"
    cc.loc[(cc["SK_DPD_N_IMP"] >= 30) & (
                cc["SK_DPD_N_IMP"] < 60), "DPD_N_IMP_CAT"] = "30_59"
    cc.loc[(cc["SK_DPD_N_IMP"] > 0) & (
                cc["SK_DPD_N_IMP"] < 30), "DPD_N_IMP_CAT"] = "1_29"

    cc.loc[cc["SK_DPD_DEF"] == 0, "DPD_DEF_CATE"] = "0"
    cc.loc[cc["SK_DPD_DEF"] >= 60, "DPD_DEF_CATE"] = "60+"
    cc.loc[(cc["SK_DPD_DEF"] >= 30) & (
                cc["SK_DPD_DEF"] < 60), "DPD_DEF_CATE"] = "30_59"
    cc.loc[(cc["SK_DPD_DEF"] > 0) & (
                cc["SK_DPD_DEF"] < 30), "DPD_DEF_CATE"] = "1_29"

    # Deferrals per operation
    cc.loc[cc["SK_DPD_DEF"] > 0, "CNT_SK_DPD_DEF"] = 1
    cc.loc[cc["SK_DPD_DEF"] == 0, "CNT_SK_DPD_DEF"] = 0
    cc.loc[cc["SK_DPD_N_IMP"] > 0, "CNT_SK_DPD_N_IMP"] = 1
    cc.loc[cc["SK_DPD_N_IMP"] == 0, "CNT_SK_DPD_N_IMP"] = 0

    delete_list = ["AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_OTHER_CURRENT",
                   "AMT_DRAWINGS_POS_CURRENT", "AMT_INST_MIN_REGULARITY",
                   "AMT_PAYMENT_TOTAL_CURRENT", "AMT_RECEIVABLE_PRINCIPAL",
                   "AMT_RECEIVABLE", "CNT_DRAWINGS_ATM_CURRENT",
                   "CNT_DRAWINGS_OTHER_CURRENT", "CNT_DRAWINGS_POS_CURRENT",
                   "SK_DPD", 'SK_ID_PREV']
    cc.drop(delete_list, inplace=True, axis=1)

    cc, cat_cols = one_hot_encoder(cc)

    # General aggregations
    cc_agg = cc.groupby('SK_ID_CURR').agg(
        ['min', 'max', 'mean', 'sum', 'count'])
    cc_agg.columns = pd.Index(
        ['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    del cc
    gc.collect()
    return cc_agg


# Preprocess previous_applications.csv
def previous_applications(num_rows=None):
    df = pd.read_csv('D:/Data Science/dsmlbc4/datasets/home_credit/previous_application.csv', nrows=num_rows)

    drop_list = ['WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
                 'FLAG_LAST_APPL_PER_CONTRACT',
                 'NFLAG_LAST_APPL_IN_DAY',
                 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
                 'NAME_PAYMENT_TYPE', 'NAME_TYPE_SUITE',
                 'NAME_CASH_LOAN_PURPOSE', 'NAME_GOODS_CATEGORY',
                 'CHANNEL_TYPE', 'SELLERPLACE_AREA', 'NAME_SELLER_INDUSTRY',
                 'NAME_PORTFOLIO', 'NAME_CONTRACT_TYPE', 'NAME_PRODUCT_TYPE',
                 'NAME_YIELD_GROUP', 'DAYS_DECISION'
                 ]

    for col in drop_list:
        df.drop(col, axis=1, inplace=True)

    df["NEW_CREDIT_COUNT"] = 1

    # Comparison of loans applied and loans granted
    df.loc[(df["AMT_APPLICATION"]) > df["AMT_CREDIT"],
           "NEW_APP_CRED_COND"] = "LOW"
    df.loc[(df["AMT_APPLICATION"]) == df["AMT_CREDIT"],
           "NEW_APP_CRED_COND"] = "NORMAL"
    df.loc[(df["AMT_APPLICATION"]) < df["AMT_CREDIT"],
           "NEW_APP_CRED_COND"] = "HIGH"

    # Rate of application amount to loan
    df["NEW_CREDIT_RATIO"] = df["AMT_APPLICATION"] / df["AMT_CREDIT"]

    # RAte of price of the goods to application amount
    df["NEW_GOODS_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_APPLICATION"]

    # Total payment amount
    df["NEW_TOTAL_PAYMENT"] = df["CNT_PAYMENT"] * df["AMT_ANNUITY"]

    # Rate of total payments to total loan
    df["NEW_TOTAL_PAYMENT_TO_CREDIT_RATIO"] = df["NEW_TOTAL_PAYMENT"] / \
        df["AMT_CREDIT"]

    # Simple Interest Ratewith Light GBM
    df['NEW_SIMPLE_INTERESTS'] = (df['NEW_TOTAL_PAYMENT'] /
                                  df['AMT_CREDIT'] - 1) / df['CNT_PAYMENT']

    df.loc[(df["NAME_CONTRACT_STATUS"]) != "Refused",
           "NEW_CODE_REJECT_REASON"] = "NotImp"
    df.loc[(df["NAME_CONTRACT_STATUS"]) == "Refused",
           "NEW_CODE_REJECT_REASON"] = df["CODE_REJECT_REASON"]
    df.loc[(df["NAME_CONTRACT_STATUS"]) == "Unused offer",
           "NAME_CONTRACT_STATUS"] = "UnusedOffer"

    df['PRODUCT_COMBINATION'] = df['PRODUCT_COMBINATION'].astype(str)

    df['PRODUCT_COMBINATION'] = [row.replace(":", "") for row in
                                 df['PRODUCT_COMBINATION']]
    df['PRODUCT_COMBINATION'] = [row.replace("-", "_") for row in
                                 df['PRODUCT_COMBINATION']]
    df['PRODUCT_COMBINATION'] = [row.replace(" ", "_") for row in
                                 df['PRODUCT_COMBINATION']]

    df.drop("CODE_REJECT_REASON", axis=1, inplace=True)

    # Days 365.243 values -> nan
    df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # One-Hot Encoding
    ohe_cols = [col for col in df.columns if 20 >= len(
        df[col].unique()) > 2 and col not in "NFLAG_INSURED_ON_APPROVAL"]
    df, new_columns = one_hot_encoder(df, ohe_cols)

    df_column1 = ['SK_ID_CURR', 'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT',
                  'AMT_DOWN_PAYMENT',
                  'AMT_GOODS_PRICE', 'RATE_DOWN_PAYMENT',
                  'CNT_PAYMENT', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
                  'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE',
                  'DAYS_TERMINATION', 'NEW_CREDIT_RATIO', 'NEW_GOODS_RATIO',
                  'NEW_TOTAL_PAYMENT', 'NEW_SIMPLE_INTERESTS']

    df_column2 = ['SK_ID_CURR', 'NEW_CREDIT_COUNT', 'NFLAG_INSURED_ON_APPROVAL',
                  'NEW_TOTAL_PAYMENT_TO_CREDIT_RATIO',
                  'NEW_SIMPLE_INTERESTS',
                  'NAME_CONTRACT_STATUS_Approved',
                  'NAME_CONTRACT_STATUS_Canceled',
                  'NAME_CONTRACT_STATUS_Refused',
                  'NAME_CONTRACT_STATUS_UnusedOffer', 'NAME_CLIENT_TYPE_New',
                  'NAME_CLIENT_TYPE_Refreshed',
                  'NAME_CLIENT_TYPE_Repeater', 'NAME_CLIENT_TYPE_XNA',
                  'NEW_APP_CRED_COND_HIGH',
                  'NEW_APP_CRED_COND_LOW', 'NEW_APP_CRED_COND_NORMAL',
                  'NEW_CODE_REJECT_REASON_HC',
                  'NEW_CODE_REJECT_REASON_LIMIT',
                  'NEW_CODE_REJECT_REASON_NotImp', 'NEW_CODE_REJECT_REASON_SCO',
                  'NEW_CODE_REJECT_REASON_SCOFR',
                  'NEW_CODE_REJECT_REASON_SYSTEM',
                  'NEW_CODE_REJECT_REASON_VERIF',
                  'NEW_CODE_REJECT_REASON_XAP', 'NEW_CODE_REJECT_REASON_XNA',
                  'PRODUCT_COMBINATION_Card_Street',
                  'PRODUCT_COMBINATION_Card_X_Sell',
                  'PRODUCT_COMBINATION_Cash',
                  'PRODUCT_COMBINATION_Cash_Street_high',
                  'PRODUCT_COMBINATION_Cash_Street_low',
                  'PRODUCT_COMBINATION_Cash_Street_middle',
                  'PRODUCT_COMBINATION_Cash_X_Sell_high',
                  'PRODUCT_COMBINATION_Cash_X_Sell_low',
                  'PRODUCT_COMBINATION_Cash_X_Sell_middle',
                  'PRODUCT_COMBINATION_POS_household_with_interest',
                  'PRODUCT_COMBINATION_POS_household_without_interest',
                  'PRODUCT_COMBINATION_POS_industry_with_interest',
                  'PRODUCT_COMBINATION_POS_industry_without_interest',
                  'PRODUCT_COMBINATION_POS_mobile_with_interest',
                  'PRODUCT_COMBINATION_POS_mobile_without_interest',
                  'PRODUCT_COMBINATION_POS_other_with_interest',
                  'PRODUCT_COMBINATION_POS_others_without_interest',
                  'PRODUCT_COMBINATION_nan']
    df.drop(['SK_ID_PREV'], axis=1, inplace=True)
    df_agg1 = df[df_column1].groupby('SK_ID_CURR').agg(['max', 'min', 'mean'])
    df_agg2 = df[df_column2].groupby('SK_ID_CURR').agg(['sum'])
    df_agg = df_agg1.join(df_agg2, how='inner', on='SK_ID_CURR')


    # Multi-index fix
    df_agg.columns = pd.Index([ind[0] + '_' + ind[1].upper()
                               for ind in df_agg.columns.tolist()])

    del df, df_agg1, df_agg2
    gc.collect()
    return df_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows=None):
    """
    Performs feature engineering and aggregation operations on given csv.

    Parameters
    ----------
    num_rows
            Number of rows of file to read. Useful for reading pieces of large
            files.

    Returns
    -------
    pandas.core.frame.DataFrame
            Aggregated dataframe of given dataframe.
    """

    ins = pd.read_csv('D:/Data Science/dsmlbc4/datasets/home_credit/installments_payments.csv', nrows=num_rows)
    # Feature Engineering

    # 1 represents late payment, 0 represents payment on time or early
    ins['NEW_DAYS_PAID_LATER'] = (ins['DAYS_INSTALMENT'] -
                                  ins['DAYS_ENTRY_PAYMENT']).\
        map(lambda x: 1 if x < 0 else 0)

    # 1 represents short-change, 0 represents overpayment or full payment
    ins['NEW_AMT_INSTALMENT_SHORT'] = (
                ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']). \
        map(lambda x: 1 if x < 0 else 0)

    # 1 represents customer made short-change and late payment on instalment
    ins['NEW_SHORT_AND_LATE_PAYMENT'] = (
            (ins['NEW_DAYS_PAID_LATER'] == 1) &
            (ins['NEW_AMT_INSTALMENT_SHORT'] == 1)).map(lambda x: 1 if x else 0)

    aggregations = {'NUM_INSTALMENT_VERSION': ['nunique'],
                    'NUM_INSTALMENT_NUMBER': 'max',
                    'DAYS_INSTALMENT': ['min', 'max'],
                    'DAYS_ENTRY_PAYMENT': ['min', 'max'],
                    'AMT_INSTALMENT': ['min', 'max', 'sum', 'mean'],
                    'AMT_PAYMENT': ['min', 'max', 'sum', 'mean'],
                    'NEW_DAYS_PAID_LATER': 'sum',
                    'NEW_AMT_INSTALMENT_SHORT': 'sum',
                    'NEW_SHORT_AND_LATE_PAYMENT': 'sum'}

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)

    # Multi-index fix
    ins_agg.columns = pd.Index([ind[0] + '_' + ind[1].upper()
                                for ind in ins_agg.columns.tolist()])

    # drop operation
    ins_agg.drop(['DAYS_INSTALMENT_MIN',
                  'DAYS_INSTALMENT_MAX',
                  'DAYS_ENTRY_PAYMENT_MIN',
                  'DAYS_ENTRY_PAYMENT_MAX'], axis=1, inplace=True)

    # Payment percentage of customer, actual payment over expected payment
    ins_agg['NEW_PAYMENT_PERC'] = ins_agg['AMT_PAYMENT_SUM'] / \
                                  ins_agg['AMT_INSTALMENT_SUM']

    # Payment difference of customer, difference between actual payment and
    # expected payment
    ins_agg['NEW_PAYMENT_DIFF'] = ins_agg['AMT_INSTALMENT_SUM'] - \
                                  ins_agg['AMT_PAYMENT_SUM']

    # Installment count after aggregation
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()

    del ins
    gc.collect()
    return ins_agg


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    # Divide in training/validation and test data
    import re

    train_df = df[df['TARGET'].notnull()]
    train_df = train_df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    test_df = df[df['TARGET'].isnull()]
    test_df = test_df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization

        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=200, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)
    display_importances(feature_importance_df)
    return feature_importance_df


def main(debug=False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds=10, stratified=False, debug=debug)


if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()
