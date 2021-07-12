##############################################################
# PROJECT: Make CLTV Prediction using BGNBD ve GG Models.
##############################################################

##############################################################
# Task-1
##############################################################
# - Make 6 months prediction for 2010-2011 UK customers.
# - Discuss and interpret the results.
# - Highlight the precise and non accurate scores.
# - Watch out!! CLTV is expected not the 6 months expected sales.
#   Construct bgnbd & gamma models and
# - put 6 months for cltv prediction.


##############################################################
# 1. Data Preperation
##############################################################
pip install lifetimes

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.preprocessing import MinMaxScaler

def outlier_thresholds(dataframe, variable):
    quartile1=dataframe[variable].quantile(0.01)
    quartile3=dataframe[variable].quantile(0.99)
    interquantile_range=quartile3-quartile1
    up_limit=quartile3+1.5*interquantile_range
    low_limit=quartile1*1.5*interquantile_range
    return low_limit,up_limit

def replace_with_tresholds(dataframe, variable):
    low_limit, up_limit=outlier_thresholds(dataframe,variable)
    #dataframe.loc[(dataframe[variable]<low_limit), variable]=low_limit
    dataframe.loc[(dataframe[variable]>up_limit), variable]=up_limit


df_ = pd.read_excel("D:/Data Science/dsmlbc4/datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df=df_.copy()
df.head()


df.shape
df.head()
df.info()
df.describe().T
df.isnull().sum()


##############################################################
# 1. Data Preprocessing
##############################################################

#Dropping of null values
df.dropna(inplace=True)
df.isnull().sum()

#Elemination of Invoices starting with "C" ("Cancelled" or Returned
df=df[~df["Invoice"].str.contains("C", na=False)]
df=df[df["Quantity"]>0]
df.describe().T

#Selection of customers from UK
df=df[df["Country"]=="United Kingdom"]
df.head()

#Application of thresholds
replace_with_tresholds(df, "Quantity")
replace_with_tresholds(df,"Price")
df.describe().T

#TotalPrice Column
df["TotalPrice"]=df["Quantity"]*df["Price"]
df.head()

#Maxdate & today_date
df["InvoiceDate"].max()
today_date=dt.datetime(2011,12,11)

##############################################################
# Task-1
##############################################################
# - Make 6 months prediction for 2010-2011 UK customers.
# - Discuss and interpret the results.
# - Highlight the precise and non accurate scores.
# - Watch out!! CLTV is expected not the 6 months expected sales.
# - Construct bgnbd & gamma models and
# - Use 6 months for cltv prediction.

#############################################
# RFM Table
#############################################

rfm = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                     lambda date: (today_date - date.min()).days],
                                     'Invoice': lambda num: num.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()

rfm.columns=rfm.columns.droplevel(0)

## recency_cltv_p
rfm.columns=['recency_cltv_p', 'T', 'frequency', 'monetary']
rfm.head()

#"T" time that the customer spent in the system(from first transaction up to date)

## simplified monetary_avg
rfm["monetary"] = rfm["monetary"] / rfm["frequency"]

rfm.rename(columns={"monetary": "monetary_avg"}, inplace=True)

rfm.index=rfm.index.astype(int)
rfm.head()

#Setting daily values to weekly values (division by 7)
rfm["recency_weekly_p"]=rfm["recency_cltv_p"]/7
rfm["T_weekly"]=rfm["T"]/7


#Check
rfm = rfm[rfm["monetary_avg"] > 0]

## freq > 1
rfm = rfm[(rfm['frequency'] > 1)]
#Gamma-Gamma needs Frequency as int
rfm["frequency"] = rfm["frequency"].astype(int)

#Checking correlations(BGNBD suggests that there is no correlation between monetary_avg and recency)
rfm[['monetary_avg', 'recency_weekly_p']].corr()

rfm.head()


##############################################################
# 2. BG/NBD Model
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(rfm['frequency'],
        rfm['recency_weekly_p'],
        rfm['T_weekly'])

rfm_x=rfm.copy()
################################################################
# Expected number of sales in 6 months?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                        rfm['frequency'],
                                                        rfm['recency_weekly_p'],
                                                        rfm['T_weekly']).sort_values(ascending=False).head(10)


rfm["expected_number_of_purchases_6m"] = bgf.predict(4*6,
                                                        rfm['frequency'],
                                                        rfm['recency_weekly_p'],
                                                        rfm['T_weekly'])


rfm.head()

rfm.sort_values(by="expected_number_of_purchases_6m", ascending=False).head(20)

#total ecpected sale in 6 months
bgf.predict(4*6,
            rfm['frequency'],
            rfm['recency_weekly_p'],
            rfm['T_weekly']).sum()

################################################################
# Transaction plot
################################################################

plot_period_transactions(bgf)
plt.show()

##############################################################
# 3. GAMMA-GAMMA Model
##############################################################


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(rfm['frequency'], rfm['monetary_avg'])

ggf.conditional_expected_average_profit(rfm['frequency'],
                                        rfm['monetary_avg']).sort_values(ascending=False).head(10)


rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm['frequency'],
                                                                              rfm['monetary_avg'])


rfm.sort_values("expected_average_profit", ascending=False).head(20)


##############################################################
# 4. Calculation of CLTV using BG-NBD & GG models
##############################################################


cltv_6 = ggf.customer_lifetime_value(bgf,
                                     rfm['frequency'],
                                     rfm['recency_weekly_p'],
                                     rfm['T_weekly'],
                                     rfm['monetary_avg'],
                                     time=6,  # 6 months
                                     freq="W",  # T frequency
                                     discount_rate=0.01)

?ggf.customer_lifetime_value
cltv_6.head()

cltv_6.shape
cltv_6 = cltv_6.reset_index()
cltv_6.sort_values(by="clv", ascending=False).head(50)
cltv_6.rename(columns={"clv":"clv_6"}, inplace =True)
cltv_6.head()

rfm_cltv_final = rfm.merge(cltv_6, on="Customer ID", how="left")
rfm_cltv_final.head()

rfm_cltv_final.sort_values(by="clv_6", ascending=False).head(50)


rfm_cltv_final.describe().T






##############################################################
# TASK-2
##############################################################
# - Calculate 1-month and 12-months CLTV for 2010-2011 UK customers.
# - Analyze top 10 in 1-month and 12-months.
# - If there is a difference in the lists, what are the reasons?
#  Watch out!!! No need to construct the model from scratch.
#  CLTV can be calculated from existing  bgf & ggf
rfm1=rfm_x.copy()
rfm12=rfm_x.copy()
rfm1["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm1['frequency'],
                                                                              rfm1['monetary_avg'])

rfm1.head()

rfm1["expected_number_of_purchases_1m"] = bgf.predict(4*1,
                                                        rfm1['frequency'],
                                                        rfm1['recency_weekly_p'],
                                                        rfm1['T_weekly'])

cltv_1 = ggf.customer_lifetime_value(bgf,
                                     rfm1['frequency'],
                                     rfm1['recency_weekly_p'],
                                     rfm1['T_weekly'],
                                     rfm1['monetary_avg'],
                                     time=1,  # 1 month
                                     freq="W",  # T frequency.
                                     discount_rate=0.01)

cltv_1 = cltv_1.reset_index()
cltv_1.head()
cltv_1.rename(columns={"clv":"clv_1"}, inplace =True)
cltv_1.sort_values(by="clv_1", ascending=False).head(50)

rfm1= rfm1.merge(cltv_1, on="Customer ID", how="left")
rfm1.head()


rfm1.sort_values(by="clv_1", ascending=False).head(50)



rfm12["expected_number_of_purchases_12m"] = bgf.predict(4*12,
                                                        rfm12['frequency'],
                                                        rfm12['recency_weekly_p'],
                                                        rfm12['T_weekly'])
rfm12.head()
cltv_12 = ggf.customer_lifetime_value(bgf,
                                      rfm12['frequency'],
                                      rfm12['recency_weekly_p'],
                                      rfm12['T_weekly'],
                                      rfm12['monetary_avg'],
                                      time=12,  # 12 months
                                      freq="W",  # T frequency
                                      discount_rate=0.01)
?ggf.customer_lifetime_value
cltv_12.head()
cltv_12 = cltv_12.reset_index()
cltv_12.head()
cltv_12.rename(columns={"clv":"clv_12"}, inplace =True)
cltv_12.sort_values(by="clv_12", ascending=False).head(50)

rfm12 = rfm12.merge(cltv_12, on="Customer ID", how="left")
rfm12.head()


rfm1_12=rfm1.merge(rfm12, on="Customer ID", how="inner")
rfm1_12.head()
rfm1_12.drop(columns=["recency_cltv_p_y","T_y","frequency_y","monetary_avg_y",
                      "recency_weekly_p_y","T_weekly_y"], inplace=True)
rfm1_12.head()

rfm1.sort_values(by="clv_1", ascending=False).head(10)
rfm12.sort_values(by="clv_12", ascending=False).head(10)
rfm1_12.sort_values(by="clv_1", ascending=False).head(10)
rfm1_12.sort_values(by="clv_12", ascending=False).head(10)


rfm1_12.sort_values(by="expected_number_of_purchases_12m", ascending=False).head(10)




##############################################################
# TASK-3
##############################################################
# Divide the 2010-2011 UK customers into 3 segments according to 6-months CLTV
# add segment names in dataset ie. (C, B, A)
# Select the top 20% and print top_flag.  if in top 20% print 1, if not 0.

rfm_cltv_final.head()
rfm_cltv_final["Customer Segments"]=pd.qcut(rfm_cltv_final["clv_6"], 3, labels=["C", "B", "A"])
rfm_cltv_final=[[rfm_cltv_final["top_flag"]=1 if rfm_cltv_final[cltv_6]>]]
rfm_cltv_final["top_flag"]=pd.qcut(rfm_cltv_final["clv_6"],q=[0,0.8,1.], labels=[0,1])

rfm_cltv_final.sort_values("clv_6", ascending=False)



# Analyze 3 segments with respect to other variables.
rfm_cltv_final.describe().T
rfm_cltv_final.groupby("Customer Segments").agg({"clv_6": ["min", "mean", "max"],
                                                 "expected_average_profit": ["min", "mean", "max"],
                                                 "frequency":["min", "mean", "max"],
                                                 "T":["min", "mean", "max"],
                                                 "recency_cltv_p":["min", "mean", "max"]})
# Recommend 6 month actions for 3 segments.
rfm_cltv_final.groupby("Customer Segments")["recency_cltv_p", "T","frequency","expected_average_profit","expected_number_of_purchases_6m","clv_6"].describe()


