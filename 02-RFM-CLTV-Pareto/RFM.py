############################################
# PROJE: RFM ile Müşteri Segmentasyonu
############################################
# A commercial website wants to divide their customers into segments and develop marketing
# strategies according to these segments.
#  Apply rfm analysis on "Year 2010-2011" sheet of "online_retail_II.xlsx" dataset

# Download Dataset from following urls.alan "online_retail_II.xlsx"
# https://www.kaggle.com/nathaniel/uci-online-retail-ii-data-set or
# https://archive.ics.uci.edu/ml/machine-learning-databases/00502/



import numpy as np
import pandas as pd
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
#pd.set_option('Display.float_format', lambda x:'%.5f'%x)

df_ = pd.read_excel("D:/Data Science/dsmlbc4/datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df=df_.copy()

df.head()

#############################################
# TASK 1: Do same application on dataset as like in course
############################################

#Missing Values
df.isnull().any()
df.isnull().sum()

# Number of unique items?
df["StockCode"].nunique()

# Total quantity for each item?
df["StockCode"].value_counts().head()

# Item ordered most?
df.groupby("StockCode").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head

# Total number of invoices?
df["Invoice"].nunique()

# Reform df by dropping out "Cancelled" items
df=df[~df["Invoice"].str.contains("C", na=False)]
df

# Total amount earned for each invoice? ,
# (New variable needs to be formed by multiplication of two variables
df["TotalPrice"]=df["Quantity"]*df["Price"]
df


# Most expensive item?

df.sort_values("Price",ascending=False).head()

# Number of order from each country?
df["Country"].value_counts()

# Revenue per country?
df.groupby("Country").agg({"TotalPrice": "sum"}).sort_values("TotalPrice", ascending=False).head()



############################################
# TASK 2: Divide the customers into segment then Interpret selected 3 segments according to action decisions
# and segment properties (mean RFM values)
############################################

###############################################################
# Data Preparation
###############################################################

df.isnull().sum()
df.dropna(inplace=True)

df.describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

###############################################################
# Calculating RFM Metrics
###############################################################

# Recency, Frequency, Monetary

# Recency : Time passed since the last transaction of the customer
# In other words “Time passed since the last contact of the customer”

# Today's Date - Last transaction Date

df["InvoiceDate"].max()

today_date = dt.datetime(2011, 12, 11)


rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                     'Invoice': lambda num: num.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm = rfm[(rfm["Monetary"]) > 0 & (rfm["Frequency"] > 0)]

###############################################################
# Calculating RFM Scores
###############################################################

# Recency
rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])


rfm["RFM_SCORE"] = (rfm['RecencyScore'].astype(str) +
                    rfm['FrequencyScore'].astype(str) +
                    rfm['MonetaryScore'].astype(str))

rfm
#reset index yapilirsa idler float olmaz

seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At_Risk',
    r'[1-2]5': 'Cant_Loose',
    r'3[1-2]': 'About_to_Sleep',
    r'33': 'Need_Attention',
    r'[3-4][4-5]': 'Loyal_Customers',
    r'41': 'Promising',
    r'51': 'New_Customers',
    r'[4-5][2-3]': 'Potential_Loyalists',
    r'5[4-5]': 'Champions'
}

rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)
df[["Customer ID"]].nunique()
rfm[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "count"])

rfm[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "count",
                                                                             "min", "median", "max"])


############################################
# Task 3: Select "Loyal Customers"  segment and import Customer ID's to an excel file.
############################################

new_df = pd.DataFrame()

new_df["Loyal_Customers"] = rfm[rfm["Segment"] == "Loyal_Customers"].index
new_df["Loyal_Customers"] = rfm[rfm["Loyal_customers"].astype(int)
new_df.to_excel("D:/Data Science/git/projects/2-RFM-CLTV-Pareto/Loyal_Customers.xlsx")







