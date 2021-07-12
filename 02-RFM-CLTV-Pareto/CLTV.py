##########################################################
# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
##########################################################

# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 20)
# pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_excel("D:/Data Science/dsmlbc4/datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df=df_.copy()
df.head()



##################################################
# Data Preprocessing
##################################################
#Number of null values
df.isnull().sum()

#Dropping samples containing null values
df.dropna(inplace=True)
df.isnull().sum()

#Dropping out "Cancelled" transactions
df=df[~df["Invoice"].str.contains("C", na=False)]

#Dropping out any mistaken transactions
df= df[(df["Quantity"]>0)]

df.shape

df["TotalPrice"]=df["Quantity"]*df["Price"]
df.head()

cltv_df=df.groupby("Customer ID").agg({'Invoice': lambda x: x.nunique(),
                                      'Quantity': lambda x: x.sum(),
                                      'TotalPrice': lambda x: x.sum()})

cltv_df.columns=['total_transaction', 'total_unit', 'total_price']
cltv_df.head()

##################################################
# 1. Calculate Average Order Value
##################################################

# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin

# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
cltv_df["avg_order_value"]=cltv_df["total_price"]/cltv_df["total_transaction"]
cltv_df.head()


##################################################
# 2. Calculate Purchase Frequency
##################################################
# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin


# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
#total number of customers
cltv_df.shape[0]

cltv_df["purchase_frequency"]=cltv_df["total_transaction"]/cltv_df.shape[0]
cltv_df.head()


##################################################
# 3. Calculate Repeat Rate and Churn Rate
##################################################
# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin

Churn_Rate = 1 - Repeat_Rate

#repeat rate=total number of customer having more than 1 transaction over total customers
repeat_rate=cltv_df[cltv_df.total_transaction>1].shape[0]/cltv_df.shape[0]
repeat_rate

churn_rate=1-repeat_rate
churn_rate

##################################################
# 4. Calculate Profit Margin
##################################################
#for estimated 5% profit, cost(100x)+profit(5x)=price(105x) so the profit shoul be alculated as (5/105)

cltv_df["profit_margin"]=cltv_df["total_price"]*(5/105)
cltv_df.head()

##################################################
# 5. Calculate Customer Lifetime Value
##################################################


# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin

# Customer_Value = Average_Order_Value * Purchase_Frequency
cltv_df["CV"]=cltv_df["avg_order_value"]*cltv_df["purchase_frequency"]
cltv_df.head()

# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
cltv_df["CLTV"]=(cltv_df["CV"]/churn_rate)*cltv_df["profit_margin"]
cltv_df.head()

cltv_df.sort_values("CLTV", ascending=False)

scaler=MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_df[["CLTV"]])

cltv_df["SCALED_CLTV"]=scaler.transform(cltv_df[["CLTV"]])
cltv_df.head()

cltv_df.sort_values("CLTV", ascending=False)

cltv_df["Segment"]=pd.qcut(cltv_df["SCALED_CLTV"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

cltv_df[["Segment","total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].sort_values(by="SCALED_CLTV",
                                                                                               ascending=False).head()

cltv_df.groupby("Segment")[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].agg(
    {"count", "mean", "sum"})




























