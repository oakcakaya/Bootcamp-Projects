##################################################
# Pareto Analysis
##################################################

# Business management --> "80% of sales come from 20% of clients"

##################################################
# IMPORT LIBRARIES AND DATA
##################################################

import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

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



################################################################################
# 1. Find the products that make up about 80 percent of the company's revenues.
# What is the ratio of these products to all products?
################################################################################

product_df=df.groupby(["StockCode"]).agg({"TotalPrice":"sum"}).sort_values("TotalPrice",ascending=False)
product_df.reset_index()

total_revenue=product_df["TotalPrice"].sum()


indx=0
revenue=0
for i in product_df["TotalPrice"]:
    indx += 1
    revenue += i
    if (revenue/total_revenue)>=0.8:
        break
    print(indx, revenue)

product_perc=indx/product_df.shape[0]*100

print("Revenue Ratio:{}, Product Ratio:{}".format((revenue/total_revenue)*100,product_perc))

print("{} % of Revenue is obtained from {} % of products".format((revenue/total_revenue)*100,product_perc))

################################################################################
# 2. Find customers that make up about 80 percent of the company's revenues.
# What is the ratio of these customers to all customers?
################################################################################

customer_df=df.groupby(["Customer ID"]).agg({"TotalPrice":"sum"}).sort_values("TotalPrice",ascending=False)
customer_df.reset_index(inplace=True)

total_revenue=customer_df["TotalPrice"].sum()

indx=0
cus_revenue=0
for i in customer_df["TotalPrice"]:
    indx += 1
    cus_revenue += i
    if (cus_revenue/total_revenue)>=0.8:
        break
print(indx, revenue)

cus_perc=indx/customer_df.shape[0]*100

print("Revenue Ratio:{}, Product Ratio:{}".format((cus_revenue/total_revenue)*100,cus_perc))
print("{} % of Revenue is obtained from {} % of customers".format((cus_revenue/total_revenue)*100,cus_perc))























