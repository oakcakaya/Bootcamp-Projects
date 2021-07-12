"""
Purpose of the project is to form new customer classes using current customer profiles to divide
customers into segments and to predict the segment of the new customers.

There two tables. 'user' table show the general characteristics of the customers
'purchase' holds the purchase information.
"""




import pandas as pd
import numpy as np

# 1-Loading datasets
# Tables merged on 'uid"

users=pd.read_csv('dsmlbc4/datasets/users.csv')
purchases=pd.read_csv('dsmlbc4/datasets/purchases.csv')

users.head()
purchases.head()

df=purchases.merge(users, how='inner', on='uid')
df.head()

# 2- Total sale in terms of country, device gender and age
df.groupby(["country","device", "gender", "age"]).agg({"price":"sum"})

# 3- Values sorted in descending order
agg_df=(df.groupby(["country","device", "gender", "age"]).agg({"price":"sum"})).sort_values(by="price",
        ascending=False)
agg_df.head()

# 4 - Index is added to agg_df
agg_df.columns

agg_df.reset_index(inplace=True)
agg_df.head()

# 5- Forming age categories as "age_cat"
agg_df["age"].max()
bins = [0, 18, 23, 30, 40, agg_df["age"].max()]
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_'+str(agg_df["age"].max())]
agg_df["age_cat"]=pd.cut(agg_df["age"], bins=bins,labels=mylabels)
agg_df.head()

# 6- Adding level based customers and purchases
agg_df["customers_level_based"]=[row[0] + "_" + row[1].upper() + "_" + row[2] + "_" + row[5] for row in agg_df.values]
agg_df=agg_df[["customers_level_based","price"]]
agg_df.head()

# 7- Dividing segments according to mean sales in every category
agg_df["customers_level_based"].count()
agg_df = agg_df.groupby("customers_level_based").agg({"price": "mean"})
agg_df = agg_df.reset_index()
agg_df["customers_level_based"].count()

agg_df.head()
agg_df["Segment"]=pd.qcut(agg_df["price"], 4, labels=["D", "C", "B", "A"])
agg_df.head()

agg_df.groupby("Segment").agg({"price":"mean"})

# 8- A new female IOS operating user from Turkey fall isto which category?
new_user="TUR_IOS_F_41_75"

print (agg_df[agg_df["customers_level_based"] == new_user])








