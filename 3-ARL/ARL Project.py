################################
#ARL (Association Rule Learning)
################################

pip install mlxtend
pip install pymysql
pip install mysql
pip install mysql-connector
pip install mysql-connector-python
pip install mysql-connector-python-rf
import pymysql
import mysql_connector
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

##########################################
# Downloading Data set from db
##########################################

# credentials.
creds = {'user': 'group3',
         'passwd': 'haydegidelum',
         'host': 'db.github.rocks',
         'port': 3306,
         'db': 'group3'}

pip install sqlalchemy

# MySQL connection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))

"""retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)
retail_mysql_df.info()"""
# retail_mysql_df["InvoiceDate"] = pd.to_datetime(retail_mysql_df["InvoiceDate"])


############################################
# Data Preprocessing
############################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df_ = pd.read_excel("D:/Data Science/dsmlbc4/datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df=df_.copy()
df.info()
df.head()


from helpers.helpers import check_df
check_df(df)

from helpers.helpers import crm_data_prep

df=crm_data_prep(df)
check_df(df)

df_ger=df[df["Country"]=="Germany"]
df_ger.head()
check_df(df_ger)

df_ger.groupby(["Invoice", "StockCode"]).agg({"Quantity":"sum"}).head(100)

df_ger.groupby(["Invoice", "StockCode"]).agg({"Quantity":"sum"}).unstack().iloc[0:50,0:50]

df_ger=df_ger[df_ger["Description"]!="POSTAGE"]
df_ger
df_ger.groupby(["Invoice", "StockCode"]).\
    agg({"Quantity":"sum"}).\
unstack().fillna(0).iloc[0:5,0:5]

df_ger.groupby(["Invoice", "StockCode"]).\
    agg({"Quantity":"sum"})\
    .unstack().fillna(0).\
    applymap(lambda x:1 if x>0 else 0).iloc[0:5,0:5]



def create_invoice_product_df(dataframe):
    return dataframe.groupby(["Invoice", "Description"])["Quantity"].\
               sum().unstack().fillna(0).\
               applymap(lambda x:1 if x>0 else 0)

ger_inv_pro_df=create_invoice_product_df(df_ger)

ger_inv_pro_df.head()

############################################
# Association Rule Mining
############################################

frequent_itemsets=apriori(ger_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

rules=association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

rules.head()
rules.sort_values("support", ascending=False).head(100)

rules.sort_values("lift", ascending=False).head(100)

rules.sort_values("antecedent support", ascending=False).head(100)


# How many unique items in each invoice.

unique_number_of_products = ger_inv_pro_df.aggregate("sum", axis=1)
unique_number_of_product.head()

# Number of unique basket for each item.
unique_number_of_invoice = ger_inv_pro_df.aggregate("sum", axis=0)
unique_number_of_invoice.head()



"""
    ROUND SNACK BOXES SET OF4 WOODLAND and ROUND SNACK BOXES SET OF 4 FRUITS are bought together often.
Their support (0.133630) is the highest among the all items however lift is only 3.3. First item is not 
promoting the second enough. A discoun in the second item may boost the sales.

    (SET/6 RED SPOTTY PAPER CUPS)&(SET/6 RED SPOTTY PAPER PLATES) are generally bought together. Their support 
is 0.046771 where the lift is 15.110577. An increase in the price may boost profits.

    CHILDRENS CUTLERY SPACEBOY & CHILDRENS CUTLERY DOLLY GIRL have support of 0.040089 and lift of 15.972332.
An increase in the price may boost profits. 

    ROUND SNACK BOXES SET OF4 WOODLAND is the most sold item(support=0.249443) a price increase for this item
may boost profit and price discount for the consequents may improve lift values and hence sales. 
"""