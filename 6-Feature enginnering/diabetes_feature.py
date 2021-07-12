"""


Title: Pima Indians Diabetes Database

Sources: (a) Original owners: National Institute of Diabetes and Digestive and Kidney Diseases
(b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu) Research Center, RMI Group
Leader Applied Physics Laboratory The Johns Hopkins University Johns Hopkins Road Laurel, MD 20707
(301) 953-6231 © Date received: 9 May 1990

Past Usage:

Smith,~J.~W., Everhart,~J.~E., Dickson,~W.~C., Knowler,~W.~C., & Johannes,~R.~S. (1988). Using the
ADAP learning algorithm to forecast the onset of diabetes mellitus. In {it Proceedings of the Symposium
on Computer Applications and Medical Care} (pp. 261–265). IEEE Computer Society Press.

The diagnostic, binary-valued variable investigated is whether the patient shows signs of diabetes
according to World Health Organization criteria (i.e., if the 2 hour post-load plasma glucose was at
least 200 mg/dl at any survey examination or if found during routine medical care). The population
lives near Phoenix, Arizona, USA.

Results: Their ADAP algorithm makes a real-valued prediction between 0 and 1. This was transformed
into a binary decision using a cutoff of 0.448. Using 576 training instances, the sensitivity and
specificity of their algorithm was 76% on the remaining 192 instances.

Relevant Information: Several constraints were placed on the selection of these instances from a
larger database. In particular, all patients here are females at least 21 years old of Pima Indian
heritage. ADAP is an adaptive learning routine that generates and executes digital analogs of
perceptron-like devices. It is a unique algorithm; see the paper for details.

Number of Instances: 768

Number of Attributes: 8 plus class

For Each Attribute: (all numeric-valued)

Number of times pregnant
Plasma glucose concentration a 2 hours in an oral glucose tolerance test
Diastolic blood pressure (mm Hg)
Triceps skin fold thickness (mm)
2-Hour serum insulin (mu U/ml)
Body mass index (weight in kg/(height in m)^2)
Diabetes pedigree function
Age (years)
Class variable (0 or 1)
Missing Attribute Values: None

"""






import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


def load_diabetes():
    data = pd.read_csv("dsmlbc4/datasets/diabetes.csv")
    return data

df = load_diabetes()
df.head()
df.shape
df.describe().T

df.isnull().any()


cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
if len(cat_cols) == 0:
    print("There is not Categorical Column",",","Number of Numerical Columns: ", len(num_cols), "\n", num_cols)
elif len(num_cols) == 0:
    print("There is not Numerical Column",",","Number of Categorical Column: ", len(cat_cols), "\n", cat_cols)
else:
    print("")


num_cols = [col for col in df.columns if len(df[col].unique()) > 20
            and df[col].dtypes != 'O'
            and col not in "Outcome"]
num_cols

###Pregnancies is num but cat column.

#Distribution of variables

df.describe([0.05,0.25,0.50,0.75,0.90,0.95,0.99]).T
df.isnull().sum()
#df shown no missing values however,
#values other than pregnancies cannot be 0, these should be missing values
# 0 values to be replace by Nan .

nan_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in nan_cols:
    df[col].replace(0,np.NaN,inplace=True)


def missin_val_count(dataframe):
    print(da)

df.corr()
df.info


# FEATURE ENGINEERING
# New categories are produced
df['Age_CAT']=pd.cut(x=df["Age"], bins=[0,18,35,60,100], labels=["young","young_adult","older_adult", "senior"])

df['BMI_CAT']=pd.cut(x=df["BMI"], bins=[0,18.5,25,30,35,100], labels=["underweight","healty_weight","over_weight", "obese", "extemely_obese"])

df['BloodPressure_CAT']=pd.cut(x=df["BloodPressure"], bins=[0,80,90,120,200], labels=["normal","High_blood_pressure_1","High_blood_pressure_2", "'hypertensive_crisis"])

df['Glucose_CAT']=pd.cut(x=df["Glucose"], bins=[0,140,200], labels=["normal","prediabetes",])

df['SkinThickness_Range']=pd.cut(x=df["SkinThickness"], bins=[0,22,28,33,100], labels=["lean","ideal","average", "overfat"])

df['Insulin_Range']=pd.cut(x=df["Insulin"], bins=[0,80,150,1000], labels=["low","normal","abnormal"])


#Since the correlation between BMI and other variables high and the missing values in BMI low in number,
#dropping out missing BMI is a good option, since the missing value count is low for "Glucose" these values
# are dropped as well.

df=df.dropna(subset=["BMI", "Glucose"])
df
df.isnull().sum()
df.info

#Filling missing values
df.head()


df["SkinThickness"].fillna(df.groupby("BMI_CAT")["SkinThickness"].transform("mean"), inplace=True)
df.isnull().sum()
df["Insulin"].fillna(df.groupby("Glucose_CAT")["Insulin"].transform("mean"), inplace=True)
df.isnull().sum()
df.corr()


df["BloodPressure"].fillna(df["BloodPressure"].mean(), inplace=True)
df.isnull().sum()

#checking outliers

from helpers.data_prep import check_outlier

for col in num_cols:
    print(col, check_outlier(df, col))

#recplacing with tresholds.

from helpers.data_prep import replace_with_thresholds

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

from helpers.eda import check_df
check_df(df)
# Re-assignment of feature classes after filling missing values

df['Age_CAT']=pd.cut(x=df["Age"], bins=[0,18,35,60,100], labels=["young","young_adult","older_adult", "senior"])

df['BMI_CAT']=pd.cut(x=df["BMI"], bins=[0,18.5,25,30,35,100], labels=["underweight","healty_weight","over_weight", "obese", "extemely_obese"])

df['BloodPressure_CAT']=pd.cut(x=df["BloodPressure"], bins=[0,80,90,120,200], labels=["normal","High_blood_pressure_1","High_blood_pressure_2", "'hypertensive_crisis"])

df['Glucose_CAT']=pd.cut(x=df["Glucose"], bins=[0,140,200], labels=["normal","prediabetes",])

df['SkinThickness_Range']=pd.cut(x=df["SkinThickness"], bins=[0,22,28,33,100], labels=["lean","ideal","average", "overfat"])

df['Insulin_Range']=pd.cut(x=df["Insulin"], bins=[0,80,150,1000], labels=["low","normal","abnormal"])

df.head()

#####################################################################
#LABEL ENCODING
#####################################################################
binary_cols = [col for col in df.columns if len(df[col].unique()) ==2 and df[col].dtypes == 'O']

binary_cols

#No binary columns in dataset
from helpers.data_prep import label_encoder

for col in binary_cols:
    df = label_encoder(df, col)

df.head()


#####################################################################
#ONE HOT ENCODING
#####################################################################
ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
ohe_cols
from helpers.data_prep import one_hot_encoder
df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


#####################################################################
#RARE ENCODING
#onehot encoding oncesi yapmak önemli. Eğer değişkenler içerisinde sınıf sayısı az olan değerler olsaydı:
from helpers.data_prep import rare_analyser

rare_analyser(df, "Outcome", 0.05)

from helpers.data_prep import rare_encoder
df = rare_encoder(df, 0.01)
rare_analyser(df, "Outcome", 0.01)

df.head()
######################################################################


