import numpy as np
import pandas as pd

import pickle
from helpers.data_prep import *
from helpers.eda import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

"""
Data Dictionary
Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
"""

#Loading Titanic dataset
def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()

#Deriving new variables

def titanic_data_prep(dataframe):

    # FEATURE ENGINEERING
    #Cabin tpyes 1st = Upper, 2nd = Middle, 2nd = Middle, no cabin=crew
    dataframe["NEW_CABIN_BOOL"] = dataframe["Cabin"].isnull().astype('int')
    dataframe["NEW_NAME_COUNT"] = dataframe["Name"].str.len()
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["Name"].apply(lambda x: len(str(x).split(" ")))
    dataframe["NEW_NAME_DR"] = dataframe["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    # Title abbreviations shows certain social status or job positions
    dataframe['NEW_TITLE'] = dataframe.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SibSp"] + dataframe["Parch"] + 1
    dataframe["NEW_AGE_PCLASS"] = dataframe["Age"] * dataframe["Pclass"]

    dataframe.loc[((dataframe['SibSp'] + dataframe['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SibSp'] + dataframe['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

    dataframe.loc[(dataframe['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['Age'] >= 18) & (dataframe['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    dataframe.loc[(dataframe['Sex'] == 'male') & (dataframe['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['Sex'] == 'male') & ((dataframe['Age'] > 21) & (dataframe['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['Sex'] == 'male') & (dataframe['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['Sex'] == 'female') & (dataframe['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['Sex'] == 'female') & ((dataframe['Age'] > 21) & (dataframe['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['Sex'] == 'female') & (dataframe['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # Outliers
    num_cols = [col for col in dataframe.columns if len(dataframe[col].unique()) > 20
                and dataframe[col].dtypes != 'O'
                and col not in "PASSENGERID"]

    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # for col in num_cols:
    #    print(col, check_outlier(df, col))
    # print(check_df(df))


    # Missing Values
    dataframe.drop(["TICKET", "NAME", "CABIN"], inplace=True, axis=1)
    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))

    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    # LABEL ENCODING
    binary_cols = [col for col in dataframe.columns if len(dataframe[col].unique()) == 2 and dataframe[col].dtypes == 'O']

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    dataframe = rare_encoder(dataframe, 0.01)

    ohe_cols = [col for col in dataframe.columns if 10 >= len(dataframe[col].unique()) > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)

    return dataframe

titanic_data_prep(df)

