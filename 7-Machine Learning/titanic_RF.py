import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
from helpers.eda import *
from helpers.data_prep import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


#Loading Titanic dataset
def load():
    data = pd.read_csv("D:/Data Science/dsmlbc4/Courses/Week-8/Assignments/train.csv")
    return data

def load_test():
    data = pd.read_csv("D:/Data Science/dsmlbc4/Courses/Week-8/Assignments/test.csv")
    return data

df_train = load()
df_test = load_test()
df_train.head()

df_train.isnull().sum()

df_test.isnull().sum()

#Deriving new variables
def titanic_data_prep(dataframe):

    # FEATURE ENGINEERING
    dataframe["NEW_CABIN_BOOL"] = dataframe["Cabin"].isnull().astype('int')
    dataframe["NEW_NAME_COUNT"] = dataframe["Name"].str.len()
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["Name"].apply(lambda x: len(str(x).split(" ")))
    dataframe["NEW_NAME_DR"] = dataframe["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
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


train_df = titanic_data_prep(df_train)
df_train.isnull().sum()

test_df=titanic_data_prep(df_test)
test_df.isnull().sum()

#Dealing with missing values in Test set
test_df.head()
test_df["FARE"].fillna(test_df["FARE"].median(), inplace=True)
test_df["AGE"].fillna(test_df["AGE"].median(), inplace=True)
X_test.fillna(X_test["NEW_AGE_PCLASS" ].median(), inplace=True)


##################################################
#Machine Learning Random Forests
##################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


X = train_df.drop(["SURVIVED","PASSENGERID"], axis=1)
y = train_df["SURVIVED"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)


#RANDOM FORESTS
rf_model = RandomForestClassifier(random_state=42).fit(X_train,y_train)

# Training Error
y_pred = rf_model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print("Accuracy:{}%".format(accuracy * 100.0))
# Accuracy: 99.44 %

y_prob = rf_model.predict_proba (X_train)[:, 1]
roc_auc_score (y_pred, y_prob)
# AUC => 0.99

#Test Error
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:{}%".format(accuracy * 100.0))
#Accuracy: 82.12 %

rf_params = {"max_depth": [5, 8, 10, None],
             "max_features": [3, 5, 12],
             "n_estimators": [200, 300, 500, 1000],
             "min_samples_split": [2, 5, 8, 12]}


rf_model = RandomForestClassifier(random_state=42)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_

"""
{'max_depth': 5,
 'max_features': 12,
 'min_samples_split': 2,
 'n_estimators': 1000}"""


rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_, random_state=42 ).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:{}%".format(accuracy * 100.0))
#Accuracy:82.68%


#Model test for all test data

X = train_df.drop(["SURVIVED","PASSENGERID"], axis=1)
y = train_df["SURVIVED"]


rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_, random_state=42).fit(X, y)
y_pred = rf_tuned.predict(X)

accuracy = accuracy_score(y, y_pred)
print("Accuracy:{}%".format(accuracy * 100.0))
#Accuracy:85.63%
y_prob = rf_tuned.predict_proba (X)[:, 1]
roc_auc_score (y_pred, y_prob)
print("AUC:{}".format(roc_auc_score (y_pred, y_prob)))
# AUC score=> 1.0



#fitting model to test set


X_test = test_df.drop(["PASSENGERID"], axis=1)

#rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_, random_state=42).fit(X, y)
y_pred = rf_tuned.predict(X_test)
X_test.isnull().sum()


final_df=test_df[["PASSENGERID","Survived"]]


final_df.reset_index(drop=True, inplace=True)
final_df.to_csv('D:/Data Science/dsmlbc4/Courses/Week-8/Assignments/kagglesubmission.csv', index=False)

#kaggle score=0.76555







































































