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

df.isnull().sum()
df.corr()
df.info


# FEATURE ENGINEERING
# New categories are produced
df['Age_CAT']=pd.cut(x=df["Age"], bins=[0,18,35,60,100], labels=["young","young_adult","older_adult", "senior"])

df['BMI_CAT']=pd.cut(x=df["BMI"], bins=[0,18.5,25,30,35,100], labels=["underweight","healty_weight","over_weight", "obese", "extemely_obese"])

df['BloodPressure_CAT']=pd.cut(x=df["BloodPressure"], bins=[0,80,90,120,200], labels=["normal","High_blood_pressure_1","High_blood_pressure_2", "'hypertensive_crisis"])

df['Glucose_CAT']=pd.cut(x=df["Glucose"], bins=[0,140,200, 250], labels=["normal","high", "very_high"])

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

df['Glucose_CAT']=pd.cut(x=df["Glucose"], bins=[0,140,200, 250], labels=["normal","prediabetes", "very_high"])

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
ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) >= 2 ]
ohe_cols.pop(0)
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
#Machine Learning With CART
######################################################################

import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn.tree import export_graphviz, export_text
from skompiler import skompile

from helpers.eda import *
pd.set_option('display.max_columns', None)

df

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)
y_pred = cart_model.predict(X)
y_prob = cart_model.predict_proba(X)[:, 1]
print(classification_report(y, y_pred))
roc_auc_score(y, y_prob)
"""
  precision    recall  f1-score   support
           0       1.00      1.00      1.00       488
           1       1.00      1.00      1.00       264
    accuracy                           1.00       752
   macro avg       1.00      1.00      1.00       752
weighted avg       1.00      1.00      1.00       752
AUC: 1.0
"""


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# training error
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)
"""
 precision    recall  f1-score   support
           0       1.00      1.00      1.00       395
           1       1.00      1.00      1.00       206
    accuracy                           1.00       601
   macro avg       1.00      1.00      1.00       601
weighted avg       1.00      1.00      1.00       601
AUC: 1.0
"""


# test error
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

"""
precision    recall  f1-score   support
           0       0.81      0.83      0.82        93
           1       0.71      0.69      0.70        58
    accuracy                           0.77       151
   macro avg       0.76      0.76      0.76       151
weighted avg       0.77      0.77      0.77       151
AUC: 0.7588060808305525
"""

# Decision tree
################################

def tree_graph_to_png(tree, feature_names, png_file_to_save):
    tree_str = export_graphviz(tree, feature_names=feature_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(png_file_to_save)

tree_graph_to_png(tree=cart_model, feature_names=X_train.columns, png_file_to_save='_diabetes_cart.png')

# Feature Importances
################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(cart_model, X_train)

# Hyperpameter Tuning
################################


cart_params = {'max_depth': range(1, 11),
               "min_samples_split": [2, 3, 4]}

cart_cv = GridSearchCV(cart_model, cart_params, cv=10, n_jobs=-1, verbose=True)
cart_cv.fit(X_train, y_train)

cart_cv.best_params_
# > {'max_depth': 3, 'min_samples_split': 2}

cart_tuned = DecisionTreeClassifier(**cart_cv.best_params_, random_state=17).fit(X_train, y_train)



# training error
y_pred = cart_tuned.predict(X_train)
y_prob = cart_tuned.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# test error
y_pred = cart_tuned.predict(X_test)
y_prob = cart_tuned.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

"""
 precision    recall  f1-score   support
           0       0.83      0.84      0.83        93
           1       0.74      0.72      0.73        58
    accuracy                           0.79       151
   macro avg       0.78      0.78      0.78       151
weighted avg       0.79      0.79      0.79       151
AUC: 0.8555802743789396
"""






















