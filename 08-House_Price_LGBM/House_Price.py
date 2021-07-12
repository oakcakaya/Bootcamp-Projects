"""
### DATA SUMMARY ###

#Data is from Ames, Iowa, US
#very student oriented town in US
#66K population in 2019, 33K of them are students and 17K of this population is employed by University of Iowa

Dean De Cock wanted an alternative to Boston Housing Data used for regression analysis (1978)
from 2006 to 2010

only included the residential sales
if there are more sales for the same property, only most recent one is included in dataset
5 known outliers( 3 of them are partial sales and 2 of them are over 4000 square foot)
sale condition includes abnormal, which means short sales & foreclosures
Dean De Cock wrote that 80% of house price can be explained by total square footage ( Total BSMT SF + GRLIV AREA)
Being near a park >> positive relationship
Being near a rail road >> negative relationship

2919 observations 83 variables

###VARIABLES####
MSSubClass: Identifies the type of dwelling involved in the sale.

        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES

MSZoning: Identifies the general zoning classification of the sale.

       A	Agriculture
       C	Commercial
       FV	Floating Village Residential
       I	Industrial
       RH	Residential High Density
       RL	Residential Low Density
       RP	Residential Low Density Park
       RM	Residential Medium Density

LotFrontage: Linear feet of street connected to property

LotArea: Lot size in square feet

Street: Type of road access to property

       Grvl	Gravel
       Pave	Paved

Alley: Type of alley access to property

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access

LotShape: General shape of property

       Reg	Regular
       IR1	Slightly irregular
       IR2	Moderately Irregular
       IR3	Irregular

LandContour: Flatness of the property

       Lvl	Near Flat/Level
       Bnk	Banked - Quick and significant rise from street grade to building
       HLS	Hillside - Significant slope from side to side
       Low	Depression

Utilities: Type of utilities available

       AllPub	All public Utilities (E,G,W,& S)
       NoSewr	Electricity, Gas, and Water (Septic Tank)
       NoSeWa	Electricity and Gas Only
       ELO	Electricity only

LotConfig: Lot configuration

       Inside	Inside lot
       Corner	Corner lot
       CulDSac	Cul-de-sac
       FR2	Frontage on 2 sides of property
       FR3	Frontage on 3 sides of property

LandSlope: Slope of property

       Gtl	Gentle slope
       Mod	Moderate Slope
       Sev	Severe Slope

Neighborhood: Physical locations within Ames city limits

       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       IDOTRR	Iowa DOT and Rail Road
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	Northwest Ames
       OldTown	Old Town
       SWISU	South & West of Iowa State University
       Sawyer	Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	Timberland
       Veenker	Veenker

Condition1: Proximity to various conditions

       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street
       Norm	Normal
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad

Condition2: Proximity to various conditions (if more than one is present)

       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street
       Norm	Normal
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad

BldgType: Type of dwelling

       1Fam	Single-family Detached
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex
       TwnhsE	Townhouse End Unit
       TwnhsI	Townhouse Inside Unit

HouseStyle: Style of dwelling

       1Story	One story
       1.5Fin	One and one-half story: 2nd level finished
       1.5Unf	One and one-half story: 2nd level unfinished
       2Story	Two story
       2.5Fin	Two and one-half story: 2nd level finished
       2.5Unf	Two and one-half story: 2nd level unfinished
       SFoyer	Split Foyer
       SLvl	Split Level

OverallQual: Rates the overall material and finish of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor

OverallCond: Rates the overall condition of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor

YearBuilt: Original construction date

YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)

RoofStyle: Type of roof

       Flat	Flat
       Gable	Gable
       Gambrel	Gabrel (Barn)
       Hip	Hip
       Mansard	Mansard
       Shed	Shed

RoofMatl: Roof material

       ClyTile	Clay or Tile
       CompShg	Standard (Composite) Shingle
       Membran	Membrane
       Metal	Metal
       Roll	Roll
       Tar&Grv	Gravel & Tar
       WdShake	Wood Shakes
       WdShngl	Wood Shingles

Exterior1st: Exterior covering on house

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles

Exterior2nd: Exterior covering on house (if more than one material)

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles

MasVnrType: Masonry veneer type

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone

MasVnrArea: Masonry veneer area in square feet

ExterQual: Evaluates the quality of the material on the exterior

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor

ExterCond: Evaluates the present condition of the material on the exterior

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor

Foundation: Type of foundation

       BrkTil	Brick & Tile
       CBlock	Cinder Block
       PConc	Poured Contrete
       Slab	Slab
       Stone	Stone
       Wood	Wood

BsmtQual: Evaluates the height of the basement

       Ex	Excellent (100+ inches)
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement

BsmtCond: Evaluates the general condition of the basement

       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed
       Fa	Fair - dampness or some cracking or settling
       Po	Poor - Severe cracking, settling, or wetness
       NA	No Basement

BsmtExposure: Refers to walkout or garden level walls

       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement

BsmtFinType1: Rating of basement finished area

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement

BsmtFinSF1: Type 1 finished square feet

BsmtFinType2: Rating of basement finished area (if multiple types)

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement

BsmtFinSF2: Type 2 finished square feet

BsmtUnfSF: Unfinished square feet of basement area

TotalBsmtSF: Total square feet of basement area

Heating: Type of heating

       Floor	Floor Furnace
       GasA	Gas forced warm air furnace
       GasW	Gas hot water or steam heat
       Grav	Gravity furnace
       OthW	Hot water or steam heat other than gas
       Wall	Wall furnace

HeatingQC: Heating quality and condition

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor

CentralAir: Central air conditioning

       N	No
       Y	Yes

Electrical: Electrical system

       SBrkr	Standard Circuit Breakers & Romex
       FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)
       FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
       FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
       Mix	Mixed

1stFlrSF: First Floor square feet

2ndFlrSF: Second floor square feet

LowQualFinSF: Low quality finished square feet (all floors)

GrLivArea: Above grade (ground) living area square feet

BsmtFullBath: Basement full bathrooms

BsmtHalfBath: Basement half bathrooms

FullBath: Full bathrooms above grade

HalfBath: Half baths above grade

Bedroom: Bedrooms above grade (does NOT include basement bedrooms)

Kitchen: Kitchens above grade

KitchenQual: Kitchen quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor

TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

Functional: Home functionality (Assume typical unless deductions are warranted)

       Typ	Typical Functionality
       Min1	Minor Deductions 1
       Min2	Minor Deductions 2
       Mod	Moderate Deductions
       Maj1	Major Deductions 1
       Maj2	Major Deductions 2
       Sev	Severely Damaged
       Sal	Salvage only

Fireplaces: Number of fireplaces

FireplaceQu: Fireplace quality

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace

GarageType: Garage location

       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage

GarageYrBlt: Year garage was built

GarageFinish: Interior finish of the garage

       Fin	Finished
       RFn	Rough Finished
       Unf	Unfinished
       NA	No Garage

GarageCars: Size of garage in car capacity

GarageArea: Size of garage in square feet

GarageQual: Garage quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage

GarageCond: Garage condition

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage

PavedDrive: Paved driveway

       Y	Paved
       P	Partial Pavement
       N	Dirt/Gravel

WoodDeckSF: Wood deck area in square feet

OpenPorchSF: Open porch area in square feet

EnclosedPorch: Enclosed porch area in square feet

3SsnPorch: Three season porch area in square feet

ScreenPorch: Screen porch area in square feet

PoolArea: Pool area in square feet

PoolQC: Pool quality

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool

Fence: Fence quality

       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence

MiscFeature: Miscellaneous feature not covered in other categories

       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None

MiscVal: $Value of miscellaneous feature

MoSold: Month Sold (MM)

YrSold: Year Sold (YYYY)

SaleType: Type of sale

       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash
       VWD	Warranty Deed - VA Loan
       New	Home just constructed and sold
       COD	Court Officer Deed/Estate
       Con	Contract 15% Down payment regular terms
       ConLw	Contract Low Down payment and low interest
       ConLI	Contract Low Interest
       ConLD	Contract Low Down
       Oth	Other

SaleCondition: Condition of sale

       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
       AdjLand	Adjoining Land Purchase
       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit
       Family	Sale between family members
       Partial	Home was not completed when last assessed (associated with New Homes)
"""

### IMPORTING LIBRARIES

from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from helpers.data_prep import *
from helpers.eda import *


# MERGING TRAIN AND TEST
train = pd.read_csv("dsmlbc4/datasets/house price/train.csv")
test = pd.read_csv("dsmlbc4/datasets/house price/test.csv")
df = train.append(test).reset_index(drop=True)
df.head()


######################################
# EDA
######################################

check_df(df)

#Correting the typo of GarageYrBlt-2207 to 2007
df.loc[df["GarageYrBlt"]==2207,"GarageYrBlt"] = 2007

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)



######################################
# Categorical variable analysis
######################################

for col in cat_cols:
    cat_summary(df, col)

for col in cat_but_car:
    cat_summary(df, col)

for col in num_but_cat:
    cat_summary(df, col)

######################################
# Numerical Value Analysis
######################################

df[num_cols].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

for col in num_cols:
    num_summary(df, col, plot=True)




#############################################
####Correlations Check for Num Cols #########
##############################################
####Visualisation of the data ####

#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show();




# Correlation of independent variables with target 
def find_correlation(dataframe, numeric_cols, corr_limit=0.40):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col)
            else:
                low_correlations.append(col)
    return low_correlations, high_correlations

#setting high and low correlating
low_corrs, high_corrs = find_correlation(df, num_cols)

#high corrs columns
high_corrs
"""
['OverallQual',
 'YearBuilt',
 'YearRemodAdd',
 'MasVnrArea',
 'TotalBsmtSF',
 '1stFlrSF',
 'GrLivArea',
 'TotRmsAbvGrd',
 'GarageYrBlt',
 'GarageArea']"""


######################################
# DATA PREPROCESSING & FEATURE ENGINEERING
######################################

######################################
# FEATURE ENGINEERING
######################################
#Age of the house at the sale date
df['NEW_Age_at_Sale'] =df["YrSold"]-df["YearBuilt"]

#Age of the last remodelling at the sale date
df["NEW_Age_of_Restoration"]=df["YrSold"]-df["YearRemodAdd"]

# Total sqfootage
df['NEW_TotalSQFoot'] = df['TotalBsmtSF'] + df['GrLivArea']

#Total number of baths
df["NEW_Bath"] = df["FullBath"] + (0.5 * df["HalfBath"])


quality = {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1}
df["NEW_ExterQual"] = df["ExterQual"].map(quality).astype('int')
df["NEW_GarageQual"]= df["GarageQual"].map(quality)
df["BsmtQual"]= df["BsmtQual"].map(quality)

df["NEW_General_Quality"]=(df["NEW_ExterQual"]**2)*(df["NEW_GarageQual"]**2)*df["BsmtQual"]

df["NEW_Age_Range"]=pd.cut(x=df['NEW_Age_at_Sale'],bins=[-2,0,6,16,31,150], labels=["BrandNew", "New", "FairlyOld" ,"Old", "VeryOld"] )


df['NEW_Fireplace'] = pd.cut(x=df['Fireplaces'],bins=[-1,0,1,5], labels=["None", "Normal", "AbvAvg"])


df["NEW_Garage_Cars"]=pd.cut(x=df['GarageCars'],bins=[-1,0,1,2,5], labels=["None", "Single", "Double", "AbvAvg"])

derived_cat_cols=["NEW_Garage_Cars", 'NEW_Fireplace', "NEW_Age_Range"]

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)


######################################
# RARE ENCODING
######################################

rare_analyser(df, "SalePrice", 0.01)
df = rare_encoder(df, 0.01)

rare_analyser(df, "SalePrice", 0.01)


drop_list = ["Street", "Utilities", "LandSlope", "PoolQC", "MiscFeature"]
#alt droplist
#drop_list = ["Street", "Utilities", "LandSlope", "PoolQC", "MiscFeature", 'GrLivArea','TotalBsmtSF', "YearRemodAdd", "YearBuilt" ]

cat_cols = [col for col in cat_cols if col not in drop_list]

for col in drop_list:
    df.drop(col, axis=1, inplace=True)


rare_analyser(df, "SalePrice", 0.01)


######################################
# MISSING_VALUES
######################################
import missingno as msno
missing_values_table(df)

#Missing values will not be treated since a LGBM is to be used
######################################
# OUTLIERS
######################################

for col in num_cols:
    print(col, check_outlier(df, col))


df["SalePrice"].describe().T

replace_with_thresholds(df, "SalePrice")





######################################
# LABEL ENCODING & ONE-HOT ENCODING
######################################

cat_cols = cat_cols + cat_but_car

df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)

# def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
#     original_columns = list(dataframe.columns)
#     dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
#     new_columns = [c for c in dataframe.columns if c not in original_columns]
#     return dataframe, new_columns




######################################
# Seperation of TEST from TRAIN
######################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

train_df.to_pickle("D:/Data Science/dsmlbc4/Courses/Week-9/Assignment/train_df.pkl")
test_df.to_pickle("D:/Data Science/dsmlbc4/Courses/Week-9/Assignment//test_df.pkl")

#######################################
# MODEL: LGBM
#######################################
import warnings
from lightgbm import LGBMRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


X = train_df.drop(['SalePrice', "Id"], axis=1)
y = np.log1p(train_df['SalePrice'])

# y = train_df["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=46)

lgb_model = LGBMRegressor(random_state=46).fit(X_train, y_train)
y_pred = lgb_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

y.mean()

y_pred = lgb_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
#0.1218

#######################################
# Model Tuning
#######################################
lgb_params = {"learning_rate": [0.001, 0.01, 0.05, 0.1],
               "n_estimators": [1000, 1500, 2000, 2500],
               "max_depth": [3, 5, 8, 10],
               "num_leaves":[5, 8, 15],
               "colsample_bytree": [0.8, 0.5, 0.3]}

lgb_model = LGBMRegressor(random_state=42)
lgb_cv_model = GridSearchCV(lgb_model, lgb_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
lgb_cv_model.best_params_
"""
{'colsample_bytree': 0.3,
 'learning_rate': 0.01,
 'max_depth': 3,
 'n_estimators': 2500,
 'num_leaves': 8}"""

#######################################
# Final Model
#######################################

lgb_tuned = LGBMRegressor(**lgb_cv_model.best_params_).fit(X_train, y_train)

y_pred = lgb_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

y_pred = lgb_tuned.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
#0.1057

#######################################
# Feature Importance
#######################################

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


plot_importance(lgb_tuned, X_train, 50)



#######################################
# Uploading Results
#######################################

submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"]

y_pred_sub = lgb_tuned.predict(test_df.drop("Id", axis=1))
y_pred_sub = np.expm1(y_pred_sub)

submission_df['SalePrice'] = y_pred_sub

submission_df.head()

submission_df.to_csv('D:/Data Science/dsmlbc4/Courses/Week-9/Assignment/houseprice_lgbm.csv', index=False)

#Kaggle Score:0.12956





























































































