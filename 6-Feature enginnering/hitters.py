"""

Content
This dataset was originally taken from the StatLib library which is maintained at Carnegie Mellon University. This is part of the data that was used in the 1988 ASA Graphics Section Poster Session. The salary data were originally from Sports Illustrated, April 20, 1987. The 1986 and career statistics were obtained from The 1987 Baseball Encyclopedia Update published by Collier Books, Macmillan Publishing Company, New York.

Format
A data frame with 322 observations of major league players on the following 20 variables.
AtBat Number of times at bat in 1986
Hits Number of hits in 1986
HmRun Number of home runs in 1986
Runs Number of runs in 1986
RBI Number of runs batted in in 1986
Walks Number of walks in 1986
Years Number of years in the major leagues
CAtBat Number of times at bat during his career
CHits Number of hits during his career
CHmRun Number of home runs during his career
CRuns Number of runs during his career
CRBI Number of runs batted in during his career
CWalks Number of walks during his career
League A factor with levels A and N indicating player’s league at the end of 1986
Division A factor with levels E and W indicating player’s division at the end of 1986
PutOuts Number of put outs in 1986
Assists Number of assists in 1986
Errors Number of errors in 1986
Salary 1987 annual salary on opening day in thousands of dollars
NewLeague A factor with levels A and N indicating player’s league at the beginning of 1987


http://m.mlb.com/glossary/standard-stats

Offense Stats:
At-bat (AB):An official at-bat comes when a batter reaches base via a fielder's choice,
hit or an error (not including catcher's interference) or when a batter is
put out on a non-sacrifice. (Whereas a plate appearance refers to each
completed turn batting, regardless of the result.)

Hit (H):A hit occurs when a batter strikes the baseball into fair territory and reaches
base without doing so via an error or a fielder's choice. There are four types of hits
in baseball: singles, doubles, triples and home runs. All four are counted equally when
deciphering batting average. If a player is thrown out attempting to take an extra base
(e.g., turning a single into a double), that still counts as a hit.

Home Run (HR):A home run occurs when a batter hits a fair ball and scores on the play
without being put out or without the benefit of an error.

Run (R):A player is awarded a run if he crosses the plate to score his team a run.
When tallying runs scored, the way in which a player reached base is not considered.
If a player reaches base by an error or a fielder's choice, as long as he comes around
to score, he is still credited with a run. If a player enters the game as a pinch-runner
and scores, he is also credited with a run.

Runs Batted In (RBI):A batter is credited with an RBI in most cases where the result of
his plate appearance is a run being scored. There are a few exceptions, however. A player
does not receive an RBI when the run scores as a result of an error or ground into double play.

Walk (BB):A walk (or base on balls) occurs when a pitcher throws four pitches out of the strike
zone, none of which are swung at by the hitter. After refraining from swinging at four pitches
out of the zone, the batter is awarded first base. In the scorebook, a walk is denoted by the letters BB.

Defense Stats:
Putout (PO):A fielder is credited with a putout when he is the fielder who physically records
the act of completing an out -- whether it be by stepping on the base for a forceout, tagging a runner,
catching a batted ball, or catching a third strike. A fielder can also receive a putout when he is the
fielder deemed by the official scorer to be the closest to a runner called out for interference.

Assist (A):An assist is awarded to a fielder who touches the ball before a putout is recorded by another
fielder. Typically, assists are awarded to fielders when they throw the ball to another player -- but a fielder
receives an assist as long as he touches the ball, even if the contact was unintentional. For example, on a line
drive that strikes the pitcher before caroming to the shortstop, both the pitcher and shortstop are awarded an
assist if the out is made on a throw to first base.

Error (E):A fielder is given an error if, in the judgment of the official scorer, he fails to convert an out on
a play that an average fielder should have made. Fielders can also be given errors if they make a poor play that
allows one or more runners to advance on the bases. A batter does not necessarily need to reach base for a fielder
to be given an error. If he drops a foul ball that extends an at-bat, that fielder can also be assessed an error.


OBP=(hit+walks+hit by pitcher)/(At bat+walks+hit by pither+sacrifice flies)
    OBP – On-base percentage: times reached base (H + BB + HBP) divided by at bats plus
    walks plus hit by pitch plus sacrifice flies (AB + BB + HBP + SF
BA – Batting average (also abbreviated AVG): hits divided by at bats (H/AB)
HR/H – Home runs per hit: home runs divided by total hits

When a home run is scored, the batter is also credited with a hit and a run scored,
    and an RBI for each runner that scores, including himself.

hits per run (H/R), also known as hit conversion rate[1] (HCR) is the ratio between
    hits and runs scored. It is the average number of hits it takes to score a run. H/R
    is the measure of the effectiveness of hitting in scoring a run. Teams having a lower
    hits-to-run ratio would likely have a good offense and could be expected to win more games.
"""




import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
import pickle
from helpers.data_prep import *
from helpers.eda import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


def load_hitters():
    data = pd.read_csv("datasets/hitters.csv")
    return data

df = load_hitters()
df.head()
df.isnull().sum()

#Single Season
#batting average (BA) Hits/Atbat
df["BA"]=df["Hits"]/df["AtBat"]
df

#HCR  hits per run (H/R), also known as hit conversion rate[1] (HCR)
df["HCR"]=df["Hits"]/df["Runs"]

#HR avarage homerun/AtBat
df["HR_AVG"]=df["HmRun"]/df["AtBat"]



#OBP=(hit+walks+hit by pitcher)/(At bat+walks+hit by pitcher+sacrifice flies)
df["OBP"]=(df["Hits"]+df["Walks"])/(df["AtBat"]+df["Walks"])
df.sort_values("Score_Contribution/Hit", ascending=False)
df.describe().T
df.drop(columns="Score_Contribution", inplace=True)
df["Score_Contribution/Hit"]=(df["Runs"]+df["RBI"]-df["HmRun"])/df["Hits"]
df["Score_Contribution/AtBat"]=(df["Runs"]+df["RBI"]-df["HmRun"])/df["AtBat"]


#Career
yearly_avg=["CAtBat", "CHits", "CHmRun", "CRuns", "CRBI", "CWalks"]
def c_avg(x):
    df["CAVG_"+x[1:]]=df[x]/df["Years"]
    df.head()

for x in yearly_avg:
    c_avg(x)


df

#batting average (BA) Hits/Atbat
df["CBA"]=df["CHits"]/df["CAtBat"]
df




#HCR  hits per run (H/R), also known as hit conversion rate[1] (HCR)
df["CHCR"]=df["CHits"]/df["CRuns"]

#HR avarage homerun/AtBat
df["CHR_AVG"]=df["CHmRun"]/df["CAtBat"]

df["value"]=df["OBP"]*df["Score_Contribution/AtBat"]
df["Cvalue"]=df["COBP"]*df["CScore_Contribution/AtBat"]
df["progress"]=df["value"]/df["Cvalue"]



def progression(dataframe):

#OBP=(hit+walks+hit by pitcher)/(At bat+walks+hit by pitcher+sacrifice flies)
df["COBP"]=(df["CHits"]+df["CWalks"])/(df["CAtBat"]+df["CWalks"])
df.sort_values("progress", ascending=False)
df.describe().T
df.drop(columns="Score_Contribution", inplace=True)
df["CScore_Contribution/Hit"]=(df["Runs"]+df["RBI"]-df["HmRun"])/df["Hits"]
df["CScore_Contribution/AtBat"]=(df["Runs"]+df["RBI"]-df["HmRun"])/df["AtBat"]













off=["Hits", "HmRun", "Runs", "RBI", "Walks"]
df["HIT_AVG"], df["HMRUN_AVG"], df["RUNS_AVG"], df["RBI_AVG"], df["WALKS_AVG"]=df[off].apply(lambda x: x*2)
df.head()
df = df[["Quantity", "Price", "TotalPrice"]]
liste = ["Quantity", "Price", "TotalPrice"]
df.columns
df.drop(["Quantity_AVG", "Price_AVG", "TotalPrice_AVG"], axis=1, inplace=True)
df["Quantity_AVG"], df["Price_AVG"], df["TotalPrice_AVG"] = df[liste].apply(lambda x: x.median() * 2)

def c_avg(x):
    df["CAVG_"+x[1:]]=df[x]/df["Years"]
    df.head()

for x in ca_avg:
    c_avg(x)

df.head()

off=["Hits", "HmRun", "Runs", "RBI", "Walks"]
def my_func(x):
    df[x+"_AVG"]=df[x]/df["AtBat"]
    return df

for x in off:
    my_func(x)

df.head()





from sklearn.preprocessing import StandardScaler
scaled=["AtBat","Hits","HmRun","Runs","RBI","Walks"]

for x in scaled:
    scaler = StandardScaler().fit(df[[x]])
    df["SCALED_"+x]=scaler.transform(df[[x]])



from sklearn.preprocessing import MinMaxScaler

for x in scaled:
    transformer = MinMaxScaler((1,10)).fit(df[[x]])
    df["SCALED_"+x]=transformer.transform(df[[x]])

transformer = MinMaxScaler().fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])

df["Age"].describe().T

df.describe().T
df.head()

df["COMP"]=df["SCALED_AtBat"]*df["SCALED_Hits"]*df["SCALED_HmRun"]*df["SCALED_Runs"]*df["SCALED_RBI"]*df["SCALED_Walks"]


df.sort_values("Salary", ascending=False)

scaler = StandardScaler().fit(df[["Hits_AVG"]])

df["SHits_AVG"] = scaler.transform(df[["Hits_AVG"]])
df["Age"].describe().T
df.head()


df["HIT_AVG"], df["HMRUN_AVG"], df["RUNS_AVG"], df["RBI_AVG"], df["WALKS_AVG"]=df[lambda x: for x in off   df[x]/df["AtBat"]]

df["HIT_AVG"], df["HMRUN_AVG"], df["RUNS_AVG"], df["RBI_AVG"], df["WALKS_AVG"]=df.apply[my_func(off)]
my_func(off)


df["HIT_AVG"], df["HMRUN_AVG"], df["RUNS_AVG"], df["RBI_AVG"], df["WALKS_AVG"]=df[off].apply[lambda x: df[x]/df["AtBat"]]

new_career_cols = [col for col in df.columns if (col.startswith ("C")) & (df[col].dtype != "O")]

df["Years_exc_86"]
career_avgs = [df[col] / df["Years_exc_86"] for col in new_career_cols]

exc_86_statistics = [df[row1] - df[row2] for (row1, row2) in zip (career_cols, cols_86)]

for (row1, row2) in zip (career_cols, exc_86_statistics):
    df[row1] = row2

df['p1'], df['p2'], df['p3'], df['p4'], df['p5'], df['p6'] = \
>>>     zip(*df['num'].map(powers))

def pc_avg(x):
    if df["Years"]==1:
        df["PCAVG_" + x[1:]] = df[x] / df["Years"]
    else:
        df["PCAVG_" + x[1:]] = df[x] /( df["Years"] - 1)

for x in ca_avg:
    pc_avg(x)

check_df(df)
ca_avg=["CAtBat","CHits", "CHmRun", "CRuns", "CRBI" ,"CWalks"]
df["CAVG_ATBAT", "CAVG_Hits", "CAVG_HmRun", "CAVG_Runs", "CAVG_RBI","CAVG_Walks"]=df[lambda x: x for x in c_avg (df[x]/df["Years"])]

for x in c_avg:
    df["CAVG_ATBAT", "CAVG_Hits", "CAVG_HmRun", "CAVG_Runs", "CAVG_RBI", "CAVG_Walks"]=df[x]/df["Years"]


df.head()



df = df[["Quantity", "Price", "TotalPrice"]]

liste = ["Quantity", "Price", "TotalPrice"]
df.columns

df.drop(["Quantity_AVG", "Price_AVG", "TotalPrice_AVG"], axis=1, inplace=True)
df["Quantity_AVG"], df["Price_AVG"], df["TotalPrice_AVG"] = df[liste].apply(lambda x: x.median() * 2)