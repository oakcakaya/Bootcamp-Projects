
###################################################
# PROJECT: Rating Product & Sorting Reviews in Amazon
###################################################

# http://jmcauley.ucsd.edu/data/amazon/links.html

# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# asin - ID of the product, e.g. 0000013714
# reviewerName - name of the reviewer
# helpful - helpfulness rating of the review, e.g. 2/3
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)


import pandas as pd
import math
import scipy.stats as st
import datetime as dt

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)






###################################################
# Task 1: Calculate the actual rating according to the up to date reviews and compare to previous rating.
###################################################

###################################################
# Step 1. Read the datset df_sub_csv
###################################################
df_=pd.read_csv("dsmlbc4/datasets/df_sub.csv")
df_sub=df_.copy()
df_sub.head()
df_sub.isnull().any()
df_sub.shape
df_sub['asin'].value_counts()

# http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics.json.gz
"""
#Download the dataset below and use the codes to read the data
import pandas as pd
import gzip

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


df_ = get_df('datasets/reviews_Electronics_5.json.gz')
df = df_.copy()
df.head()
df.shape

###################################################
# Step 1.2 Find the most reviewed item in the data set.
###################################################

# 1. Find the most sold item in SQL and reduce the data before transfering to the pycharm.
# 2. Most sold item by pandas....


###################################################
# Step 1.3. Reduce the dataset according to the most reviewed item.
###################################################
"""
###################################################
# Step 2. What is the average grade of the item?
###################################################
df_sub.head()
df_sub["overall"].mean()
# Avg_grade=4.587589013224822


###################################################
# Step 3. Calculate weighted average according to time.
###################################################

# day_diff calculation: (days passed after the review)
df_sub['reviewTime'] = pd.to_datetime(df_sub['reviewTime'], dayfirst=True)
current_date = pd.to_datetime('2014-12-08 0:0:0')
df_sub["day_diff"] = (current_date - df_sub['reviewTime']).dt.days

# Zamanı çeyrek değerlere göre bölüyorum.
a = df_sub["day_diff"].quantile(0.25)
b = df_sub["day_diff"].quantile(0.50)
c = df_sub["day_diff"].quantile(0.75)
df_sub.head()
df_sub["day_diff"].describe()
###################################################
# Step 4. Calculate weighted average with respect to day_diff calculated in previous step.
###################################################
df_sub.loc[df_sub["day_diff"]<=a, "overall"].mean()*28/100 + \
    df_sub.loc[(df_sub["day_diff"] >a) & (df_sub["day_diff"] <= b), "overall"].mean()*26/100+ \
    df_sub.loc[(df_sub["day_diff"] >b) & (df_sub["day_diff"] <= c), "overall"].mean()*24/100+ \
    df_sub.loc[(df_sub["day_diff"] >c), "overall"].mean()*22/100

#Weighted Avg according to time: 4.601228735299981
# Weights <25=28%, 25<W<50=26%, 50<W<75=24%, 75<W=20%


#Previous Avg_grade=4.587589013224822
#Weighted Avg_grade=4.601228735299981


###################################################
# Task 2: Determine the top 20 reviews to be presented in promotion page
###################################################

###################################################
# Step 1. Derive 3 Variables from "Helpful" variable 1: helpful_yes, 2: helpful_no,  3: total_vote
###################################################

# Two values exist in "Helpful". Former is the ones that find the review helpful and the latter is the total votes.
# First, two values need to be recorded sperately and then by (total_vote - helpful_yes)  'helpful_no' should be calculated

df_sub.head(100)




split_df= df_sub["helpful"].str.split(",", expand=True)

split_df = split_df.astype("string")
helpful_yes = split_df[0].str.lstrip("[")
helpful_yes = helpful_yes.astype("int")

df_sub["yes_alt"]=df_sub["helpful"].apply(lambda x: x[0])
df_sub.head()

df["helpful_yes"] = [row[row.index('[') + 1: row.index(',')] for row in df["helpful"]]
df["total_vote"] = [row[row.index(',') + 2: row.index(']')] for row in df["helpful"]]

df_sub['helpful_yes'] = df_sub[['helpful']].applymap(lambda x : x.split(', ')[0].strip('[')).astype(int)
df_sub['total_vote'] = df_sub[['helpful']].applymap(lambda x : x.split(', ')[1].strip(']')).astype(int)


total_vote = split_df[1].str.rstrip("]")
total_vote = total_vote.astype("int")

helpful_no= total_vote- helpful_yes


df_sub["helpful_yes"]=helpful_yes
df_sub["helpful_no"]=helpful_no
df_sub["total_vote"]=total_vote

df_sub.head(20)

###################################################
# Step 2 Derive score with respect to  score_pos_neg_diff and order scores
###################################################
def score_pos_neg_diff(pos, neg):
    return pos - neg

df_sub["score_pos_neg_diff"]=df_sub.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],x["helpful_no"]),axis=1)

df_sub.sort_values("score_pos_neg_diff",ascending=False)

###################################################
# Step 3. Derive score with respect to score_average_rating.
###################################################
def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)

df_sub["score_average_rating"]=df_sub.apply(lambda x: score_average_rating(x["helpful_yes"],x["helpful_no"]),axis=1)
df_sub.sort_values("score_average_rating",ascending=False)


##################################################
# Step 4 . Derive scores with respect to wilson_lower_bound.
###################################################
def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score

    - Lower boundary of confidence interval calculated by Bernoulli paremater p is acceptedas WLB score.
    - The score calculated is used for product order.
    - Note: If the scores are between  1-5:   socores betwween 1-3 are marked as  down, and scores 4-5 are marked as up
    by this way the scores are meade compatible with ve bernoulli.

    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df_sub["wilson_lower_bound"]=df_sub.apply(lambda x: wilson_lower_bound(x["helpful_yes"],x["helpful_no"]),axis=1)
df_sub.sort_values("wilson_lower_bound",ascending=False)





##################################################
# Step 5. Determine the top 20 reviews that are to be presented in the promotion page and discuss the results
###################################################
df_sub[["reviewerID", "reviewerName","reviewText","total_vote","score_pos_neg_diff","score_average_rating","wilson_lower_bound"]].\
    sort_values("wilson_lower_bound",ascending=False).head(20)

#   To determine the top 20 reviews, Wilson Lower Bound is used because:
#   Positive-Negative difference just gives the difference between positives and negatives, it does not concern
# amount of the total votes.
#   Score Average rating only concerns positive to total votes ratio. It creat bias when lower amount of voted are
# examined. ie total vote=1 Helpful=1 gives a rating of 1.
#   Wilson lower Bound takes bot total vote and score average rating into account. Also it gives us a 95% confidence
# interval. To sum up, high scored  reviews calculated by using Wilson Lower Bound, are the best to
# present on promotion page.

df_sub[["reviewerID", "reviewerName","reviewText","total_vote","score_pos_neg_diff","score_average_rating","wilson_lower_bound"]].\
    sort_values("total_vote",ascending=False).head(20)