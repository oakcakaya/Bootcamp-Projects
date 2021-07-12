
#############################################
# PROJECT: Hybrid Recommender System
#############################################

# Construct a Recommendation system for a given user ID by using item-based and user-based recommender methods.

# 5 recommendations from user-based model and 5 recommendations from item-based model, total of 10 recommendations.
# user_id = 108170

#############################################
# Step 1: Data Preperation
#############################################
import pandas as pd
pd.set_option('display.max_columns', 30)

movie = pd.read_csv('dsmlbc4/datasets/the movies//movie.csv')
rating = pd.read_csv('dsmlbc4/datasets/the movies//rating.csv')
df_ = movie.merge(rating, how="left", on="movieId")
df=df_.copy()
df.head()

#################
# title
#################

df['year_movie'] = df.title.str.extract('(\(\d\d\d\d\))', expand=False)
df['year_movie'] = df.year_movie.str.extract('(\d\d\d\d)', expand=False)
df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
df['title'] = df['title'].apply(lambda x: x.strip())

df.shape
#(20000797, 7)
df.head()

#################
# genres
#################

df["genre"] = df["genres"].apply(lambda x: x.split("|")[0])

df.drop("genres", inplace=True, axis=1)
df.head()

#################
# timestamp
#################

df.info()

df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d')
df.info()

df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df.head()


a = pd.DataFrame(df["title"].value_counts())
#26213 total movies
rare_movies = a[a["title"] <= 1000].index
rare_movies.nunique()
#23079 number of rare movies that are rated less than 1000 times
common_movies = df[~df["title"].isin(rare_movies)]
common_movies["movieId"].nunique()
#3507 number of common movies that are rated mote than 1000 times


#Pivot table of movie rating vs userId
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df



#############################################
# Step 2: Determination of movies watched by intended user
#############################################
#user_id = 108170 is the intended user

random_user_df = user_movie_df[user_movie_df.index == 108170]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

len(movies_watched)
#user_id = 108170 watched 186 movies so far.



#############################################
# Adım 3: Fetching the IDs of other users watched the same movies as  user_id = 108170
#############################################


movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape
#(138493, 186)

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

#Selecting the 60% of same movies watched
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

users_same_movies.nunique()


#############################################
# Step 4: Determining the similar users
#############################################

# Will be carried out in  3  steps:
# 1. Combining the intended user and similar users data.
# 2. Formation of Correlation df.
# 3. Finding out (Top Users).


final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      random_user_df[movies_watched]])

final_df.head()

final_df.T.corr()

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()


top_users = corr_df[(corr_df["user_id_1"] == 108170) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users


top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings


#############################################
# Step 5: Calculation of Weighted rating
#############################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()


#############################################
# Step 6: Calculation of Weighted average recommendation score
#############################################

temp = top_users_ratings.groupby('movieId').sum()[['corr', 'weighted_rating']]
temp.columns = ['sum_corr', 'sum_weighted_rating']

temp.head()

userbased_recommendation_df = pd.DataFrame()
userbased_recommendation_df['weighted_average_recommendation_score'] = temp['sum_weighted_rating'] / temp['sum_corr']
userbased_recommendation_df['movieId'] = temp.index
userbased_recommendation_df = userbased_recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)
userbased_recommendation_df.head(10)



movie.loc[movie['movieId'].isin(userbased_recommendation_df.head(10)['movieId'])]

movies_from_user_based=movie.loc[movie['movieId'].isin(userbased_recommendation_df.head(5)['movieId'])][["title"]]
movies_from_user_based["title"]
"""
891                      North by Northwest (1959)
937            Mr. Smith Goes to Washington (1939)
952                      African Queen, The (1951)
3986                              Baby Boom (1987)
5872  My Neighbor Totoro (Tonari no Totoro) (1988)"""
#############################################
# Adım 7: Recommend 5 movies with respect to the last 5.0 rated movie by user 108170
#############################################
movie.head()
movie_id = rating[(rating["userId"] == 108170) & (rating["rating"] == 5.0)]. \
    sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
movie_id
#ID of the last 5-rated movie: 7044
movie_title = movie[movie["movieId"] == movie_id]["title"].str.replace('(\(\d\d\d\d\))', '').str.strip().values[0]

movie_title
#Name of the last 5-rated movie: Wild at Heart

movie = user_movie_df[movie_title]

movies_from_item_based=user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

movies_from_item_based[1:6].index
#(['My Science Project', 'Mediterraneo', 'National Lampoon's Senior Trip','Old Man and the Sea, The', 'Cashback'], \
# dtype='object', name='title')


