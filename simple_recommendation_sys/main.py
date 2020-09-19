import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cols= ['user_id', 'item_id', 'rating', 'timestamp']
user_data = pd.read_csv('u.data', sep='\t', names=cols)
# print(user_data.head())


movie_titles = pd.read_csv("Movie_Id_Titles")
# print(movie_titles.head())

user_data_with_titles = pd.merge(user_data,movie_titles,on='item_id')
# print(user_data_with_titles.head())

movies_mean_rating = user_data_with_titles.groupby('title')['rating'].mean().sort_values()[::-1]
# print(movies_mean_rating.head())

movies_rating_count = user_data_with_titles.groupby('title')['rating'].count().sort_values()[::-1]
# print(movies_rating_count.head())

joined_data = pd.DataFrame(data={'title':movies_mean_rating.index.to_numpy(), 'rating': movies_mean_rating.values}).merge(pd.DataFrame(data={'title':movies_rating_count.index.to_numpy(), 'count': movies_rating_count.values}), on="title")
# print(joined_data.head())


joined_data['count'].hist()
# plt.show()

joined_data['rating'].hist()
# plt.show()

sns.jointplot(x='rating',y='count',data=joined_data,alpha=0.5)
# plt.show()

moviepv = user_data_with_titles.pivot_table(index='user_id',columns='title',values='rating')
# print(moviepv.head())

joined_data.sort_values('count')[::-1].head(10)
print(joined_data.head())

starwars_user_ratings = moviepv['Star Wars (1977)']
liarliar_user_ratings = moviepv['Liar Liar (1997)']
print(starwars_user_ratings.head())

similar_to_starwars = moviepv.corrwith(starwars_user_ratings)
similar_to_liarliar = moviepv.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
print(corr_starwars.head())

corr_starwars.sort_values('Correlation')[::-1].head(10)

corr_starwars = corr_starwars.join(joined_data['count'])
print(corr_starwars.head())

corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(joined_data['count'])
print(corr_liarliar[corr_liarliar['count']>100].sort_values('Correlation')[::-1].head())