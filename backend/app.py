#content based filtering
import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer

# ast.literal_eval(expression)



movies=pd.read_csv('data/tmdb_5000_movies.csv')

credits=pd.read_csv('data/tmdb_5000_credits.csv')
# print(credits.head(1)['cast'].values)


# merge both dataframes
movies=movies.merge(credits,on='title')
# print(movies.shape)

# remove the uncessaary columns "gahan vichar"
# we need=> 'genre', 'id',keywords, title,overview,release_data,movie_id,cast,crew

#so update the dataframe with specified columns
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# print(movies.head(2))
#now we have to create a dataframe containing columns ="movie_id","title",and tags
#for tags merge overview,genre,keywords,cast,crew

# print(movies.isnull())#return a boolean table
# print(movies.isnull().sum())
#if any drop them
movies.dropna(inplace=True)
# check duplicate data
# print(movies.duplicated().sum())

def convert(text):#this function returns an array of keyword related to a film or array of cast in a film
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

movies['keywords'] = movies['keywords'].apply(convert)
# print(movies.head())
# print(movies["keywords"])

ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 

movies['cast'] = movies['cast'].apply(convert3)# so now the cast column will only have  name of the actors
# print(movies['cast'])
movies['genres'] = movies['genres'].apply(convert3)
# print(movies['genres'])

# movies['cast'] = movies['cast'].apply(lambda x:x[0:3])#from the array of cast of a film we extracted first three members
# print(movies['cast'])#instead of this we have used convert3 function

def fetch_director(text):
    L = []# beacuse it may have array of directors
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 
movies['crew'] = movies['crew'].apply(fetch_director)# in crew column apply the fetch fucntion to each element in the columns
# print(movies['crew'])

movies['overview'] = movies['overview'].apply(lambda x:x.split())
# print(movies['overview'].head())


# print(movies.sample(5))

# now code below will convert the array into a string and remove', ' with space
# movies['cast'] = movies['cast'].apply(lambda x: ' '.join(x))
# movies['crew'] = movies['crew'].apply(lambda x: ' '.join(x))
# movies['keywords'] = movies['keywords'].apply(lambda x: ' '.join(x))
# print(movies['cast'])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(' ','') for i in x])
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(' ','') for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(' ','') for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(' ','') for i in x])

# print(movies['overview'].head())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
# print(new.head())
new['tags'] = new['tags'].apply(lambda x: " ".join(x))


#lowercase tags
new['tags'] = new['tags'].apply(lambda x:x.lower())
print(new['tags'].head())

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
print(vectors)