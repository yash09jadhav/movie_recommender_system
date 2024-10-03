#!/usr/bin/env python
# coding: utf-8

# - recommender systems are of 3 types - **content based**, **collaboration filtering**, **hybrid**
# - content based focus purely on content features
# - col. fil. is based on assumptions, suppose A & B are friends, then movies liked by A can be recommended to B

# In[4]:


import numpy as np
import pandas as pd


# ### 1. data collection
# 

# In[5]:


credits = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Desktop\ds\datasets\tmdb_5000_credits.csv")
movies = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Desktop\ds\datasets\tmdb_5000_movies.csv")
print(credits.shape, movies.shape)


# In[6]:


movies.sample(1)


# In[7]:


credits.sample(1)


# ### 2. preprocessing

# In[8]:


movies = movies.merge(credits, on='title')
movies.shape


# - instead of 2 tables, we'll use 1 by joining them along a common col
# - feature extraction - based on info gain

# In[9]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# - missing values

# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)
movies.isnull().sum()


# - duplicate values

# In[12]:


movies.duplicated().sum()


# - cleaning

# In[13]:


movies.iloc[0].genres


# - this is a str, needs to be converted to list by ast.literal_eval()

# In[14]:


def convert(l):
    genre=[]
    for i in ast.literal_eval(l):
        genre.append(i['name'])
    return genre


# In[15]:


import ast


# In[16]:


movies['genres'] = movies['genres'].apply(convert)
movies.sample(1)


# - doing the same on keywords

# In[17]:


movies.iloc[0].keywords


# In[18]:


def convert_keywords(l):
    key_words = []
    for i in ast.literal_eval(l):
        key_words.append(i['name'])
    return key_words


# In[19]:


movies['keywords'] = movies['keywords'].apply(convert_keywords)
movies.sample(1)


# - for cast col, we'll select first 3 actors/actresses

# In[20]:


movies.iloc[0].cast


# In[21]:


def convert_cast(l):
    cast = []
    counter = 0
    for i in ast.literal_eval(l):
        if counter < 3:
            cast.append(i['name'])
            counter += 1
        else:
            break
    return cast


# In[22]:


movies['cast'] = movies['cast'].apply(convert_cast)
movies.sample(1)


# In[23]:


def convert_crew(l):
    crew = []
    for i in ast.literal_eval(l):
        if i['job'] == 'Director' :
            crew.append(i['name'])
    return crew


# In[24]:


movies['director'] = movies['crew'].apply(convert_crew)
movies.drop(columns = ['crew'], inplace=True)
movies.sample(1)


# - converting overview as str to list i.e same format as others

# In[25]:


movies['overview'] = movies['overview'].apply(lambda x: x.split())


# - we wish to compare similar movies using similar tags which shouldnt have spaces in btw them

# In[26]:


def collapse(l):
    L=[]
    for i in l:
        L.append(i.replace(' ','_'))
    return L


# In[27]:


movies['cast'] = movies['cast'].apply(collapse)
movies['director'] = movies['director'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['overview'] = movies['overview'].apply(collapse)


# In[28]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['director']


# In[29]:


movies = movies.drop(columns=['overview','genres','keywords','cast','director'])


# - list to str

# In[30]:


movies['tags'] = movies['tags'].apply(lambda x: ' '.join(x))


# - recommended to convert all in lowercase

# In[31]:


movies['tags'] = movies['tags'].apply(lambda x: x.lower())


# In[32]:


movies.iloc[0].tags


# In[33]:


movies.sample(1)


# ### 3. text vectorization
# - converts each tag into a vectors then will give out the similar vector to it based on its proximity
# - we'll use bag of words technique, which adds all the tags together & creates a matrix of top 5000 common words
# - in which each row is a vector

# In[34]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english', max_features=5000)
vectors = cv.fit_transform(movies['tags']).toarray()


# In[35]:


cv.get_feature_names_out()


# - observation : too many words like love, loving, lover
# - whose meaning here is same as 'love'
# - so importing nltk

# In[36]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return ' '.join(y)


# In[37]:


stem('he loving you, loved you, love you')


# In[38]:


movies['tags'] = movies['tags'].apply(stem)


# - run cv object again
# - now we'll use cosine similarity (measuring angle btw 2 vectors) rather than euclidean distance as the later fails in large datasets

# In[39]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
similarity


# ### 4. model creation
# - we'll create a function which'll take 1 movie & will return 5 similar movies
# - it'll compare its similarity array with every other, sort it & will return the top 5

# In[40]:


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(movies.iloc[i[0]].title)


# In[41]:


recommend('Avatar')




