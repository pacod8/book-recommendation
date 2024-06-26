import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_option('deprecation.showPyplotGlobalUse', False)

books = pd.read_csv('data/Books.csv')
ratings = pd.read_csv('data/Ratings.csv')
users = pd.read_csv('data/Users.csv')



st.title("Book Recommendation System")
st.subheader("Francisco Dominguez")
books.isna().sum()
# books[books['Book-Author'].isna()]
# books.iloc[187689]['Book-Title']
books.iloc[187689]['Book-Author'] = 'Downes, Larissa Anne'
users.isna().sum()
users.drop(columns=['Age'], inplace=True)
ratings.isnull().sum()
books.duplicated().sum()
users.duplicated().sum()
ratings.duplicated().sum()
books.head(10)

books.iloc[209538]['Book-Author'] = 'Michael Teitelbbaum'
books.iloc[209538][
    'Book-Title'] = 'DK Readers: The Story of the X-Men, How It All Began (Level 4: Proficient Readers)'
books.iloc[209538]['Year-Of-Publication'] = 2000
books.iloc[209538]['Publisher'] = 'DK Publishing Inc'

books.iloc[220731]['Book-Title'] = "Peuple du ciel, suivi de 'Les Bergers'"
books.iloc[220731]['Book-Author'] = 'Jean-Marie Gustave Le Clézio'
books.iloc[220731]['Year-Of-Publication'] = 1990
books.iloc[220731]['Publisher'] = 'Gallimard'

books.iloc[221678][
    'Book-Title'] = 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)'
books.iloc[221678]['Book-Author'] = 'James Buckley'
books.iloc[221678]['Year-Of-Publication'] = 2000
books.iloc[221678]['Publisher'] = 'DK Publishing Inc'

books['Year-Of-Publication'] = books['Year-Of-Publication'].astype('int64')
#books['Year-Of-Publication'].value_counts().sort_index(ascending=False).iloc[:20]

#books[books['Year-Of-Publication']>2021][['Book-Title','Year-Of-Publication','Publisher','Book-Author']]

#'MY TEACHER FRIED MY BRAINS (RACK SIZE) (MY TEACHER BOOKS)'
books.loc[37487, 'Year-Of-Publication'] = 1991

# 'MY TEACHER FLUNKED THE PLANET (RACK SIZE) (MY TEACHER BOOKS)'
books.loc[55676, 'Year-Of-Publication'] = 2005

# 'MY TEACHER FLUNKED THE PLANET (RACK SIZE) (MY TEACHER BOOKS)'
books.loc[37487, 'Book-Author'] = 'Bruce Coville'

# "Alice's Adventures in Wonderland and Through the Looking Glass (Puffin Books)"
books.loc[80264, 'Year-Of-Publication'] = 2003

# 'Field Guide to the Birds of North America, 3rd Ed.'
books.loc[192993, 'Year-Of-Publication'] = 2003

# Crossing America
books.loc[78168, 'Year-Of-Publication'] = 2001

# Outline of European Architecture (Pelican S.)
books.loc[97826, 'Year-Of-Publication'] = 1981

# Three Plays of Eugene Oneill
books.loc[116053, 'Year-Of-Publication'] = 1995

# Setting to current date of project since no information could be found
# Das groÃ?Â?e BÃ?Â¶se- MÃ?Â¤dchen- Lesebuch.
books.loc[118294, 'Year-Of-Publication'] = 2023

# FOREST PEOPLE (Touchstone Books (Hardcover))
books.loc[228173, 'Year-Of-Publication'] = 1987

# In Our Time: Stories (Scribner Classic)
books.loc[240169, 'Year-Of-Publication'] = 1996

# CLOUT
books.loc[246842, 'Year-Of-Publication'] = 1925

# To Have and Have Not
books.loc[255409, 'Year-Of-Publication'] = 1937

# FOOTBALL SUPER TEAMS : FOOTBALL SUPER TEAMS
books.loc[260974, 'Year-Of-Publication'] = 1991



bookRating = pd.merge(ratings, books, on="ISBN")
bookRating.head()
bookRating.drop(columns=['Image-URL-S','Image-URL-M','Image-URL-L'],inplace=True)
averageRating = pd.DataFrame(bookRating.groupby('ISBN')['Book-Rating'].mean().round(1))
averageRating.reset_index(inplace=True)
averageRating.head()

#averageRating.shape
averageRating.rename(columns={'Book-Rating':'Average-Rating'}, inplace=True)
averageRating.head()

averageRatingdf = pd.merge(bookRating, averageRating, on='ISBN')
averageRatingdf.head()

averageRatingOnly = averageRatingdf[['ISBN','Average-Rating']]
averageRatingOnly.head()

averageRatingUnique = averageRatingOnly[['ISBN','Average-Rating']].drop_duplicates(subset=['ISBN'])
averageRatingUnique.head()

ratingBooks = pd.merge(books, averageRatingUnique, on='ISBN', how='inner')


books_with_rating = pd.merge(books, averageRatingUnique, on='ISBN')
#books_with_rating.shape

books_with_rating = books_with_rating[['ISBN','Book-Title','Book-Author','Average-Rating','Year-Of-Publication','Publisher','Image-URL-S','Image-URL-M','Image-URL-L']]
books_with_rating.head()

books_with_rating.sort_values(by=['Average-Rating'], ascending=False).head(30)

ratings_sorted = books_with_rating['Average-Rating'].value_counts().sort_index(ascending=False)
#display(ratings_sorted)
books_with_rating['Average-Rating'].value_counts(normalize=True).round(4).sort_index(ascending=False)



ratings_books_merged = ratings.merge(books, on='ISBN')
#display(ratings_books_merged.head())
#ratings_books_merged.shape

ratings_books_nonzero = ratings_books_merged[ratings_books_merged['Book-Rating']!=0]
num_rating_df = ratings_books_nonzero.groupby('Book-Title').count()['Book-Rating'].sort_values(ascending=False).reset_index()
num_rating_df.rename(columns={'Book-Rating':'Number-of-Ratings'}, inplace=True)
#display(num_rating_df)

avg_rating_df = ratings_books_nonzero.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'Average-Rating'}, inplace=True)
avg_rating_df.head()

popularity_df = pd.merge(num_rating_df, avg_rating_df, on='Book-Title')
#popularity_df

popularity_df_above_100 = popularity_df[popularity_df['Number-of-Ratings']>=100]
popularity_df_above_50 = popularity_df[popularity_df['Number-of-Ratings'] >= 50]
popularity_df_above_250 = popularity_df[popularity_df['Number-of-Ratings'] >= 250]
popularity_df_above_100.sort_values(by='Number-of-Ratings', ascending=False).head()

# Defining a new function that can calculate the metric
def calcWeightedRating(row, avgRating, numOfRatings, minThres, defRating):
    weightedRating = ((row[avgRating] * row[numOfRatings]) + (minThres * defRating))/(row[numOfRatings] + minThres)
    return weightedRating

# For number of ratings above 100
popularity_df_above_100 = popularity_df_above_100.copy()
popularity_df_above_100['Weighted-Rating'] = popularity_df_above_100.apply(lambda x: calcWeightedRating(
     x, 'Average-Rating', 'Number-of-Ratings', 100, 5),axis=1)
popularity_df_above_100.sort_values(
    'Weighted-Rating', ascending=False).head(20)

# For number of ratings above 50
popularity_df_above_50 = popularity_df_above_50.copy()
popularity_df_above_50['Weighted-Rating'] = popularity_df_above_50.apply(lambda x: calcWeightedRating(
    x, 'Average-Rating', 'Number-of-Ratings', 50, 5), axis=1)
popularity_df_above_50.sort_values(
    'Weighted-Rating', ascending=False).head(20)

# For number of ratings above 250
popularity_df_above_250 = popularity_df_above_250.copy()
popularity_df_above_250['Weighted-Rating'] = popularity_df_above_250.apply(lambda x: calcWeightedRating(
    x, 'Average-Rating', 'Number-of-Ratings', 250, 5), axis=1)
popularity_df_above_250.sort_values(
    'Weighted-Rating', ascending=False).head(20)

popular_df_merge = pd.merge(popularity_df_above_100, books, on='Book-Title').drop_duplicates('Book-Title',keep='first')
popular_df_merge = popular_df_merge.drop(columns=['Image-URL-S', 'Image-URL-L'])
#display(popular_df_merge.shape)
popular_df_merge.sort_values('Weighted-Rating', ascending=False).head(10)

users_ratings_count = ratings_books_merged.groupby('User-ID').count()['ISBN']
users_ratings_count = users_ratings_count.sort_values(ascending=False).reset_index()
users_ratings_count.rename(columns={'ISBN':'No-of-Books-Rated'}, inplace=True)
#display(users_ratings_count.shape)
users_ratings_count.head()

users_200 = users_ratings_count[users_ratings_count['No-of-Books-Rated']>=200]
#display(users_200.shape)

books_with_users_200 = pd.merge(users_200, ratings_books_merged, on='User-ID')
#display(books_with_users_200.shape)
books_with_users_200.head()

#display(ratings_books_merged.shape)
ratings_books_merged.head()

books_ratings_count = ratings_books_merged.groupby('Book-Title').count()['ISBN'].sort_values(ascending=False).reset_index()
books_ratings_count.rename(columns={'ISBN':'Number-of-Book-Ratings'}, inplace=True)
books_ratings_count.head()

books_ratings_50 = books_ratings_count[books_ratings_count['Number-of-Book-Ratings']>=50]
#display(books_ratings_50.shape)
#books_ratings_50.head()
filtered_books = pd.merge(books_ratings_50, books_with_users_200,  on='Book-Title')
#display(filtered_books.shape)
#filtered_books.head()

famous_books = filtered_books.groupby('Book-Title').count().reset_index()
famous_books = famous_books['Book-Title']
famous_books = books[books['Book-Title'].isin(famous_books)]
famous_books = famous_books.copy()
famous_books.drop_duplicates(subset=['Book-Title'], inplace=True, keep='first')
famous_top = filtered_books.groupby('Book-Title').max().reset_index()[filtered_books['Book-Rating']==10]

pt = filtered_books.pivot_table(index='Book-Title',columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)
#pt


similarities = cosine_similarity(pt)
#similarities

def recommend(book_name):
    if book_name in pt.index:
        index = np.where(pt.index == book_name)[0][0]
        similar_books_list = sorted(
        list(enumerate(similarities[index])), key=lambda x: x[1], reverse=True)[1:11]
        
        st.write('-'*5)
        st.write(f'**Recommendations for the book {book_name}:**')
        for book in similar_books_list[:5]:
            st.write(pt.index[book[0]])
        print('\n')

    else:
        st.write('Book Not Found')
        st.write('\n')



st.markdown("# Recommend a book")
selected_book = st.selectbox(
   "Select a book",
   (filtered_books['Book-Title'].tolist())
)
st.write(selected_book)
recommend(selected_book)

st.markdown("# Top Books")
# Configurar el número de columnas en el grid
num_columns = 8  # Número de columnas en el grid
rows = (len(famous_top) + num_columns - 1) // num_columns  # Calcular el número de filas


# Crear el grid de tarjetas
for i in range(rows):
    cols = st.columns(num_columns)  # Crear una fila con el número especificado de columnas
    for j in range(num_columns):
        index = i * num_columns + j
        if index < len(famous_top):  # Comprobar que el índice está dentro del rango del DataFrame
            with cols[j]:
                st.write(f"**{famous_top.iloc[index, famous_top.columns.get_loc('Book-Title')]}**")
                st.write(f"Author: {famous_top.iloc[index, famous_top.columns.get_loc('Book-Author')]}")
                st.write(f"Rating: {famous_top.iloc[index, famous_top.columns.get_loc('Book-Rating')]}")
                st.image(f"{famous_top.iloc[index, famous_top.columns.get_loc('Image-URL-S')]}")
                st.markdown("---")  # Línea separadora entre tarjetas

