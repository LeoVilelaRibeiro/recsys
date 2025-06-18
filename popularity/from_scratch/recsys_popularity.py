# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Importing Libraries

# COMMAND ----------

import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# COMMAND ----------

# MAGIC %md
# MAGIC We import the necessary libraries:
# MAGIC
# MAGIC - **requests** for making API calls to fetch movie data.
# MAGIC - **pandas** for data manipulation.
# MAGIC - **TfidfVectorizer** and **cosine_similarity** from **sklearn** for text processing and calculating similarities.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Defining Constants
# MAGIC We set the API key for accessing the movie database API.

# COMMAND ----------

API_KEY = 'd249f4616f6e86182d5e12fa3cb03a02'

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Fetching Data
# MAGIC We fetch the movies using the **get_movies** function (defined later), convert the data into a pandas DataFrame, and select relevant columns.

# COMMAND ----------

movies = get_movies(API_KEY)
movies_df = pd.DataFrame(movies)
movies_df = movies_df[['id', 'title', 'overview', 'vote_average', 'vote_count', 'popularity', 'genre_ids']]

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Function Definition

# COMMAND ----------

# MAGIC %md
# MAGIC We define the **get_movies** function, which:
# MAGIC
# MAGIC - Takes the API key and the number of pages to fetch as input.
# MAGIC - Iterates over the specified number of pages to fetch movie data from the API.
# MAGIC - Collects the results and returns a list of movies.

# COMMAND ----------

def get_movies(api_key, num_pages=500):
  movies = []
  for page in range(1, num_pages + 1):
    url = f'https://api.themoviedb.org/3/movie/popular?api_key={api_key}&language=en-US&page={page}'
    response = requests.get(url)
    data = response.json()
    movies.extend(data['results'])
  return movies

# COMMAND ----------

# MAGIC %md
# MAGIC We define the **get_popular_movies** function, which:
# MAGIC
# MAGIC - Sorts the movies DataFrame by popularity in descending order.
# MAGIC - Returns the top **n** most popular movies along with their titles and popularity scores.

# COMMAND ----------

def get_popular_movies(n=10):
  popular_movies = movies_df.sort_values(by='popularity', ascending=False)
  return popular_movies.head(n)[['title', 'popularity']]

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Printing Results

# COMMAND ----------

# MAGIC %md
# MAGIC We display the entire movies DataFrame.

# COMMAND ----------

movies_df

# COMMAND ----------

# MAGIC %md
# MAGIC We call the **get_popular_movies** function to print the top 10 most popular movies.

# COMMAND ----------

get_popular_movies()

# COMMAND ----------

# MAGIC %md
# MAGIC **Libraries Overview:**
# MAGIC - **requests**: A simple and elegant HTTP library for Python, built for making HTTP requests and interacting with web APIs.
# MAGIC - **pandas**: A powerful Python library for data manipulation and analysis. It provides data structures and functions needed to manipulate structured data seamlessly.
# MAGIC - **scikit-learn (sklearn)**: A machine learning library for Python that provides simple and efficient tools for data mining and data analysis. In this notebook, we use TfidfVectorizer to transform text data into TF-IDF features and cosine_similarity to calculate similarities between vectors.