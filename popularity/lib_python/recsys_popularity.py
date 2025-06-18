# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Installing Libraries
# MAGIC We install the **surprise** library, which is commonly used for building and analyzing recommender systems.

# COMMAND ----------

# MAGIC %pip install surprise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Importing Libraries

# COMMAND ----------

import requests
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC We import the necessary libraries:
# MAGIC
# MAGIC - **requests** for making API calls to fetch movie data.
# MAGIC - **pandas** for data manipulation and analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Defining Constants
# MAGIC We set the API key for accessing the movie database API.

# COMMAND ----------

API_KEY = 'd249f4616f6e86182d5e12fa3cb03a02'

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Function Definition

# COMMAND ----------

# MAGIC %md
# MAGIC We define and use the **get_movies** function to fetch movie data:
# MAGIC
# MAGIC - **API_KEY**: The API key used to authenticate requests to the movie database API.
# MAGIC - **get_movies Function**:
# MAGIC Fetches data from the movie database API for the specified number of pages.
# MAGIC Converts the fetched data into a pandas DataFrame.
# MAGIC Selects relevant columns: 'id', 'title', 'overview', 'vote_average', 'vote_count', 'popularity', and 'genre_ids'.

# COMMAND ----------

def get_movies(api_key, num_pages=500):
  movies = []
  for page in range(1, num_pages + 1):
    url = f'https://api.themoviedb.org/3/movie/popular?api_key={api_key}&language=en-US&page={page}'
    response = requests.get(url)
    data = response.json()
    movies.extend(data['results'])
  return movies

movies = get_movies(API_KEY)
movies_df = pd.DataFrame(movies)
movies_df = movies_df[['id', 'title', 'overview', 'vote_average', 'vote_count', 'popularity', 'genre_ids']]

# COMMAND ----------

# MAGIC %md
# MAGIC We define the **get_popular_movies** function, which:
# MAGIC
# MAGIC - Sorts the movies DataFrame by the 'popularity' column in descending order.
# MAGIC - Returns the top n most popular movies along with their titles and popularity scores.

# COMMAND ----------

def get_popular_movies(n=10):
  popular_movies = movies_df.sort_values(by='popularity', ascending=False)
  return popular_movies.head(n)[['title', 'popularity']]

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Printing Results

# COMMAND ----------

# MAGIC %md
# MAGIC We display the entire movies DataFrame, which contains data fetched from the movie database API.

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
# MAGIC - **requests**: A simple and elegant HTTP library for Python, built for making HTTP requests and interacting with web APIs. In this notebook, it is used to fetch data from the movie database API.
# MAGIC - **pandas**: A powerful Python library for data manipulation and analysis. It provides data structures and functions needed to manipulate structured data seamlessly. Here, it is used to handle and analyze the movie data fetched from the API.
# MAGIC - **surprise**: Although not used directly in the current code, the surprise library is typically used for building and evaluating recommender systems, particularly collaborative filtering models. It provides tools to work with explicit rating data, such as user-item matrices.