# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Installing Libraries
# MAGIC First, we install the **openpyxl** library, which is required to read Excel files.

# COMMAND ----------

# MAGIC %pip install openpyxl

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Fetching Data
# MAGIC We load the movies and ratings data from the specified Excel files into pandas DataFrames.

# COMMAND ----------

movies=pd.read_excel('/Workspace/Users/leonardo.ribeiro@omni.chat/collaborative_filtering_system/filtering_system_scratch/data/movies.xlsx')
ratings=pd.read_excel('/Workspace/Users/leonardo.ribeiro@omni.chat/collaborative_filtering_system/filtering_system_scratch/data/ratings.xlsx',sheet_name="ratings")

# COMMAND ----------

# MAGIC %md
# MAGIC We display the first 20 records of the movies DataFrame to inspect the data.

# COMMAND ----------

movies.head(20)

# COMMAND ----------

#the column 1 consist of user ids which have rated the movies present in data set
#movie id connects movies with userId

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Data Manipulation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Merging Data
# MAGIC We merge the movies and ratings DataFrames to create a single DataFrame that includes movie titles and ratings. We also rename the 'title' column to 'Movies' for clarity.

# COMMAND ----------

#merging the two dataframes together
ratings=pd.merge(movies,ratings)
ratings=ratings.iloc[:,[0,1,-2,-1]]
ratings = ratings.rename(columns={'title': 'Movies'})
ratings

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Creating Pivot Table
# MAGIC We create a pivot table with **userId** as the index, **Movies** as the columns, and **rating** as the values. This format is required for calculating the correlation between movies.

# COMMAND ----------

#making a pivot table similar to excel 
#to create a correlation table we have to convert the data to this format
user_ratings=ratings.pivot_table(index=['userId'],columns=['Movies'],values='rating')
user_ratings

# COMMAND ----------

#many nan values
#we have to drop some values as many movies are rated by less than 10 userids which may create noise in data set

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Dropping NaN Values
# MAGIC We note the presence of many NaN values and mention the need to drop movies that are rated by fewer than 10 users to reduce noise.

# COMMAND ----------

#dropping The movies which are rated by less than 10 people
#replacing NAN with 0
#we could have standardize the value here but we didnt do that beacuse there were many 0s
user_ratings=user_ratings.dropna(thresh=10,axis=1).fillna(0)
user_ratings

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Using Correlation
# MAGIC We explain that we are calculating the correlation between movies based on user ratings using Pearson's correlation method.

# COMMAND ----------

##using correlation 
#here we are observing the correlation between the movies based on ratings of various users
#we are using pearson's method of correlation
item_similarity=user_ratings.corr(method='pearson')
item_similarity

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Declaring Functions
# MAGIC We introduce the function **get_similar_movies**, which will generate movie recommendations based on item similarity scores.

# COMMAND ----------

#to get recommendation based on items correlation on ratings

#WE ARE DEFINING A FUNCTION WHICH WILL SORT THE VALUES OF CORRELATED VALUES AND WILL SHOW US THE VALUES
def get_similar_movies(movie_name,user_rating):
    similar_score = item_similarity[movie_name]*(user_rating-2.5)
    similar_score = similar_score.sort_values(ascending=False)
    
    return similar_score

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Usage Example
# MAGIC We provide an example of a user who likes action movies. We use the movie "Iron Man (2008)" with a rating of 5 to find similar movies.

# COMMAND ----------

#a user who has rated these movies the user is action lover
#we can add new values to the tuple\
p='Iron Man (2008)'
q=5

# COMMAND ----------

# MAGIC %md
# MAGIC We find similar movies for the action lover by using the get_similar_movies function and display the top 10 similar movies, and finally we display the top 20 similar movies based on the summed similarity scores.

# COMMAND ----------

action_lover = [(p,q)]
similar_movies = pd.DataFrame()
for movie,rating in action_lover:
    similar_movies = similar_movies.append(get_similar_movies(movie,rating),ignore_index = True)

similar_movies.head(10)
similar_movies.sum().sort_values(ascending=False).head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC Libraries Overview:
# MAGIC - **pandas**: A powerful Python library for data manipulation and analysis. It provides data structures and functions needed to manipulate structured data seamlessly.
# MAGIC - **numpy**: A fundamental package for scientific computing with Python. It provides support for arrays, matrices, and many mathematical functions.
# MAGIC - **openpyxl**: A Python library to read/write Excel 2010 xlsx/xlsm/xltx/xltm files. It is useful for reading data from Excel files and converting them into pandas DataFrames.