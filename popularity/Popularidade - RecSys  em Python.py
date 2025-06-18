# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Importing Libraries
# MAGIC

# COMMAND ----------

#Este código foi mantido
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC **Library Imports:**
# MAGIC - **numpy, scipy, pandas, math, random:** Fundamental libraries for numerical operations, scientific computing, data manipulation, and mathematical operations.
# MAGIC - **sklearn**: Library providing tools for machine learning, including model selection and evaluation.
# MAGIC - **nltk**: Natural Language Toolkit for text processing tasks.
# MAGIC - **stopwords**: Stopwords corpus from NLTK for filtering out common words.
# MAGIC - **TfidfVectorizer**, **cosine_similarity**: Tools from **sklearn** for text feature extraction and similarity calculation.
# MAGIC - **svds**: Singular Value Decomposition (SVD) from **scipy.sparse.linalg** for matrix factorization.
# MAGIC - **matplotlib.pyplot**: Plotting library for visualizations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Loading Movie Data
# MAGIC Reads the CSV file movies.csv into a pandas DataFrame movies_df. This file contains metadata about movies.

# COMMAND ----------

#O arquivo movies_metadata.cvs foi tratado de forma a excluir os registros com campos nulos.
movies_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/recsys/movies.csv', delimiter=',',quotechar='"')


# COMMAND ----------

# MAGIC %md
# MAGIC Displays the first 20 rows of the movies_df DataFrame to inspect the movie metadata.

# COMMAND ----------

movies_df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Fetching Data
# MAGIC Reads the CSV file **ratings.csv** into a pandas DataFrame **interactions_df**. This file contains user interactions (ratings) with movies.

# COMMAND ----------

#Carregamento da base de dados de interações.
interactions_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/recsys/ratings.csv', delimiter=',')

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Data Manipulation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Counting User Interactions
# MAGIC Computes the number of different movies each user has interacted with and prints the total number of users.

# COMMAND ----------

# Conta com quantos filmes diferentes cada usuário interagiu
users_interactions_count_df = interactions_df.groupby(['userId', 'movieId']).size().groupby('userId').size()
print('# users: %d' % len(users_interactions_count_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Displaying User Interaction Counts
# MAGIC Displays the count of interactions for each user.

# COMMAND ----------

users_interactions_count_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Filtering Users by Interaction Count
# MAGIC Filters users who have interacted with at least 5 movies and prints the count of such users.

# COMMAND ----------

# Filtra apenas os usuários com pelo menos 5 filmes assistidos
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['userId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Filtering Interactions by Selected Users
# MAGIC Filters interactions of users who have interacted with at least 5 movies and prints the count of such interactions.

# COMMAND ----------

# Filtra apenas as interações dos usuários selecionados com pelo menos 5 filmes assistidos
print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'userId',
               right_on = 'userId')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))

# COMMAND ----------

interactions_from_selected_users_df

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 User Ratings
# MAGIC - Applies a logarithmic transformation to user ratings (**smooth_user_preference** function).
# MAGIC - Aggregates user interactions by summing ratings and applies the transformation.
# MAGIC - Prints the count of unique user/item interactions and displays the first 20 rows of the **interactions_full_df** DataFrame.

# COMMAND ----------

def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = interactions_from_selected_users_df \
                    .groupby(['userId', 'movieId'])['rating'].sum() \
                    .apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(20)

#Executei o código a seguir uma única vez para gerar o arquivo CVS completo com os ratings
#interactions_full_df.to_csv('interactions_full_df.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Train/Test Split of Interactions
# MAGIC - Splits the **interactions_full_df** DataFrame into training and testing sets using **train_test_split**.
# MAGIC - Prints the number of interactions in the training and testing sets and displays the first 10 rows of **interactions_train_df**.

# COMMAND ----------

#Deixei pronto o código caso não seja necessário aplicar a função smooth_user_preference
'''interactions_train_df, interactions_test_df = train_test_split(interactions_df,
                                   stratify=interactions_df['userId'], 
                                   test_size=0.20,
                                   random_state=42)
'''
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['userId'], 
                                   test_size=0.20,
                                   random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))

interactions_train_df.head(10)

#Executei o código a seguir uma única vez para gerar os arquivos CVS de treino e teste com os ratings
#interactions_train_df.to_csv('interactions_train_df.csv')
#interactions_test_df.to_csv('interactions_test_df.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Indexing DataFrames
# MAGIC Indexes the DataFrames **interactions_full_df**, **interactions_train_df**, and **interactions_test_df** by userId for faster search operations during evaluation.

# COMMAND ----------

#Indexing by userId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('userId')
interactions_train_indexed_df = interactions_train_df.set_index('userId')
interactions_test_indexed_df = interactions_test_df.set_index('userId')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Function to Get Interacted Items
# MAGIC Defines a function **get_items_interacted** to retrieve movies interacted by a user from **interactions_df**.

# COMMAND ----------

def get_items_interacted(user_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[user_id]['movieId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Model Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Defining Evaluator Class
# MAGIC Defines the method **evaluate_model** in **ModelEvaluator** to evaluate a recommendation model for all users based on global metrics and detailed results and creates an instance of **ModelEvaluator** for evaluating recommendation models.

# COMMAND ----------

#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:


    def get_not_interacted_items_sample(self, user_id, sample_size, seed=42):
        interacted_items = get_items_interacted(user_id, interactions_full_indexed_df)
        all_items = set(movies_df['movieId'])
        #all_items = set(interactions_df['movieId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(sorted(list(non_interacted_items)), sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, user_id):
        #Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[user_id]
        if type(interacted_values_testset['movieId']) == pd.Series:
            user_interacted_items_testset = set(interacted_values_testset['movieId'])
        else:
            user_interacted_items_testset = set([int(interacted_values_testset['movieId'])])  
        interacted_items_count_testset = len(user_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        user_recs_df = model.recommend_items(user_id, 
                                               items_to_ignore=get_items_interacted(user_id, 
                                                                                    interactions_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in user_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(user_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = user_recs_df[user_recs_df['movieId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['movieId'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        user_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return user_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, user_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            user_metrics = self.evaluate_model_for_user(model, user_id)  
            user_metrics['_user_id'] = user_id
            people_metrics.append(user_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Computing Item Popularity
# MAGIC - Computes the popularity of items based on total ratings (**rating** sum) from **interactions_full_df**.
# MAGIC - Displays the top 10 most popular items.

# COMMAND ----------

#Computes the most popular items
item_popularity_df = interactions_full_df.groupby('movieId')['rating'].sum().sort_values(ascending=False).reset_index()
item_popularity_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Popularity Recommender Class
# MAGIC - Defines the **PopularityRecommender** class to recommend popular items based on **popularity_df**.
# MAGIC - Includes methods to get model name and recommend items.

# COMMAND ----------

class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['movieId'].isin(items_to_ignore)] \
                               .sort_values('rating', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'movieId', 
                                                          right_on = 'movieId')[['rating', 'movieId','title']]


        return recommendations_df
    
popularity_model = PopularityRecommender(item_popularity_df, movies_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 Evaluating Popularity Model
# MAGIC Evaluates the **popularity_model** using **ModelEvaluator** and prints global metrics and detailed results for the top 10 users.

# COMMAND ----------

print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5 Creating Pivot Table
# MAGIC - Creates a sparse pivot table **users_items_pivot_matrix_df** with users in rows and items in columns based on **interactions_train_df**.
# MAGIC - Displays the first 10 rows of the pivot table.

# COMMAND ----------

#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = interactions_train_df.pivot(index='userId', 
                                                          columns='movieId', 
                                                          values='rating').fillna(0)

users_items_pivot_matrix_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.6 Generating Recommendations
# MAGIC - Generates recommendations (**p_items**) for user **1** using the **popularity_model**.
# MAGIC - Displays the top 20 recommended items including **rating**, **movieId**, and **title**.

# COMMAND ----------

#Popularidade
p_items = popularity_model.recommend_items(1, topn=20, verbose=True)
p_items.head(20)