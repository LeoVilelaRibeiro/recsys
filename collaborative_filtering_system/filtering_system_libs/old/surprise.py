# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Installing Libraries

# COMMAND ----------

# MAGIC %pip install surprise mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Importing Libraries

# COMMAND ----------

import pandas as pd
import mlflow
import mlflow.sklearn
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
from mlflow.models.signature import infer_signature

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Fetching Data

# COMMAND ----------

# Load ratings data into a DataFrame
ratings_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/recsys/collaborative_filtering_system/filtering_system_libs/data/ratings.csv')

# Load movies data into a DataFrame
movies_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/recsys/collaborative_filtering_system/filtering_system_libs/data/movies.csv')

# Display the first few records of each DataFrame
print("First records of ratings:")
print(ratings_df.head())

print("\nFirst records of movies:")
print(movies_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Training

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Const Definitions

# COMMAND ----------

# Define the Reader object to specify the rating scale (from 0 to 5 in the case of ratings)
reader = Reader(rating_scale=(0, 5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Transforming Data

# COMMAND ----------

# Load the data from the ratings DataFrame into the surprise format
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Splitting Training and Testing Data

# COMMAND ----------

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Training

# COMMAND ----------

# Initialize the SVD algorithm
algo = SVD()

# Start MLFlow run
with mlflow.start_run(run_name="SVD_Training"):
    # Train the model with the training set
    algo.fit(trainset)

    # Log model parameters
    mlflow.log_param("algorithm", "SVD")
    mlflow.log_param("test_size", 0.2)

    # Infer the model signature
    example_input = ratings_df[['userId', 'movieId', 'rating']].head(5)
    signature = infer_signature(example_input, example_input['rating'])

    # Log the trained model with the inferred signature
    mlflow.sklearn.log_model(algo, "model", signature=signature)
        # Evaluate the model with the testing set
    predictions = algo.test(testset)

    # Calculate RMSE (Root Mean Squared Error)
    rmse = accuracy.rmse(predictions)
    mlflow.log_metric("rmse", rmse)

    # Calculate MAE (Mean Absolute Error)
    mae = accuracy.mae(predictions)
    mlflow.log_metric("mae", mae)

    # Additional Metrics
    def precision_recall_at_k(predictions, k=10, threshold=3.5):
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:k])

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        precision_at_k = sum(prec for prec in precisions.values()) / len(precisions)
        recall_at_k = sum(rec for rec in recalls.values()) / len(recalls)
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) != 0 else 0

        return precision_at_k, recall_at_k, f1_at_k

    # Calculate Precision@K, Recall@K, and F1@K
    precision, recall, f1 = precision_recall_at_k(predictions, k=10)
    mlflow.log_metric("precision_at_10", precision)
    mlflow.log_metric("recall_at_10", recall)
    mlflow.log_metric("f1_at_10", f1)

    # Register the model
    model_name = "SVD_Recommender"
    mlflow.register_model("runs:/{run_id}/model".format(run_id=mlflow.active_run().info.run_id), model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 Testing

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Function Definitions

# COMMAND ----------

# Function to get recommendations for a specific user
def get_recommendations(user_id, top_n=10):
    # List all unique items
    all_movie_ids = set(ratings_df['movieId'].unique())
    rated_movie_ids = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].unique())

    # Get IDs of items not seen by the user
    unseen_movie_ids = list(all_movie_ids - rated_movie_ids)

    # Prepare the data in the format required for surprise
    testset = [[user_id, movie_id, 0] for movie_id in unseen_movie_ids]

    # Generate recommendations for the user
    recommendations = algo.test(testset)

    # Sort recommendations by estimated rating
    recommendations.sort(key=lambda x: x.est, reverse=True)

    # Return the top N recommendations
    top_recs = recommendations[:top_n]

    # Map the recommended item IDs to their titles
    recommended_movies = [(rec.iid, movies_df[movies_df['movieId'] == rec.iid]['title'].iloc[0], rec.est) for rec in top_recs]

    return recommended_movies

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Usage

# COMMAND ----------

# Example usage of the get_recommendations function for user with ID 1
user_id = 1
recommendations = get_recommendations(user_id)

# Display the recommendations
print(f'Recommendations for user {user_id}:')
for i, (movie_id, title, est_rating) in enumerate(recommendations, 1):
    print(f'{i}: {title} (ID: {movie_id}) - Estimated Rating: {est_rating:.2f}')