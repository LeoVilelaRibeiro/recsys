# Databricks notebook source
# MAGIC %md
# MAGIC # Arrumar

# COMMAND ----------

# MAGIC %pip install tensorflow_recommenders

# COMMAND ----------

import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
import random

# COMMAND ----------

# # Gerar dados sintéticos
# np.random.seed(42)
# num_users = 1000
# num_movies = 500
# num_ratings = 10000
# num_categories = 5

# user_ids = np.random.choice([f"user_{i}" for i in range(num_users)], num_ratings)
# movie_ids = np.random.choice([f"movie_{i}" for i in range(num_movies)], num_ratings)
# ratings = np.random.uniform(1, 5, num_ratings).astype(np.float32)
# categories = np.random.choice([f"category_{i}" for i in range(num_categories)], num_users)

# # Criar DataFrame de avaliações
# ratings_df = pd.DataFrame({
#   "userId": user_ids,
#   "movieId": movie_ids,
#   "rating": ratings
# })

# # Criar DataFrame de categorias dos clientes
# customer_categories_df = pd.DataFrame({
#   "userId": [f"user_{i}" for i in range(num_users)],
#   "category": categories
# })

# # Mesclar os dados de categorias dos clientes com as avaliações
# ratings_df = ratings_df.merge(customer_categories_df, on='userId')

# # Converter para TensorFlow Dataset
# ratings = tf.data.Dataset.from_tensor_slices(dict(ratings_df))

# COMMAND ----------

movies_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/recsys/collaborative_filtering_system/filtering_system_libs/data/movies.csv')
ratings_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/recsys/collaborative_filtering_system/filtering_system_libs/data/ratings.csv')

# COMMAND ----------

movies_df['category'] = movies_df['genres'].apply(lambda x: random.choice(x.split('|')))
movies_df = movies_df.merge(ratings_df, on='movieId')
movies_df = movies_df[['userId', 'movieId', 'rating', 'category']]
movies_df['userId'] = movies_df['userId'].astype(str)
movies_df['movieId'] = movies_df['movieId'].astype(str)
movies_df['rating'] = movies_df['rating'].astype(float)
movies_df['category'] = movies_df['category'].astype(str)
ratings = tf.data.Dataset.from_tensor_slices(dict(movies_df))

# COMMAND ----------

movies_df.head(6)

# COMMAND ----------


# userId	movieId	rating	category
# 0	user_102	movie_441	3.050364	category_0
# 1	user_102	movie_462	3.383707	category_0
# 2	user_102	movie_215	1.186791	category_0
# 3	user_102	movie_18	1.746255	category_0
# 4	user_102	movie_478	3.066617	category_0
# ...	...	...	...	...
# 9995	user_807	movie_375	1.070466	category_3
# 9996	user_807	movie_332	1.901608	category_3
# 9997	user_807	movie_2	1.489459	category_3
# 9998	user_807	movie_97	4.053174	category_3
# 9999	user_807	movie_1	4.449367	category_3

# COMMAND ----------

# Dividir os dados
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train_size = int(0.8 * len(shuffled))
train = shuffled.take(train_size)
test = shuffled.skip(train_size)

# COMMAND ----------

class RecommenderModel(tfrs.Model):
  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(vocabulary=movies_df['userId'].unique()),
      tf.keras.layers.Embedding(len(movies_df['userId'].unique()) + 1, embedding_dimension)
    ])

    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(vocabulary=movies_df['movieId'].unique()),
      tf.keras.layers.Embedding(len(movies_df['movieId'].unique()) + 1, embedding_dimension)
    ])

    self.category_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(vocabulary=movies_df['category'].unique()),
      tf.keras.layers.Embedding(len(movies_df['category'].unique()) + 1, embedding_dimension)
    ])

    self.rating_model = tf.keras.Sequential([
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(1)
    ])

    self.task = tfrs.tasks.Ranking(
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, inputs):
    user_embeddings = self.user_embeddings(inputs["userId"])
    movie_embeddings = self.movie_embeddings(inputs["movieId"])
    category_embeddings = self.category_embeddings(inputs["category"])
    return self.rating_model(tf.concat([user_embeddings, movie_embeddings, category_embeddings], axis=1))

  def compute_loss(self, features, training=False):
    user_embeddings = self.user_embeddings(features["userId"])
    movie_embeddings = self.movie_embeddings(features["movieId"])
    category_embeddings = self.category_embeddings(features["category"])

    rating_predictions = self.rating_model(
      tf.concat([user_embeddings, movie_embeddings, category_embeddings], axis=1)
    )

    return self.task(
      labels=features["rating"],
      predictions=rating_predictions,
    )

  def recommend_top_movies(self, user_id, top_k=5):
    movie_ids = movies_df['movieId'].unique()
    user_ids = np.array([user_id] * len(movie_ids))
    categories = movies_df['category'].unique()
    category = random.choice(categories)
    categories = np.array([category] * len(movie_ids))

    pred_data = {
      "userId": tf.constant(user_ids),
      "movieId": tf.constant(movie_ids),
      "category": tf.constant(categories)
    }

    pred_ds = tf.data.Dataset.from_tensor_slices(pred_data).batch(64)
    predictions = self.predict(pred_ds)

    movie_scores = list(zip(movie_ids, predictions))
    return sorted(movie_scores, key=lambda x: x[1], reverse=True)[:top_k]

model = RecommenderModel()


# COMMAND ----------

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# COMMAND ----------

# Treinar o modelo
cached_train = train.batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=5)

# Avaliar o modelo
test_loss = model.evaluate(cached_test, return_dict=True)

# COMMAND ----------

test_loss

# COMMAND ----------

pred_data = {
  "userId": ["user_1", "user_2"],  # IDs de usuários para predição
  "movieId": ["movie_1", "movie_2"],  # IDs de filmes para predição
  "category": ["category_0", "category_1"]  # Categorias dos usuários para predição
}

pred_ds = tf.data.Dataset.from_tensor_slices(pred_data).batch(1)
sample_prediction = model.predict(pred_ds)
signature = infer_signature(pred_data, sample_prediction)

# COMMAND ----------

class RecommenderModelWrapper(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    self.model = context.artifacts["recommender_model"]

  def predict(self, context, model_input):
    user_id = model_input['userId'][0]
    top_k = model_input['top_k'][0] if 'top_k' in model_input else 5
    return self.model.recommend_top_movies(user_id, top_k)

# COMMAND ----------

with mlflow.start_run() as run:
  # Logar parâmetros e métricas como antes...
  # Logar o modelo usando mlflow.pyfunc
  mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=RecommenderModelWrapper(),
    artifacts={"recommender_model": model}
  )

  # Registrar o modelo
  mlflow.register_model(
    "runs:/{}/model".format(run.info.run_id), 
    "CustomerCategoryRecommenderModel"
  )

# COMMAND ----------

# Carregar o modelo registrado
model_name = "CustomerCategoryRecommenderModel"
model_version = 1  # ou a versão que você desejar

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

# Dados para predição
model_input = pd.DataFrame({
  'userId': ["user_1"],
  'top_k': [5]
})

# Fazer a predição
recommendations = model.predict(model_input)
print(recommendations)

# COMMAND ----------

# MAGIC %md
# MAGIC #novo (sem mlflow)

# COMMAND ----------

import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
import random

# COMMAND ----------

# Carregar dados
movies_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/recsys/collaborative_filtering_system/filtering_system_libs/data/movies.csv')
ratings_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/recsys/collaborative_filtering_system/filtering_system_libs/data/ratings.csv')

# COMMAND ----------

# Adicionar categorias aleatórias
movies_df['category'] = movies_df['genres'].apply(lambda x: random.choice(x.split('|')))
movies_df = movies_df.merge(ratings_df, on='movieId')
movies_df['userId'] = movies_df['userId'].astype(str)
movies_df['movieId'] = movies_df['movieId'].astype(str)
movies_df['rating'] = movies_df['rating'].astype(float)
movies_df['category'] = movies_df['category'].astype(str)


# Converter para TensorFlow Dataset
ratings = tf.data.Dataset.from_tensor_slices(dict(movies_df))

# COMMAND ----------

# Dividir os dados
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
train_size = int(0.8 * len(shuffled))
train = shuffled.take(train_size)
test = shuffled.skip(train_size)

# COMMAND ----------

# Definir o modelo
class RecommenderModel(tfrs.Model):
    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=movies_df['userId'].unique()),
            tf.keras.layers.Embedding(len(movies_df['userId'].unique()) + 1, embedding_dimension)
        ])

        self.movie_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=movies_df['movieId'].unique()),
            tf.keras.layers.Embedding(len(movies_df['movieId'].unique()) + 1, embedding_dimension)
        ])

        self.category_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=movies_df['category'].unique()),
            tf.keras.layers.Embedding(len(movies_df['category'].unique()) + 1, embedding_dimension)
        ])

        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, inputs):
        user_embeddings = self.user_embeddings(inputs["userId"])
        movie_embeddings = self.movie_embeddings(inputs["movieId"])
        category_embeddings = self.category_embeddings(inputs["category"])
        return self.rating_model(tf.concat([user_embeddings, movie_embeddings, category_embeddings], axis=1))

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_embeddings(features["userId"])
        movie_embeddings = self.movie_embeddings(features["movieId"])
        category_embeddings = self.category_embeddings(features["category"])

        rating_predictions = self.rating_model(
            tf.concat([user_embeddings, movie_embeddings, category_embeddings], axis=1)
        )

        return self.task(
            labels=features["rating"],
            predictions=rating_predictions,
        )

model = RecommenderModel()

# COMMAND ----------

# Compilar o modelo
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# COMMAND ----------

# Treinar o modelo
cached_train = train.batch(8192).cache()
cached_test = test.batch(4096).cache()
model.fit(cached_train, epochs=5)

# COMMAND ----------

# Avaliar o modelo
test_loss = model.evaluate(cached_test, return_dict=True)
print("Test Loss:", test_loss)

# COMMAND ----------

# Função para obter recomendações
def get_prediction(user_id: int, prediction_amount=5):
    user_id = str(user_id)
    # Obter todos os IDs de filmes que o usuário já avaliou
    rated_movies = set(movies_df[movies_df['userId'] == user_id]['movieId'])

    # Obter todos os IDs de filmes
    all_movie_ids = set(movies_df['movieId'].unique())

    # Filtrar filmes não avaliados pelo usuário
    unrated_movie_ids = all_movie_ids - rated_movies

    # Criar Dataset de predição para filmes não avaliados
    pred_data = {
        "userId": [user_id] * len(unrated_movie_ids),
        "movieId": list(unrated_movie_ids),
        "category": [movies_df[movies_df['movieId'] == movie_id]['category'].values[0] if not movies_df[movies_df['movieId'] == movie_id]['category'].empty else 'unknown' for movie_id in unrated_movie_ids]
    }

    # Converter o Dataset de entrada para TensorFlow
    pred_ds = tf.data.Dataset.from_tensor_slices(pred_data).batch(1)

    # Fazer a predição
    predictions = model.predict(pred_ds)

    # Adicionar previsões ao DataFrame
    prediction_df = pd.DataFrame({
        'movieId': list(unrated_movie_ids),
        'predicted_rating': [pred[0] for pred in predictions]
    })

    # Ordenar pelos ratings previstos e pegar os principais
    return prediction_df.sort_values(by='predicted_rating', ascending=False).head(prediction_amount)

# COMMAND ----------

print(get_prediction(5))