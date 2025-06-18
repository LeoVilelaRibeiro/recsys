# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Installing Libraries

# COMMAND ----------

# MAGIC %pip install surprise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Importing Libraries

# COMMAND ----------

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Fetching Data

# COMMAND ----------

# Carregar os dados de avaliações (ratings) em um DataFrame
ratings_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/recsys/collaborative_filtering_system/filtering_system_libs/data/ratings.csv')

# Carregar os dados de filmes (movies) em um DataFrame
movies_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/recsys/collaborative_filtering_system/filtering_system_libs/data/movies.csv')

# Exemplo de visualização dos primeiros registros de cada DataFrame
print("Primeiros registros de ratings:")
print(ratings_df.head())

print("\nPrimeiros registros de movies:")
print(movies_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Training

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Const Definitions

# COMMAND ----------

# Definir o objeto Reader para especificar a escala de classificação (de 0 a 5 no caso de ratings)
reader = Reader(rating_scale=(0, 5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Transforming Data

# COMMAND ----------

# Carregar os dados do DataFrame de avaliações para o formato do surprise
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Splitting Training and Testing Data

# COMMAND ----------

# Dividir os dados em conjuntos de treino e teste
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Training

# COMMAND ----------

# Inicializar o algoritmo SVD
algo = SVD()

# Treinar o modelo com o conjunto de treino
algo.fit(trainset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 Testing

# COMMAND ----------

# Avaliar o modelo com o conjunto de teste
predictions = algo.test(testset)

# Calcular RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)

# Calcular MAE (Mean Absolute Error)
mae = accuracy.mae(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Function Definitions

# COMMAND ----------

# Função para obter recomendações para um usuário específico
def get_recommendations(user_id, top_n=10):
  # Listar todos os itens únicos
  all_movie_ids = set(ratings_df['movieId'].unique())
  rated_movie_ids = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].unique())

  # Obter IDs dos itens não vistos pelo usuário
  unseen_movie_ids = list(all_movie_ids - rated_movie_ids)

  # Preparar os dados no formato necessário para o surprise
  testset = [[user_id, movie_id, 0] for movie_id in unseen_movie_ids]

  # Gerar recomendações para o usuário
  recommendations = algo.test(testset)

  # Ordenar as recomendações por estimativa de avaliação
  recommendations.sort(key=lambda x: x.est, reverse=True)

  # Retornar as top N recomendações
  top_recs = recommendations[:top_n]

  # Mapear IDs dos itens recomendados para seus títulos
  recommended_movies = [(rec.iid, movies_df[movies_df['movieId'] == rec.iid]['title'].iloc[0], rec.est) for rec in top_recs]

  return recommended_movies

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Usage

# COMMAND ----------

# Exemplo de uso da função get_recommendations para o usuário com ID 1
user_id = 1
recommendations = get_recommendations(user_id)

# Exibir as recomendações
print(f'Recomendações para o usuário {user_id}:')
for i, (movie_id, title, est_rating) in enumerate(recommendations, 1):
    print(f'{i}: {title} (ID: {movie_id}) - Estimativa de Avaliação: {est_rating:.2f}')
