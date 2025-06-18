# Databricks notebook source
# MAGIC %pip install tensorflow tensorflow-recommenders mlflow

# COMMAND ----------

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
import os

# COMMAND ----------

# Dados de exemplo
ratings_dict = {
  "userID": [1, 1, 1, 2, 2, 2, 3, 3, 3],
  "itemID": [1, 2, 3, 1, 2, 3, 1, 2, 3],
  "rating": [5, 4, 3, 4, 3, 5, 2, 5, 3]
}

ratings_df = pd.DataFrame(ratings_dict)

# COMMAND ----------

# Dividir dados em treino e teste
train_df, test_df = train_test_split(ratings_df, test_size=0.33, random_state=42)

# Criar Dataset para TFRS
unique_user_ids = train_df['userID'].astype(str).unique()
unique_item_ids = train_df['itemID'].astype(str).unique()

# COMMAND ----------

# Definir o treinamento
class RecommenderModel(tfrs.Model):
  def __init__(self):
    super().__init__()
    self.user_lookup = tf.keras.layers.StringLookup(vocabulary=unique_user_ids)
    self.item_lookup = tf.keras.layers.StringLookup(vocabulary=unique_item_ids)
    self.user_embeddings = tf.keras.layers.Embedding(len(unique_user_ids), 32)
    self.item_embeddings = tf.keras.layers.Embedding(len(unique_item_ids), 32)

  def call(self, inputs):
    user_ids = inputs['user_id']
    item_ids = inputs['item_id']
    user_ids = self.user_lookup(user_ids)
    item_ids = self.item_lookup(item_ids)
    user_embeddings = self.user_embeddings(user_ids)
    item_embeddings = self.item_embeddings(item_ids)
    return user_embeddings, item_embeddings

  def compute_loss(self, features, training=False):
    user_embeddings, item_embeddings = self(features)
    true_ratings = features['rating']
    return tf.reduce_mean(tf.square(user_embeddings - item_embeddings))

# Treinar o modelo
model = RecommenderModel()

# COMMAND ----------

def evaluate_model(model, test_data):
  user_ids = test_data['userID'].astype(str).values
  item_ids = test_data['itemID'].astype(str).values
  true_ratings = test_data['rating'].values
  
  user_embeddings, item_embeddings = model({
    'user_id': tf.constant(user_ids),
    'item_id': tf.constant(item_ids)
  })
  
  predictions = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)
  return np.mean(np.square(true_ratings - predictions.numpy()))

# COMMAND ----------

def get_model_signature(model):
  # Dados de exemplo para entrada
  example_input = {
    'user_id': tf.constant(['1']),
    'item_id': tf.constant(['1'])
  }
  
  # Dados de exemplo para saída
  example_output = model(example_input)
  
  # Inferir a assinatura a partir dos dados de exemplo e previsões
  return infer_signature(example_input, example_output)

# COMMAND ----------

# Compilar o modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer)

# COMMAND ----------

# Preparar os dados de treinamento
train_dataset = tf.data.Dataset.from_tensor_slices({
  'user_id': tf.constant(train_df['userID'].astype(str).values),
  'item_id': tf.constant(train_df['itemID'].astype(str).values),
  'rating': tf.constant(train_df['rating'].values)
}).batch(1)

# COMMAND ----------

# Construir o modelo com dados de exemplo
example_input = {
  'user_id': tf.constant(['1']),
  'item_id': tf.constant(['1'])
}
model(example_input)  # Isso constrói o modelo

# COMMAND ----------

# Treinar o modelo
model.fit(train_dataset, epochs=3)

# COMMAND ----------

accuracy = evaluate_model(model, test_df)


# COMMAND ----------

# Definir o caminho onde o modelo será salvo
model_save_path = "tfrs_model"

# Salvar o modelo TensorFlow
model.save(model_save_path)

# COMMAND ----------

with mlflow.start_run(run_name="TFRS_Model"):
  mlflow.log_param("epochs", 3)
  mlflow.log_metric("accuracy", accuracy)
  
  # Obtenha a assinatura do modelo
  signature = get_model_signature(model)
  
  # Salvar o modelo com assinatura
  mlflow.tensorflow.save_model(
    model,
    path=model_save_path,
    signature=signature
  )
  
  # Registrar o modelo no MLflow
  mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "TFRS_Model")

# COMMAND ----------

def recommend(model, user_id, num_items=3):
  # Convert ID do usuário para string
  user_id_str = str(user_id)
  
  # Obter o embedding do usuário
  user_embedding = model.user_embeddings(tf.constant([user_id_str]))
  
  # Gerar embeddings para todos os itens
  item_ids = np.arange(len(unique_item_ids)).astype(str)
  item_embeddings = model.item_embeddings(tf.constant(item_ids))
  
  # Calcular a pontuação (similaridade) entre o embedding do usuário e os embeddings dos itens
  scores = tf.reduce_sum(user_embedding * item_embeddings, axis=1)
  
  # Retornar os índices dos itens mais recomendados
  return np.argsort(-scores.numpy())[:num_items]

# COMMAND ----------

user_id = 1
top_items = recommend(model, user_id)
print(f"Recomendações para o usuário {user_id}: {top_items}")