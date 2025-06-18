# Databricks notebook source
# MAGIC %pip install lightfm

# COMMAND ----------

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from lightfm.evaluation import precision_at_k, recall_at_k
import joblib

# COMMAND ----------

# Dados de exemplo
ratings_dict = {
  "userID": [1, 1, 1, 2, 2, 2, 3, 3, 3],
  "itemID": [1, 2, 3, 1, 2, 3, 1, 2, 3],
  "rating": [5, 4, 3, 4, 3, 5, 2, 5, 3]
}

categories_dict = {
  "userID": [1, 2, 3],
  "category": ["A", "B", "A"]
}

ratings_df = pd.DataFrame(ratings_dict)
categories_df = pd.DataFrame(categories_dict)
ratings_df = ratings_df.merge(categories_df, on="userID")

# COMMAND ----------

# Dividir dados em treino e teste
train_df, test_df = train_test_split(ratings_df, test_size=0.33, random_state=42)

# COMMAND ----------

# Criar dataset para LightFM
dataset = Dataset()
dataset.fit(
  ratings_df["userID"].unique(),
  ratings_df["itemID"].unique(),
  user_features=categories_df["category"].unique()
)
(interactions_train, weights_train) = dataset.build_interactions(
  [(x["userID"], x["itemID"], x["rating"]) for index, x in train_df.iterrows()]
)
(interactions_test, weights_test) = dataset.build_interactions(
  [(x["userID"], x["itemID"], x["rating"]) for index, x in test_df.iterrows()]
)
user_features = dataset.build_user_features(
  [(x["userID"], [x["category"]]) for index, x in categories_df.iterrows()]
)

# COMMAND ----------

# Treinar e logar com MLflow
model = LightFM(loss='warp')
# mlflow.set_experiment("recommender_system")
with mlflow.start_run(run_name="LightFM_model"):
  model.fit(interactions_train, user_features=user_features, epochs=30, num_threads=2)

  # Avaliar o modelo
  precision = precision_at_k(model, interactions_test, user_features=user_features, k=5).mean()
  recall = recall_at_k(model, interactions_test, user_features=user_features, k=5).mean()
  
  # Logar métricas e parâmetros
  mlflow.log_params({"loss": "warp", "epochs": 30, "num_threads": 2})
  mlflow.log_metric("precision_at_k", precision)
  mlflow.log_metric("recall_at_k", recall)
  
  # Inferir assinatura do modelo
  sample_input = (np.array([1]), np.arange(interactions_train.shape[1]))
  signature = infer_signature(sample_input, model.predict(np.array([1]), np.arange(interactions_train.shape[1]), user_features=user_features))
  
  # Salvar e logar o modelo usando joblib
  model_path = "lightfm_model.pkl"
  joblib.dump(model, model_path)
  mlflow.log_artifact(model_path)
  
  # Registrar o modelo no MLflow
  mlflow.pyfunc.save_model(path="model", python_model=LightFMWrapper(model), signature=signature)
  mlflow.register_model("runs:/" + mlflow.active_run().info.run_id + "/model", "LightFM_Model")

# COMMAND ----------

# Fazer recomendações
def recommend(model, user_id, user_features, interactions, num_items=3):
  scores = model.predict(user_id, np.arange(interactions.shape[1]), user_features=user_features)
  return np.argsort(-scores)[:num_items]

# COMMAND ----------

user_id = dataset.mapping()[0]['1']  # ID interno do LightFM para o userID 1
top_items = recommend(model, user_id, user_features, interactions_train)
print(f"Recomendações para o usuário {user_id}: {top_items}")