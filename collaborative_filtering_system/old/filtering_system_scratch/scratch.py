# Databricks notebook source
import pandas as pd
import numpy as np

# COMMAND ----------

# Carregar os dados de avaliações (ratings) em um DataFrame
ratings_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/collaborative_filtering_system/filtering_system_scratch/data/ratings.csv')

# Carregar os dados de filmes (movies) em um DataFrame
movies_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/collaborative_filtering_system/filtering_system_scratch/data/movies.csv')

# Exemplo de visualização dos primeiros registros de cada DataFrame
print("Primeiros registros de ratings:")
print(ratings_df.head())

print("\nPrimeiros registros de movies:")
print(movies_df.head())

# COMMAND ----------

# Construir a matriz de utilidade
# Usuários (linhas) x Itens (colunas)
# Preencher com avaliações reais dos usuários para os itens
def build_utility_matrix(ratings_df):
    utility_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return utility_matrix

# Exemplo de cálculo de similaridade entre usuários usando similaridade de cosseno
def calculate_user_similarity(user_id, utility_matrix):
    cosine_similarities = {}
    target_user_ratings = utility_matrix.loc[user_id].values.reshape(1, -1)
    
    for index, row in utility_matrix.iterrows():
        if index != user_id:
            similarity = cosine_similarity(target_user_ratings, row.values.reshape(1, -1))
            cosine_similarities[index] = similarity[0][0]
    
    return cosine_similarities

# Função para obter as recomendações para um usuário específico
def get_recommendations(user_id, utility_matrix, movies_df, top_n=10):
    # Calcular similaridade do usuário-alvo com todos os outros usuários
    similarities = calculate_user_similarity(user_id, utility_matrix)
    
    # Ordenar usuários por similaridade (do mais similar para o menos similar)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Coletar filmes vistos pelo usuário-alvo
    target_user_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].values)
    
    # Coletar filmes recomendados pelos usuários mais similares
    recommendations = []
    for sim_user, _ in sorted_similarities:
        sim_user_ratings = ratings_df[(ratings_df['userId'] == sim_user) & (~ratings_df['movieId'].isin(target_user_movies))]
        
        for _, movie_id, rating, _ in sim_user_ratings.itertuples(index=False):
            movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
            recommendations.append((movie_id, movie_title, rating))
            if len(recommendations) >= top_n:
                return recommendations
    
    return recommendations

# Exemplo de uso da função get_recommendations para o usuário com ID 1
user_id = 1
utility_matrix = build_utility_matrix(ratings_df)
recommendations = get_recommendations(user_id, utility_matrix, movies_df)

# Exibir as recomendações
print(f'Recomendações para o usuário {user_id}:')
for i, (movie_id, title, rating) in enumerate(recommendations, 1):
    print(f'{i}: Filme ID {movie_id}, Título: {title}, Avaliação: {rating:.2f}')