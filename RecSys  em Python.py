# Databricks notebook source
# MAGIC %md
# MAGIC # Sistema de Recomendação (RecSys) em Python
# MAGIC ---
# MAGIC O objetivo aqui é apresentar as principais abordagens usadas para recomendação de itens. 
# MAGIC
# MAGIC Um [RecSys](https://pt.wikipedia.org/wiki/Sistema_de_recomenda%C3%A7%C3%A3o) sugere itens relevantes a usuários baseado em suas preferências, sejam elas implícitas (inferidas a partir de interações entre usuários e itens no sistema) ou explícitas (relevância de itens expressamente informada pelo usuário). As principais abordagens para recomendação reportadas na literatura são:
# MAGIC
# MAGIC *   [Filtragem Colaborativa](https://pt.wikipedia.org/wiki/Filtragem_colaborativa): Realiza predições (filtragem) sobre o interesse de usuários a itens a partir do histórico completo de preferências (colaborativa). Assume que a probabilidade de dois usuários terem a mesma opinião sobre um determinado item é maior se esses usuários já tiverem opiniões parecidas sobre outros conjuntos de itens.
# MAGIC *   [Filtragem Baseada em Conteúdo](https://pt.wikipedia.org/wiki/Filtragem_baseada_em_conte%C3%BAdo): Realiza predições a partir da similaridade dos descritores textuais e dos atributos de itens e usuários. Recomenda items similares aos itens que o usuário já gostou ou gosta. 
# MAGIC *   [Filtragem Híbrida](https://en.wikipedia.org/wiki/Recommender_system#Hybrid_recommender_systems): Combinação de filtragem colaborativa e baseada em conteúdo com objetivo de minimizar problemas relacionados à esparsidade e [cold-start](https://en.wikipedia.org/wiki/Cold_start).
# MAGIC
# MAGIC Aqui apresentaremos implementações em [Python](https://www.python.org/) das abordagens supra citadas avaliando resultados sobre a base de dados Movies Dataset. Primeiramente, devemos carregar algumas bibliotecas necessárias para a implementação das abordagens:
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
# MAGIC ## Carregamento da Base de Dados
# MAGIC ---
# MAGIC A base de dados contém uma amostra de dados de 1.048.576 interações de usuários em 469.172 filmes. A base de dados é composta por dois [arquivos CSV](https://pt.wikipedia.org/wiki/Comma-separated_values): 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### movies_metadata.csv
# MAGIC
# MAGIC Contém informação sobre os filmes que possuem código, título e visão geral.

# COMMAND ----------

#O arquivo movies_metadata.cvs foi tratado de forma a excluir os registros com campos nulos.
movies_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/recsys/movies.csv', delimiter=',',quotechar='"')


# COMMAND ----------

movies_df.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ratings.csv
# MAGIC
# MAGIC Contém os registros da interação dos usuários com os filmes. Primeiramente coloquei todos os atributos necessários em um único arquivo CVS (userID, movieID, rating, title e genres). Para facilitar os testes, criei um arquivo chamado ratings_reduzida.csv que contém avaliações de 100 usuários. O arquivo ratings_reduzida.cvs foi tratado de forma a excluir os registros com campos nulos (avaliações sem a identificação do respectivo filme.

# COMMAND ----------

#Carregamento da base de dados de interações.
interactions_df = pd.read_csv('/Workspace/Users/leonardo.ribeiro@omni.chat/recsys/ratings.csv', delimiter=',')


# COMMAND ----------

interactions_df.head(50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformações em Dados

# COMMAND ----------

# MAGIC %md
# MAGIC Recomendadores comumente enfrentam um problema conhecido como [cold-start](https://en.wikipedia.org/wiki/Cold_start), em que é difícil prover sugestões personalizadas para usuários com nenhuma ou pouquíssimas interações no sistema, uma vez que tem-se pouca informação para modelar suas preferências. Como o foco aqui não está no tratamento do problema de [cold-start](https://en.wikipedia.org/wiki/Cold_start), para minimizar seu efeito na recomendação utilizaremos usuários com pelo menos 5 interações no sistema.

# COMMAND ----------

users_interactions_count_df = interactions_df.groupby(['userId', 'movieId']).size().groupby('userId').size()
print('# users: %d' % len(users_interactions_count_df))


# COMMAND ----------

users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['userId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))

# COMMAND ----------

print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'userId',
               right_on = 'userId')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))

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
# MAGIC ## Avaliação
# MAGIC ---
# MAGIC Para comparar as diferentes abordagens de recomendação precisamos estabelecer métricas de comparação e adotar técnicas de validação estatística de resultados. Aqui, utilizaremos uma técnica denominada [validação cruzada](https://pt.wikipedia.org/wiki/Valida%C3%A7%C3%A3o_cruzada) (*cross-validation*), em particular utilizaremos uma 
# MAGIC simplificação do [método k-fold](https://pt.wikipedia.org/wiki/Valida%C3%A7%C3%A3o_cruzada#M%C3%A9todo_k-fold) denominado [holdout](https://pt.wikipedia.org/wiki/Valida%C3%A7%C3%A3o_cruzada#M%C3%A9todo_holdout), em que uma amostra aleatória de 80% dos dados são usadas para treino das abordagens de recomendação e os 20% restante são usados para teste.
# MAGIC
# MAGIC Uma forma mais efetiva de avaliação seria adotar a [validação cruzada k-fold](https://pt.wikipedia.org/wiki/Valida%C3%A7%C3%A3o_cruzada#M%C3%A9todo_k-fold), além de não dividir os conjuntos de treino e teste aleatoriamente, mas por datas de referência. Nesse caso, o conjunto de treino seria composto por todas as interações anteriores a uma determinada data e o conjunto de teste seria formado pelas interações restantes. Assim, poderíamos simular o desempenho da abordagem de recomendação na predição de interações futuras.

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
# MAGIC Em RecSys, existem inúmeras métricas que podem ser usadas para avaliação. Aqui utilizaremos métricas de acurácia para o topo do ranking, que avaliam a acurácia dos top-n itens recomendados a um usuário, usando como oráculo (gabarito) suas interações no conjunto de teste. Mais especificamente: 
# MAGIC
# MAGIC > Para cada usuário:
# MAGIC > > Para cada item do conjunto de teste que o usuário interagiu:
# MAGIC > > > Pegamos uma amostra de 100 outros itens que o usuário não interagiu, assumindo que itens sem interação são itens não relevantes ao usuário (o que pode não ser verdadeiro).
# MAGIC
# MAGIC > > > Utilizamos a abordagem de recomendação para produzir um ranking de itens recomendados a partir de um conjunto composto pelo item com interação e os outros 100 itens sem interação
# MAGIC
# MAGIC > > > Calculamos as métricas de acurácia para a lista de recomendação providas pela abordagem
# MAGIC
# MAGIC > > Agregamos as métricas para todos os itens
# MAGIC
# MAGIC > Agregamos as métricas para todos os usuários (global)
# MAGIC
# MAGIC As métricas de acurácia que utilizaremos são derivadas da métrica R@n, que mede a [revocação](https://pt.wikipedia.org/wiki/Precis%C3%A3o_e_revoca%C3%A7%C3%A3o#Revoca%C3%A7%C3%A3o) (*recall*) nos top-n itens ranqueados, ou seja, se um item que o usuário efetivamente interagiu está entre os top-n itens (*hits*) da lista dos 101 recomendados. Outras métricas inportantes e que poderiam ser utilizadas são o [MAP](https://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision), para medir a precisão média, e o [nDCG](https://en.wikipedia.org/wiki/Information_retrieval#Discounted_cumulative_gain), para medir o ganho descontado acumulado. Essas métricas levam em consideração a posição do item relevante no ranking de itens recomendados. Veja também a postagem sobre [avaliação de sistemas de recomendação](http://fastml.com/evaluating-recommender-systems/).
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

#Indexing by userId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('userId')
interactions_train_indexed_df = interactions_train_df.set_index('userId')
interactions_test_indexed_df = interactions_test_df.set_index('userId')

# COMMAND ----------

def get_items_interacted(user_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[user_id]['movieId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

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
# MAGIC ## Abordagens de Recomendação
# MAGIC ---
# MAGIC A seguir apresentaremos as quatro abordagens aqui implementadas para o contexto de recomendação de artigos, bem como os resultados da avaliação de cada uma delas. São elas: abordagem baseada em popularidade, abordagem baseada em conteúdo, abordagem baseada em filtragem colaborativa e abordagem híbrida.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Popularidade
# MAGIC Por ser simples de entender e implementar, esta abordagem ainda é muito utilizada, uma vez que recomenda aos usuários os itens mais populares dentre os que ele ainda não interagiu. Ela não personaliza a recomendação e se baseia na "sabedoria coletiva" (*[wisdom of the crowds](https://en.wikipedia.org/wiki/The_Wisdom_of_Crowds)*), muitas vezes provendo boas recomendações, especialmente para usuários iniciantes sem interesses específicos. Entretanto, um dos principais objetivos da recomendação é a personalização, a fim de oferecer itens que atendam a interesses específicos (e não muito populares) dos usuários (*long-tail itens*). Veja aqui um artigo interessante sobre [os desafios de prover novidade na recomendação, considerando itens na cauda-longa (long-tail)](http://vldb.org/pvldb/vol5/p896_hongzhiyin_vldb2012.pdf).
# MAGIC

# COMMAND ----------

#Computes the most popular items
item_popularity_df = interactions_full_df.groupby('movieId')['rating'].sum().sort_values(ascending=False).reset_index()
item_popularity_df.head(10)

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
# MAGIC Os resultados da avaliação apresentados abaixo mostram que a abordagem baseada em popularidade obteve R@5 de 0.2417, o que significa que aproximadamente 24% dos itens relevantes no conjunto de teste estavam presentes nos 5 primeiros itens do ranking gerado pela abordagem. Se considerarmos os 10 primeiros itens do ranking, o resultado foi ainda melhor com R@10 de 0.3729.

# COMMAND ----------

print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conteúdo
# MAGIC Abordagens de recomendação baseadas em conteúdo levam em conta a descrição (ou atributos) de itens relevantes a usuários para sugerir itens similares. Elas consideram apenas escolhas prévias do próprio usuário, o que as torna resilientes ao problema de  [cold-start](https://en.wikipedia.org/wiki/Cold_start). Para recomendação de itens textuais, como artigos, livros e notícias, o uso do texto associado ao item para construir perfis de itens e usuários se mostra uma tarefa muito simples e intuitiva.
# MAGIC
# MAGIC Aqui utilizaremos um modelo clássico de recuperação de informação para representação de itens e usuários, o [modelo vetorial](https://pt.wikipedia.org/wiki/Modelo_vetorial_em_sistemas_de_recupera%C3%A7%C3%A3o_da_informa%C3%A7%C3%A3o). Nesse modelo, textos não estruturados são convertidos em vetores de palavras onde cada palavra é representada por uma posição no vetor e o valor nessa posição indica a importância (peso) da palavra no texto. Como todos os itens e usuários são representados em um espaço vetorial, o cálculo de similaridade entre itens e usuários pode ser feito utilizando o cosseno ([cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)) entre seus respectivos vetores. Para o cálculo do peso das palavras no texto geralmente utiliza-se o [esquema TF-IDF](https://pt.wikipedia.org/wiki/Tf%E2%80%93idf), onde a frequência da palavra no texto e a raridade da palavra entre os diversos textos determinam sua importância.

# COMMAND ----------

#Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=stopwords_list)

item_ids = movies_df['movieId'].tolist()
tfidf_matrix = vectorizer.fit_transform(movies_df['title'] + "" + movies_df['genres'])
tfidf_feature_names = vectorizer.get_feature_names_out
tfidf_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC Para modelar o perfil de usuário, utilizaremos todos os perfis de itens em que o usuário tenha interagido. Mais especificamente, o vetor de palavras representativos de um usuário será constituído por todas as palavras presentes nos vetores de itens que o usuário interagiu e os pesos de cada palavra serão determinados pelo TF-IDF ponderado pelo grau de interação do usuário (visualização, comentário, etc).

# COMMAND ----------

tfidf_matrix

# COMMAND ----------

def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx+1]
    return item_profile

def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles

def build_users_profile(user_id, interactions_indexed_df):
    interactions_user_df = interactions_indexed_df.loc[user_id]
    user_item_profiles = get_item_profiles(interactions_user_df['movieId'])  
    user_item_strengths = np.array(interactions_user_df['rating']).reshape(-1,1)
    
    #Weighted average of item profiles by the interactions strength
    if (user_item_profiles.shape[0] == user_item_strengths.shape[0]):
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
        user_profile_norm = sklearn.preprocessing.normalize(np.asarray(user_item_strengths_weighted_avg))
        return user_profile_norm
    

def build_users_profiles(): 
    interactions_indexed_df = interactions_full_df[interactions_full_df['movieId'] \
                                                   .isin(movies_df['movieId'])].set_index('userId')
    user_profiles = {}
    for user_id in interactions_indexed_df.index.unique():
        user_profiles[user_id] = build_users_profile(user_id, interactions_indexed_df)
    return user_profiles

# COMMAND ----------

user_profiles = build_users_profiles()
len(user_profiles)

# COMMAND ----------

user_profiles[1]

# COMMAND ----------

# MAGIC %md
# MAGIC Para simplificar e tornar os cálculos mais rápidos, faremos uma poda nos vetores de palavras e fixaremos o tamanho máximo do vetor em 5.000 palavras. Observando o perfil de um usuário específico, espera-se que as palavras mais representativas no perfil do usuário estejam realmente relacionadas aos conceitos que o usuário se interessa. 

# COMMAND ----------

type(user_profiles)

# COMMAND ----------

myprofile = user_profiles[1]

# COMMAND ----------

myprofile

# COMMAND ----------



# COMMAND ----------

# myprofile = user_profiles[1]
# relevance_content_based = pd.DataFrame(
#     																		sorted(
#                                         zip(tfidf_feature_names,
#                                             user_profiles[1].flatten().tolist()
#                                             ),
#                                             key=lambda x: -x[1])[:20],
#             														columns=['token', 'relevance']
#                          							)
# relevance_content_based.head(20)

# COMMAND ----------

class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, user_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[user_id], tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['movieId', 'recStrength']) \
                                    .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'movieId', 
                                                          right_on = 'movieId')[['recStrength','movieId', 'title', 'genres']]

        
        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(movies_df)


# COMMAND ----------

# MAGIC %md
# MAGIC Observando os resultados da avaliação da abordagem baseada em conteúdo, constatamos um salto em efetividade com R@5 indo para 0,4145 e R@10 indo para 0,5241, o que representa um ganho de aproximadamente 71% e 40% sobre a abordagem de popularidade, respectivamente.

# COMMAND ----------

print('Evaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filtragem Colaborativa
# MAGIC Abordagens de filtragem colaborativa adotam duas estratégias diferentes:
# MAGIC
# MAGIC
# MAGIC
# MAGIC *   Baseada em Memória ([*memory-based*](https://en.wikipedia.org/wiki/Collaborative_filtering#Memory-based)): usa a memória de interações passadas para computar similaridades entre usuários baseado nos itens que eles interagiram (*user-based*), ou para computar similaridades entre itens baseado nos usuários que interagiram nesses itens (*item-based*). Um exemplo dessa estratégia é a Filtragem Colaborativa baseada na Vizinhança de Usuário, onde os top-k usuários mais similares a um usuário específico são selecionados e usados para sugerir itens que eles gostaram para o usuário específico, desde que ele ainda não tenha interagido com os itens. Apesar de ser simples de implementar, essa estratégia é cara computacionalmente quando se tem muitos usuários. Uma implementação em Python dessa estratégia está disponível no [*framework* Crab](http://muricoca.github.io/crab/).
# MAGIC *   Baseada em Modelo ([*model-based*](https://en.wikipedia.org/wiki/Collaborative_filtering#Model-based)): modelos são construídos usando diferentes algoritmos de aprendizagem [supervisionada](https://en.wikipedia.org/wiki/Supervised_learning) e [não-supervisionada](https://en.wikipedia.org/wiki/Unsupervised_learning) para sugerir itens a usuários. Para a construção de modelos podemos utilizar inúmeras abordagens, como redes neurais ([*neural networks*](https://en.wikipedia.org/wiki/Artificial_neural_network)), redes bayesianas ([*bayesian networks*](https://en.wikipedia.org/wiki/Bayesian_network)), agrupamento ([*clustering*](https://en.wikipedia.org/wiki/Cluster_analysis)), e fatores latentes (*latent factor models*), tais como [*Singular-Value Decomposition (SVD)*](https://en.wikipedia.org/wiki/Singular-value_decomposition) e [*probabilistic latent semantic analysis*](https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Fatoração de Matriz
# MAGIC Modelos de fatores latentes comprimem a matriz usuário-item em uma representação matricial de baixa dimensionalidade, considerando fatores latentes. A principal vantagem de se usar essa técnica é a substituição de uma matriz esparsa de alta dimensionalidade por uma matriz densa de baixa dimensionalidade. Essa representação reduzida pode ser tanto utilizada em estratégias baseadas em vizinhança de usuário (*user-based*), quanto em vizinhança de itens (*item-based*). Particularmente, esses modelos tratam o problema de esparsidade melhor que os modelos baseados em memória e os cálculos de similaridade na matriz de baixa dimensionalidade resultante são mais eficientes e escaláveis.
# MAGIC
# MAGIC Aqui usaremos a implementação [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html) de um modelo de fatores latentes popular denominado [*Singular-Value Decomposition (SVD)*](https://en.wikipedia.org/wiki/Singular-value_decomposition). Cabe ressaltar que existem diversos *frameworks* para fatoração de matrizes específicos para filtragem colaborativa, como [surprise](https://github.com/NicolasHug/Surprise), [mrec](https://github.com/Mendeley/mrec) e [python-recsys](https://github.com/ocelma/python-recsys). Veja um [exemplo de SVD para recomendação de filmes](https://beckernick.github.io/matrix-factorization-recommender/).
# MAGIC
# MAGIC Um importante parâmetro da fatoração é o número de fatores utilizados para fatorar a matriz. Quanto maior o número de fatores, mais precisa será a fatoração para reconstrução da matriz original. Entretanto, se o modelo memorizar muitos detalhes da matriz original, ele pode não generalizar bem sobre os dados de treino. Em suma, a redução do número de fatores utilizados aumenta a capacidade de generalização do modelo.

# COMMAND ----------

#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = interactions_train_df.pivot(index='userId', 
                                                          columns='movieId', 
                                                          values='rating').fillna(0)

users_items_pivot_matrix_df.head(10)

# COMMAND ----------

users_items_pivot_matrix = users_items_pivot_matrix_df.to_numpy() 
users_items_pivot_matrix[:10]

# COMMAND ----------

users_ids = list(users_items_pivot_matrix_df.index)
users_ids[:10]

# COMMAND ----------

#The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 5
#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)

# COMMAND ----------

U.shape

# COMMAND ----------

Vt.shape

# COMMAND ----------

sigma = np.diag(sigma)
sigma.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Após a fatoração matricial, vamos tentar reconstruir a matriz original através da multiplicação de seus fatores. A matriz resultante não será mais tão esparsa quanto a original, apresentando valores para itens sobre os quais os usuários não interagiram, o que poderá ser explorado na recomendação.

# COMMAND ----------

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
all_user_predicted_ratings

# COMMAND ----------

#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head(10)

# COMMAND ----------

len(cf_preds_df.columns)

# COMMAND ----------

class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['movieId'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False) \
                               .head(topn)
			
        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')
						
            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'movieId', 
                                                          right_on = 'movieId')[['recStrength', 'movieId', 'title', 'genres']]


        return recommendations_df
    
cf_recommender_model = CFRecommender(cf_preds_df, movies_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Observando os resultados da avaliação da abordagem de filtragem colaborativa baseada no modelo de fatores latentes (*SVD matrix factorization*), temos um R@5 de 0.3340 e um R@10 de 0.4681. Essa abordagem superou em aproximadamente 38% (R@5) e 25% (R@10) a abordagem baseada em popularidade, mas apresentou resultados piores (perda de aproximadamente 20% em R@5 e 11% em R@10) se comparada a abordagem baseada em conteúdo. Aparentemente, para essa base de dados, abordagens baseadas em conteúdo se beneficiam da "riqueza" textual dos itens para melhor modelar as preferências dos usuários.

# COMMAND ----------

print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Híbrida
# MAGIC Abordagens híbridas têm apresentado melhores resultados que abordagens puras em muitos casos reais e por esse motivo têm sido muito utilizadas na prática. Aqui, implementaremos uma abordagem híbrida simples obtida usando a multiplicação dos pesos das abordagens de filtragem colaborativa e baseada em conteúdo previamente apresentadas, para prover um ranking obtido a partir dos pesos combinados.

# COMMAND ----------

class HybridRecommender:
    
    MODEL_NAME = 'Hybrid'
    
    def __init__(self, cb_rec_model, cf_rec_model, items_df):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        #Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        
        #Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose, 
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        
        #Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how = 'inner', 
                                   left_on = 'movieId', 
                                   right_on = 'movieId')
        
        #Computing a hybrid recommendation score based on CF and CB scores
        recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
        
        #Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'movieId', 
                                                          right_on = 'movieId')[['recStrengthHybrid', 'movieId', 'title', 'genres']]


        return recommendations_df
    
hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, movies_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Como resultado da avaliação, podemos observar um R@5 de 0.4337 e um R@10 de 0.5379, o que faz com que a abordagem híbrida seja melhor que todas as outras abordagens individualmente. Comparando os resultados com a melhor abordagem dentre as 3 anteriormenrte apresentadas, a abordagem híbrida foi superior (aproximadamente 4% em R@5 e 2% em R@10) à abordagem baseada em conteúdo.

# COMMAND ----------

print('Evaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
hybrid_detailed_results_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparativo entre as Abordagens
# MAGIC ---
# MAGIC

# COMMAND ----------

global_metrics_df = pd.DataFrame([pop_global_metrics, cf_global_metrics, cb_global_metrics, hybrid_global_metrics]) \
                        .set_index('modelName')
global_metrics_df

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC ax = global_metrics_df.transpose().plot(kind='bar', figsize=(15,8))
# MAGIC for p in ax.patches:
# MAGIC     ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testando as Abordagens
# MAGIC ---
# MAGIC Vamos verificar o comportamento das abordagens implementadas para um usuário específico.

# COMMAND ----------

interactions_test_indexed_df

# COMMAND ----------

movies_df

# COMMAND ----------

def inspect_interactions(user_id, test_set=True):    
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df

    return interactions_df.loc[user_id].merge(movies_df, how = 'left', 
                                                      left_on = 'movieId', 
                                                      right_on = 'movieId') \
                          .sort_values('rating', ascending = False)[['rating', 
                                                                          'movieId',
                                                                          'title','genres']]

# COMMAND ----------

# MAGIC %md
# MAGIC Aqui é possível observar alguns artigos em que o usuário específico interagiu na plataforma Movies Dataset a partir do conjunto de treino.

# COMMAND ----------

#Teste com o usuário de número "1"
inspect_interactions(1, test_set=False).head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC Como esperado, as recomendações providas pelas abordagens implementadas realmente são úteis para esse usuário em particular.

# COMMAND ----------

# #Híbrido
h_items = hybrid_recommender_model.recommend_items(1, topn=20, verbose=True)
h_items.head(20)

# COMMAND ----------

#Popularidade
p_items = popularity_model.recommend_items(1, topn=20, verbose=True)
p_items.head(20)

# COMMAND ----------

#Conteúdo
cb_items= content_based_recommender_model.recommend_items(1, topn=20, verbose=True)
cb_items.head(20)

# COMMAND ----------

#Filtragem colaborativa
cf_items = cf_recommender_model.recommend_items(1, topn=20, verbose=True)
cf_items.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusão
# MAGIC ---
# MAGIC Nesse documento, exploramos e comparamos o desempenho de diferentes abordagens de recomendação utilizando a base de dados Movies Dataset. Nesse contexto de recomendação de filmes, as abordagens por popularidade e colaborativa apresentaram melhor desempenho que as demais. Mas ainda há muito espaço para melhoria:
# MAGIC
# MAGIC *   A dimensão temporal foi completamente ignorada e simplesmente consideramos que todos os filmes que estavam disponíveis para serem recomendados para todos os usuários a qualquer tempo. Uma estratégia melhor seria filtrar apenas filmes realmente disponíveis para usuários em um determinado momento.
# MAGIC *   Evidências de contexto, tais como tempo (período do dia, dia da semana, mês), localização (país, estado e cidade) e dispositivos (navegador, app móvel nativo), poderiam ser usadas para melhor modelar as preferências de usuários, sendo incorporadas em modelos [L2R](https://en.wikipedia.org/wiki/Learning_to_rank) ([XGBoost](https://en.wikipedia.org/wiki/Xgboost) para [Gradient Boosting Decision Trees](https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting) com propósito de ranking) ou [logísticos](https://en.wikipedia.org/wiki/Logistic_regression) (com evidências categóricas - one-hot encoding ou [feature hashing](https://en.wikipedia.org/wiki/Feature_hashing)). 
# MAGIC *   Existem abordagens mais avançadas reportadas pela comunidade científica de RecSys que poderiam ser avaliadas, tais como novas algoritmos para fatoração de matrizes e modelos de aprendizagem profunda (*[deep learning](https://en.wikipedia.org/wiki/Deep_learning)*).
# MAGIC
# MAGIC Mais sobre abordagens consideradas estado-da-arte em recomendação podem ser encontradas nos anais da [ACM RecSys conference](https://recsys.acm.org/).
# MAGIC Além disso, diversos *frameworks* encontram-se disponíveis, tais como os já citados [surprise](https://github.com/NicolasHug/Surprise), [mrec](https://github.com/Mendeley/mrec) e [python-recsys](https://github.com/ocelma/python-recsys), bem como o [Spark ALS Matrix Factorization](https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html), uma implementação distribuída para processamento de bases de dados massivas.
# MAGIC
# MAGIC # Métricas de Avaliação
# MAGIC

# COMMAND ----------

#Primeiramente vamos criar os vetores com as predições e os valores reais
#Foi utilizado interactions_full_df que contém as avaliações reais dos usuários
t = pd.DataFrame(interactions_full_df)

#Renomei a coluna para diferenciar o rating real do predito
t.rename(columns={'rating': 'rating_true'}, inplace=True)

#Vamos trabalhar com o usuário 1 como exemplo. Assim, do dataframe t vamos selecionar apenas as avaliações do usuário 1
t.drop(t[t.userId!=1].index ,inplace=True) #Selecionando apenas o usuário de número "1" 
t.drop('userId',axis=1, inplace=True) #Removendo a coluna usuário

#Para cada abordagem, temos um dataframe com as avaliações preditas que são: p_items, cb_items, cf_items, h_items
#Vamos selecionar apenas as avaliações preditas para o usuário 1 em cada abordagem

#Predições para Content Based
pred_cb = pd.DataFrame(cb_items, columns = ['recStrength','movieId'])
pred_cb.rename(columns={'recStrength': 'rating_pred'}, inplace=True) #Renomei para diferenciar o rating predito do real

#Predições para Collaborative Filtering 
pred_cf = pd.DataFrame(cf_items, columns = ['recStrength','movieId'])
pred_cf.rename(columns={'recStrength': 'rating_pred'}, inplace=True) #Renomei para diferenciar o rating predito do real

#Predições para Popularity
pred_p = pd.DataFrame(p_items, columns = ['rating','movieId'])
pred_p.rename(columns={'rating': 'rating_pred'}, inplace=True) #Renomei para diferenciar o rating predito do real

#Predições para Hybrid
pred_h = pd.DataFrame(h_items, columns = ['recStrengthHybrid','movieId'])
pred_h.rename(columns={'recStrengthHybrid': 'rating_pred'}, inplace=True) #Renomei para diferenciar o rating predito do real


# COMMAND ----------

# MAGIC %md
# MAGIC Para a implementaçao das métricas, foi utilizada a biblioteca sklearn.metrics que contém diversas funções para cálculos de métricas. Maiores detalhes sobre a biblioteca podem ser encontrados em http://sklearn.apachecn.org.

# COMMAND ----------

#Mean Absolute Error 
from sklearn.metrics import *

#Mean Absolute Error 
def mae(y_true, y_pred):
    
    if len(y_true) != len(y_pred):
        raise Exception('Error: number of elements not match!')
    error = mean_absolute_error(y_true, y_pred)
    return error

#Mean Squared Error 
def mse(y_true, y_pred):
    
    if len(y_true) != len(y_pred):
        raise Exception('Error: number of elements not match!')
    error = mean_squared_error(y_true, y_pred)
    return error

#Root Mean Squared Error
def rmse(y_true, y_pred):
    
    if len(y_true) != len(y_pred):
        raise Exception('Error: number of elements not match!')
    error = mean_squared_error(y_true, y_pred)
    error = np.sqrt(error)
    return error

#Mean Average Precision
def map(y_true, y_scores):
    
    if len(y_true) != len(y_scores):
        raise Exception('Error: number of elements not match!')
    error = average_precision_score(y_true, y_scores)
    return error

#Discounted Cumulative Gain
#r: escores de relevância 
#k: número de resultados a considerar      
def dcg(r, k): 
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

#Normalized Discounted Cumulative Gain
def ndcg(r, k):
    dcg_max = dcg(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg(r, k) / dcg_max

# COMMAND ----------

#Testando as métricas com cada abordagem 
#Criando um dataframe com as avaliações reais e preditas para o usuário
ratings_full_cb = pd.merge(t, pred_cb, on = 'movieId')
ratings_full_cf = pd.merge(t, pred_cf, on = 'movieId')
ratings_full_p = pd.merge(t, pred_p, on = 'movieId')
ratings_full_h = pd.merge(t, pred_h, on = 'movieId')

y_true_cb = ratings_full_cb.drop('movieId',axis=1).drop('rating_pred', axis=1)#Criando o vetor de ratings verdadeiros (CB)
y_pred_cb = ratings_full_cb.drop('movieId',axis=1).drop('rating_true', axis=1)#Criando o vetor de ratings preditos (CB)

y_true_cf = ratings_full_cf.drop('movieId',axis=1).drop('rating_pred', axis=1)#Criando o vetor de ratings verdadeiros (CF)
y_pred_cf = ratings_full_cf.drop('movieId',axis=1).drop('rating_true', axis=1)#Criando o vetor de ratings preditos (CF)

y_true_p = ratings_full_p.drop('movieId',axis=1).drop('rating_pred', axis=1)#Criando o vetor de ratings verdadeiros (P)
y_pred_p = ratings_full_p.drop('movieId',axis=1).drop('rating_true', axis=1)#Criando o vetor de ratings preditos (P)

y_true_h = ratings_full_h.drop('movieId',axis=1).drop('rating_pred', axis=1)#Criando o vetor de ratings verdadeiros (H)
y_pred_h = ratings_full_h.drop('movieId',axis=1).drop('rating_true', axis=1)#Criando o vetor de ratings preditos (H)

#Mean Absolute Error - Colaborative Filtering
print('Mean Absolute Error / Colaborative Filtering : ', mae(y_true_cf,y_pred_cf))


#Mean Squared Error - Colaborative Filtering
print('\nMean Squared Error  / Colaborative Filtering : ', mse(y_true_cf,y_pred_cf))

#Root Mean Squared Error - Colaborative Filtering
print('\nRoot Mean Squared Error  / Colaborative Filtering : ', rmse(y_true_cf,y_pred_cf))

#Mean Average Precision
#Observação: ainda não consegui definir o vetor y_true binário pois tenho dúvidas sobre como estrai-los dos passos anteriores
#Assim, utilizei apenas um ficticio para testar a implementação da função map
# print('\nMean Average Precision  / Content Based: ', map([0,1,1,1,1,1,0,0,1,1],y_pred_cb))
# print('Mean Average Precision  / Colaborative Filtering : ', map([0,1,1,1],y_pred_cf))
# print('Mean Average Precision  / Popularity : ', map([0,1,1,1,1],y_pred_p))
# print('Mean Average Precision / Hybrid : ', map([0,1,1,1,1,1,0,0,1,0,1],y_pred_h))

#Construção do vetor de relevâncias em cada abordagem
p_cb = (pd.DataFrame(ratings_full_cb, columns = ['movieId','rating_pred'])).sort_values(by='rating_pred', ascending=False)
t_cb = pd.DataFrame(ratings_full_cb, columns = ['movieId','rating_true']).sort_values(by='rating_true', ascending=False)

p_cf = (pd.DataFrame(ratings_full_cf, columns = ['movieId','rating_pred'])).sort_values(by='rating_pred', ascending=False)
t_cf = pd.DataFrame(ratings_full_cf, columns = ['movieId','rating_true']).sort_values(by='rating_true', ascending=False)

p_p = (pd.DataFrame(ratings_full_p, columns = ['movieId','rating_pred'])).sort_values(by='rating_pred', ascending=False)
t_p = pd.DataFrame(ratings_full_p, columns = ['movieId','rating_true']).sort_values(by='rating_true', ascending=False)

p_h = (pd.DataFrame(ratings_full_h, columns = ['movieId','rating_pred'])).sort_values(by='rating_pred', ascending=False)
t_h = pd.DataFrame(ratings_full_h, columns = ['movieId','rating_true']).sort_values(by='rating_true', ascending=False)

#A função a seguir cria um vetor de relevâncias da seguinte maneira:
#É feito um comparativo da posição real do item nas avaliações realizadas pelo usuário com a posição do itens na lista de
#recomendações feitas pela abordagem. A relevância será dada de acordo com a posição do item recomendado na lista de 
#avaliações realizadas pelo usuário. Exemplo: suponha listas com 10 elementos, se na lista de recomendações o item que está 
#na primeira posição também aparece na mesma posição na lista de recomendações, ele recebe relevância 10. A relevância 
#decresce em 1 a cada posição percorrida na lista de avaliações. Se este item aparece na posição 2 da lista, ele 
#recebe relevância 9, na terceira recebe 8 e assim por diante.
def compute_relevances(t,pred,size):   
    vet_rel = [ ]
    i = 0
    for rows_pred in pred.iterrows():
        pos = size
        x = pred.iloc[i][0]
        j = 0
        for rows_t in t.iterrows():
            if (x == t.iloc[j][0]):
                vet_rel.append(pos)
                break
            j += 1  
            pos -= 1
        i += 1
        
    return vet_rel

#Construção das listas de relevâncias para cada abordagem
relevances_cb = compute_relevances(t_cb,p_cb, len(t_cb))
relevances_cf = compute_relevances(t_cf,p_cf,len(t_cf))
relevances_p = compute_relevances(t_p,p_p,len(t_p))
relevances_h = compute_relevances(t_h,p_h,len(t_h))

#Discounted Cumulative Gain
print('\nDiscounted Cumulative Gain  / Content Based: ', dcg(relevances_cb, k=5))
print('Discounted Cumulative Gain  / Colaborative Filtering : ', dcg(relevances_cf,k=5))
print('Discounted Cumulative Gain  / Popularity : ', dcg(relevances_p,k=5))
print('Discounted Cumulative Gain / Hybrid : ', dcg(relevances_h,k=5))

#Normalized Discounted Cumulative Gain - NDCG1
print('\nNormalized Discounted Cumulative Gain - NDCG1 / Content Based: ', ndcg(relevances_cb, k=1))
print('Normalized Discounted Cumulative Gain  / Colaborative Filtering : ', ndcg(relevances_cf,k=1))
print('Normalized Discounted Cumulative Gain  / Popularity : ', ndcg(relevances_p,k=1))
print('Normalized Discounted Cumulative Gain / Hybrid : ', ndcg(relevances_h,k=1))

#Normalized Discounted Cumulative Gain - NDCG3
print('\nNormalized Discounted Cumulative Gain - NDCG3  / Content Based: ', ndcg(relevances_cb, k=3))
print('Normalized Discounted Cumulative Gain  / Colaborative Filtering : ', ndcg(relevances_cf,k=3))
print('Normalized Discounted Cumulative Gain  / Popularity : ', ndcg(relevances_p,k=3))
print('Normalized Discounted Cumulative Gain / Hybrid : ', ndcg(relevances_h,k=3))

#Normalized Discounted Cumulative Gain - NDCG5
print('\nNormalized Discounted Cumulative Gain - NDCG5 / Content Based: ', ndcg(relevances_cb, k=5))
print('Normalized Discounted Cumulative Gain  / Colaborative Filtering : ', ndcg(relevances_cf,k=5))
print('Normalized Discounted Cumulative Gain  / Popularity : ', ndcg(relevances_p,k=5))
print('Normalized Discounted Cumulative Gain / Hybrid : ', ndcg(relevances_h,k=5))

#Normalized Discounted Cumulative Gain - NDCG10
print('\nNormalized Discounted Cumulative Gain- NDCG10 / Content Based: ', ndcg(relevances_cb, k=10))
print('Normalized Discounted Cumulative Gain  / Colaborative Filtering : ', ndcg(relevances_cf,k=10))
print('Normalized Discounted Cumulative Gain  / Popularity : ', ndcg(relevances_p,k=10))
print('Normalized Discounted Cumulative Gain / Hybrid : ', ndcg(relevances_h,k=10))
