# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Installing Libraries
# MAGIC We provide an overview of the **mlxtend** library, which will be used for association rule mining.

# COMMAND ----------

!pip install mlxtend 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Importing Libraries

# COMMAND ----------

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# COMMAND ----------

# MAGIC %md
# MAGIC We import the necessary libraries:
# MAGIC
# MAGIC - **pandas** for data manipulation.
# MAGIC - **TransactionEncoder** and **apriori** from **mlxtend** for preprocessing transaction data and finding frequent itemsets.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Fetching Retailer
# MAGIC We run a SQL query to fetch a retailer from the prod.datascience.itens_association_rules table.

# COMMAND ----------

query = '''
  SELECT * FROM prod.datascience.itens_association_rules
  WHERE _p_retailer = 'GrXsrPtjK7'
'''

data = spark.sql(query)

# COMMAND ----------

# MAGIC %md
# MAGIC We display the first 20 rows of the fetched data as a pandas DataFrame.

# COMMAND ----------

data.toPandas().head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Checking if Retailer has Association

# COMMAND ----------

# MAGIC %md
# MAGIC We convert the Spark DataFrame to a pandas DataFrame and structure the items into columns.

# COMMAND ----------

df = data.toPandas()
max_len = max(len(x) for x in df['items'])
df = pd.DataFrame(df['items'].tolist(), columns=[f'item_{i+1}' for i in range(max_len)])
df

# COMMAND ----------

# MAGIC %md
# MAGIC We print the maximum length of the item lists.

# COMMAND ----------

print(max_len)

# COMMAND ----------

# MAGIC %md
# MAGIC We convert the items into a list of lists, where each sublist represents a transaction.

# COMMAND ----------

named_data = []
for i in range(df.shape[0]):  
    named_data.append([str(df.values[i, j]) for j in range(max_len)])

# COMMAND ----------

# MAGIC %md
# MAGIC We print each transaction.

# COMMAND ----------

for item in named_data:
  print(item, '\n\n')

# COMMAND ----------

# MAGIC %md
# MAGIC We use **TransactionEncoder** to convert the transaction data into a one-hot encoded format and remove any columns with 'None'.

# COMMAND ----------

te = TransactionEncoder()
te_array = te.fit(named_data).transform(named_data)
df_boolean = pd.DataFrame(te_array, columns=te.columns_)

if 'None' in df_boolean.columns:
  df_boolean.drop(columns='None', inplace=True)

df_boolean

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Logging Results
# MAGIC We apply the apriori algorithm to find frequent itemsets with a minimum support of 0.1 and display the results.

# COMMAND ----------

frequent_itemsets =  apriori(df_boolean, min_support=0.1, use_colnames=True)
frequent_itemsets

# COMMAND ----------

# MAGIC %md
# MAGIC **Libraries Overview:**
# MAGIC - **pandas**: A powerful Python library for data manipulation and analysis. It provides data structures and functions needed to manipulate structured data seamlessly. Here, it is used to handle and analyze the transaction data.
# MAGIC - **mlxtend**: A library that provides various tools for machine learning and data analysis, including the apriori algorithm for association rule mining. It is used to preprocess transaction data and find frequent itemsets.
# MAGIC - **TransactionEncoder**: A utility from mlxtend used to convert transaction data into a one-hot encoded format, making it suitable for association rule mining.
# MAGIC - **apriori**: An algorithm from **mlxtend** used for finding frequent itemsets in transaction data, which is the first step in association rule mining.