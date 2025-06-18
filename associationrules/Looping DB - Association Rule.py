# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Installing Libraries
# MAGIC We install the **mlxtend** library, which provides various tools for machine learning and data analysis, including the apriori algorithm for association rule mining.

# COMMAND ----------

# MAGIC %pip install mlxtend

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
# MAGIC - **pandas** for data manipulation.
# MAGIC - **TransactionEncoder** and **apriori** from **mlxtend** for preprocessing transaction data and finding frequent itemsets.

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Fetching Retailers List
# MAGIC We run a SQL query to fetch the list of retailers from the prod.datascience.itens_association_rules table. We filter for retailers where the average size of items is greater than 2 and the count of items is more than 1000.

# COMMAND ----------

query = '''
  SELECT _p_retailer, 
        COUNT(items) AS item_count, 
        AVG(size(items)) AS avg_size
  FROM prod.datascience.itens_association_rules
  GROUP BY _p_retailer
  HAVING AVG(size(items)) > 2 AND count(items) > 1000
  ORDER BY avg_size DESC, item_count DESC;
'''

association_items_spark = spark.sql(query)

# COMMAND ----------

# MAGIC %md
# MAGIC We convert the Spark DataFrame to a pandas DataFrame and create a list of retailer IDs.

# COMMAND ----------

association_items = association_items_spark.toPandas()
associtation_items_list = association_items['_p_retailer'].to_list()
associtation_items_list

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Checking if Retailer has Association

# COMMAND ----------

# MAGIC %md
# MAGIC We define a function has_association to check if a given retailer has associations. The function:
# MAGIC
# MAGIC - Runs a SQL query to get the data for the given retailer.
# MAGIC - Converts the Spark DataFrame to a pandas DataFrame and structures the items into columns.
# MAGIC - Uses **TransactionEncoder** to convert the transaction data into a one-hot encoded format.
# MAGIC - Applies the **apriori** algorithm to find frequent itemsets with a minimum support of 0.1.
# MAGIC - Returns **True** if there are any frequent itemsets, otherwise **False**.

# COMMAND ----------

def has_association(retailer: str):
  query = f'''
    SELECT * FROM prod.datascience.itens_association_rules
    WHERE _p_retailer = '{retailer}'
  '''

  data = spark.sql(query)

  df = data.toPandas()
  max_len = max(len(x) for x in df['items'])
  df = pd.DataFrame(df['items'].tolist(), columns=[f'item_{i+1}' for i in range(max_len)])

  named_data = []
  for i in range(df.shape[0]):  
    named_data.append([str(df.values[i, j]) for j in range(max_len)])

  te = TransactionEncoder()
  te_array = te.fit(named_data).transform(named_data)
  df_boolean = pd.DataFrame(te_array, columns=te.columns_)
  if 'None' in df_boolean.columns:
    df_boolean.drop(columns='None', inplace=True)

  frequent_itemsets = apriori(df_boolean, min_support=0.1, use_colnames=True)

  if frequent_itemsets['support'].count() > 0:
    return True
  
  return False

# COMMAND ----------

# MAGIC %md
# MAGIC We initialize a dictionary **results** to store retailers with and without associations. We then iterate over the list of retailers, excluding specific ones, and check if each retailer has associations. The results are printed and stored in the **results** dictionary.

# COMMAND ----------

results = { 'founded': [], 'not_founded': [] }

for i, retailer in enumerate(associtation_items_list):
  if retailer not in ['GrXsrPtjK7', 'O9PYpKNuvg']:
    print(f'\n\nChecando {retailer}, Item {i}\n\n')
    result = has_association(retailer)

    if has_association(retailer):
      results['founded'].append(retailer)
    else:
      results['not_founded'].append(retailer)

    print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Logging Results
# MAGIC We print the final results showing retailers with and without associations.

# COMMAND ----------

print('Dados encontrados: ', results)

# COMMAND ----------

# MAGIC %md
# MAGIC **Libraries Overview**:
# MAGIC - **pandas**: A powerful Python library for data manipulation and analysis. It provides data structures and functions needed to manipulate structured data seamlessly.
# MAGIC - **mlxtend**: A library that provides various tools for machine learning and data analysis, including feature extraction, ensemble methods, and frequent pattern mining algorithms like apriori.
# MAGIC - **TransactionEncoder**: A tool from mlxtend used to transform transaction data into a one-hot encoded format suitable for use with algorithms like apriori.
# MAGIC - **apriori**: An algorithm from mlxtend used to find frequent itemsets in a dataset and generate association rules based on these itemsets.