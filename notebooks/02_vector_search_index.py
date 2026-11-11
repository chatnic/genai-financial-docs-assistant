# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Build a Mosaic AI Vector Search index
# MAGIC
# MAGIC Creates a Vector Search endpoint and a **Delta Sync index** over `report_chunks`, with
# MAGIC embeddings computed by a Databricks-hosted embedding model. Company and fiscal year are
# MAGIC stored as filterable columns, the same metadata filtering that made the biggest
# MAGIC difference in the local retrieval tests.
# MAGIC
# MAGIC **Status:** written for Databricks and not yet executed.

# COMMAND ----------

# MAGIC %pip install -U databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "finrag")
dbutils.widgets.text("vs_endpoint", "finrag-vs-endpoint")
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VS_ENDPOINT = dbutils.widgets.get("vs_endpoint")
EMBEDDING_ENDPOINT = dbutils.widgets.get("embedding_endpoint")
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.report_chunks"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.report_chunks_index"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT not in existing:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
vsc.wait_for_endpoint(VS_ENDPOINT)

# COMMAND ----------

existing_indexes = [i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])]
if INDEX_NAME not in existing_indexes:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=INDEX_NAME,
        source_table_name=CHUNKS_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="text",
        embedding_model_endpoint_name=EMBEDDING_ENDPOINT,
    )
else:
    index = vsc.get_index(VS_ENDPOINT, INDEX_NAME)
    index.sync()

index.wait_until_ready()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Smoke test: plain semantic search vs. metadata-filtered search

# COMMAND ----------

q = "What was NovaPay Technologies' total revenue in fiscal 2025?"
cols = ["chunk_id", "company", "fiscal_year", "section", "text"]

plain = index.similarity_search(query_text=q, columns=cols, num_results=5)
filtered = index.similarity_search(
    query_text=q, columns=cols, num_results=5,
    filters={"company": "NovaPay Technologies", "fiscal_year": 2025},
)
# Hybrid (keyword + vector) search is also available on Vector Search indexes:
hybrid = index.similarity_search(query_text=q, columns=cols, num_results=5, query_type="HYBRID",
                                 filters={"company": "NovaPay Technologies", "fiscal_year": 2025})

for name, res in [("plain", plain), ("filtered", filtered), ("hybrid+filtered", hybrid)]:
    print(name, [row[0] for row in res["result"]["data_array"]])
