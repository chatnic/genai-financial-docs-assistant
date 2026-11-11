# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Deploy the assistant
# MAGIC
# MAGIC Deploys the registered model version to a Model Serving endpoint, with the Review App for
# MAGIC collecting stakeholder feedback and inference tables for monitoring, via `databricks.agents.deploy`.
# MAGIC
# MAGIC **Status:** written for Databricks and not yet executed.

# COMMAND ----------

# MAGIC %pip install -U databricks-agents mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "finrag")
dbutils.widgets.text("model_version", "1")

MODEL_NAME = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}.financial_docs_assistant"
VERSION = dbutils.widgets.get("model_version")

# COMMAND ----------

from databricks import agents

deployment = agents.deploy(MODEL_NAME, VERSION)
print("Serving endpoint:", deployment.endpoint_name)
print("Review App:", deployment.review_app_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query the endpoint

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
resp = client.predict(endpoint=deployment.endpoint_name,
                      inputs={"dataframe_records": [{"request": "How much cash did Harborline Financial hold at the end of fiscal 2025?"}]})
print(resp)
