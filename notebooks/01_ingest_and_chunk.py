# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest annual reports and chunk them into a Delta table
# MAGIC
# MAGIC Reads the Markdown reports from a Unity Catalog volume, applies the same section-aware
# MAGIC chunking used locally (`src/finrag/chunking.py`), and writes a Delta table with Change Data
# MAGIC Feed enabled, which Vector Search needs for a Delta Sync index.
# MAGIC
# MAGIC **Status:** written for Databricks and not yet executed. Run it in your workspace.
# MAGIC
# MAGIC **Before running:** upload `data/corpus/*.md` to the volume below (Catalog Explorer → Volumes → Upload),
# MAGIC and add this repo to the workspace (Repos / Git folders) so `src/` can be imported.

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "finrag")
dbutils.widgets.text("volume", "raw_reports")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.report_chunks"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

# COMMAND ----------

import sys
from pathlib import Path

# The notebook lives in <repo>/notebooks; make <repo>/src importable.
repo_root = Path.cwd().parent
sys.path.insert(0, str(repo_root / "src"))

from finrag.chunking import chunk_corpus  # noqa: E402

corpus_dir = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}")
chunks = chunk_corpus(corpus_dir)
print(f"{len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents")

# COMMAND ----------

import pandas as pd  # noqa: E402

pdf = pd.DataFrame([c.to_dict() for c in chunks])
(spark.createDataFrame(pdf)
 .write.mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(CHUNKS_TABLE))

# Vector Search Delta Sync indexes require Change Data Feed on the source table.
spark.sql(f"ALTER TABLE {CHUNKS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
display(spark.table(CHUNKS_TABLE).limit(5))
