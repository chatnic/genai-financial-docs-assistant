# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · RAG chain, MLflow logging and evaluation
# MAGIC
# MAGIC 1. Wraps retrieval (Vector Search + metadata filters) and grounded generation (Foundation
# MAGIC    Model API) in an MLflow `PythonModel`.
# MAGIC 2. Logs it to MLflow and registers it in Unity Catalog.
# MAGIC 3. Evaluates it on the 98 gold questions with **Mosaic AI Agent Evaluation**
# MAGIC    (`mlflow.evaluate(model_type="databricks-agent")`), which scores correctness against the
# MAGIC    expected answer, groundedness, and whether the expected source chunks were retrieved.
# MAGIC
# MAGIC **Status:** written for Databricks and not yet executed. The local equivalent, with
# MAGIC keyword retrieval and an extractive answerer, runs with `make all`.

# COMMAND ----------

# MAGIC %pip install -U mlflow databricks-vectorsearch databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "finrag")
dbutils.widgets.text("vs_endpoint", "finrag-vs-endpoint")
dbutils.widgets.text("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VS_ENDPOINT = dbutils.widgets.get("vs_endpoint")
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
INDEX_NAME = f"{CATALOG}.{SCHEMA}.report_chunks_index"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.financial_docs_assistant"

# COMMAND ----------

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd

repo_root = Path.cwd().parent
sys.path.insert(0, str(repo_root / "src"))
PROMPT = (repo_root / "prompts" / "answer_with_citations.md").read_text()
questions = json.loads((repo_root / "data" / "eval_questions.json").read_text())

# COMMAND ----------


class FinancialDocsRAG(mlflow.pyfunc.PythonModel):
    """Retrieve with Vector Search (filtered by company/year parsed from the question),
    then answer with a grounded, citation-required prompt."""

    def __init__(self, index_name, vs_endpoint, llm_endpoint, prompt, k=5):
        self.index_name, self.vs_endpoint, self.llm_endpoint = index_name, vs_endpoint, llm_endpoint
        self.prompt, self.k = prompt, k

    def load_context(self, context):
        from databricks.vector_search.client import VectorSearchClient
        from mlflow.deployments import get_deploy_client
        self.index = VectorSearchClient(disable_notice=True).get_index(self.vs_endpoint, self.index_name)
        self.llm = get_deploy_client("databricks")
        self.companies = ["Harborline Financial", "NovaPay Technologies", "Crestview Insurance Group",
                          "Summit Ledger Software", "Atlas Consumer Lending", "Pinecrest Wealth Partners"]

    def _filters(self, q):
        import re
        f = {}
        hit = [c for c in self.companies if c.lower() in q.lower()]
        if len(hit) == 1:
            f["company"] = hit[0]
        years = [int(y) for y in re.findall(r"fiscal\s*(20\d\d)", q.lower())]
        if "guide" in q.lower():
            years = [y - 1 for y in years]
        if years:
            f["fiscal_year"] = years if len(years) > 1 else years[0]
        return f

    def _answer(self, q):
        res = self.index.similarity_search(query_text=q, columns=["chunk_id", "text"], num_results=self.k,
                                           query_type="HYBRID", filters=self._filters(q) or None)
        rows = res["result"]["data_array"]
        context = "\n\n".join(f"[{r[0]}]\n{r[1]}" for r in rows)
        out = self.llm.predict(endpoint=self.llm_endpoint, inputs={
            "messages": [{"role": "user", "content": self.prompt.replace("{context}", context).replace("{question}", q)}],
            "temperature": 0, "max_tokens": 300})
        return {"content": out["choices"][0]["message"]["content"],
                "retrieved_context": [{"doc_uri": r[0], "content": r[1]} for r in rows]}

    def predict(self, context, model_input):
        qs = model_input["request"] if isinstance(model_input, pd.DataFrame) else [model_input["request"]]
        return [self._answer(q) for q in qs]


rag = FinancialDocsRAG(INDEX_NAME, VS_ENDPOINT, LLM_ENDPOINT, PROMPT)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name="financial-docs-rag") as run:
    mlflow.log_params({"llm_endpoint": LLM_ENDPOINT, "k": 5, "query_type": "HYBRID", "metadata_filters": True})
    mlflow.log_text(PROMPT, "prompt.md")
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=rag,
        input_example=pd.DataFrame({"request": ["What was NovaPay Technologies' total revenue in fiscal 2025?"]}),
        pip_requirements=["mlflow", "databricks-vectorsearch"],
        registered_model_name=MODEL_NAME,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate with Mosaic AI Agent Evaluation
# MAGIC Unanswerable questions expect the refusal sentence. `expected_retrieved_context` lets the
# MAGIC evaluator measure retrieval recall separately from answer quality.

# COMMAND ----------

REFUSAL = "I can't find that in the available reports."
eval_df = pd.DataFrame([{
    "request_id": q["qid"],
    "request": q["question"],
    "expected_response": REFUSAL if q["type"] == "unanswerable" else q["answer"],
    "expected_retrieved_context": [{"doc_uri": s} for s in q["gold_sources"]],
} for q in questions])

with mlflow.start_run(run_id=run.info.run_id):
    results = mlflow.evaluate(
        model=info.model_uri,
        data=eval_df,
        model_type="databricks-agent",
    )

display(results.tables["eval_results"])
print(json.dumps(results.metrics, indent=2))
