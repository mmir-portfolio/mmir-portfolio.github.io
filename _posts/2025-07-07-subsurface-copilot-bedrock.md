---
layout: post
title: Subsurface Engineering Copilot (LLM + RAG)
image: "/posts/Amazon-bedrock.png"
tags: [Amazon Bedrock, Claude, Titan Embeddings, OpenSearch, Vector Search, RAG, LLM, Semantic Search, Prompt Engineering]
---

In this project, we build a Q&A assistant that helps subsurface engineers extract actionable insights from technical documents (PDFs, reports, production logs) using **Amazon Bedrock**, **vector embeddings**, and **semantic search**.


# Table of Contents

- [00. Project Overview](#overview-main)
    - [Problem Statement](#problem-statement)
    - [Solution](#problem-solution)
    - [Architecture Overview](#architecture-overview)
- [01. Dataset Overview](#data-overview)
- [02. Feature Engineering](#fe-overview)
    - [Vector Embedding](#vector-embedding)
    - [Text Chunking](#text-chunking)
- [03. Model Selection & Training](#model-selection-application)
- [04. Real-Time Optimization Engine](#real-time-opt-engine) 
    - [Querying with LLM](#query-llm)
    - [Frontend with Streamlit](#frontend-streamlit)
- [05. Sample Result](#sample-result)

---

# Project Overview <a name="overview-main"></a>

### Problem Statement  <a name="problem-statement"></a>
Subsurface engineers often need to sift through thousands of pages of CMG simulation reports, drilling summaries, and operational logs to answer specific questions. This is time-consuming and inefficient.

### Solution  <a name="problem-solution"></a>
We create an LLM-powered assistant using:
- **Amazon Bedrock** (Claude or Titan) for generating grounded responses
- **Amazon Titan Embeddings** for semantic search
- **OpenSearch** for fast vector similarity search
- **Streamlit** for user interface

### Architecture Overview <a name="architecture-overview"></a>
```
User (Q) → Streamlit App → Vector Search (OpenSearch)
                                ↓
        Retrieved Chunks + Question → Amazon Bedrock (Claude/Titan)
                                ↓
                        Answer with Document Reference
```

---

# 1.Dataset Overview <a name="data-overview"></a>

We use a collection of synthetic and real-world petroleum engineering documents such as:
- CMG simulation reports
- Reservoir characterization studies
- Drilling and completion summaries

Documents are split into overlapping chunks and stored with vector embeddings.

---

# 2.Feature Engineering <a name="fe-overview"></a>

### Vector Embedding <a name="vector-embedding"></a>
Amazon Titan Embeddings are used to convert document chunks and queries into high-dimensional vectors.

### Text Chunking <a name="text-chunking"></a>
```python
import fitz  # PyMuPDF

def extract_chunks(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
```

---

# 3.Model Selection & Training <a name="model-selection-application"></a> 

### 🔧 Embedding + Storage
```python
import boto3

def get_embedding(text):
    bedrock = boto3.client("bedrock-runtime")
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body={"inputText": text},
        contentType="application/json"
    )
    return response["embedding"]
```

```python
from opensearchpy import OpenSearch

def init_index(index_name):
    client = OpenSearch("https://localhost:9200")
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body={
            "settings": {"index.knn": True},
            "mappings": {
                "properties": {
                    "chunk": {"type": "text"},
                    "embedding": {"type": "knn_vector", "dimension": 1536}
                }
            }
        })
    return client
```

---

# 4.Real-Time Optimization Engine model-selection-application <a name="real-time-opt-engine"></a>

### Querying with LLM <a name="query-llm"></a>
```python
import boto3

def query_bedrock(prompt):
    client = boto3.client("bedrock-runtime")
    response = client.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body={"prompt": prompt, "max_tokens": 500},
        contentType="application/json"
    )
    return response["completion"]
```

### Frontend with Streamlit <a name="frontend-streamlit"></a>
```python
import streamlit as st
from ingest.chunker import extract_chunks
from ingest.embedder import get_embedding
from search.opensearch_handler import init_index, search_similar
from qa.bedrock_query import query_bedrock

client = init_index("docs")
st.title("🛢️ Subsurface Copilot")

query = st.text_input("Ask your reservoir question:")

if query:
    query_emb = get_embedding(query)
    results = search_similar(client, query_emb)
    docs = "\n".join([hit["_source"]["chunk"] for hit in results["hits"]["hits"]])
    prompt = f"""
You are a petroleum reservoir expert.
Based on the following context:
{docs}
Answer the question: {query}
"""
    answer = query_bedrock(prompt)
    st.markdown(f"### 🧠 Answer\n{answer}")
```

---

# 5.Sample Result <a name="sample-result"></a>
**Input**:
> What is the expected oil recovery after 5 years under WAG injection?

**Response**:
> Based on the simulation data provided, the cumulative oil recovery after 5 years under WAG injection is approximately 36%, assuming optimal gas-water ratio and uniform areal sweep.











