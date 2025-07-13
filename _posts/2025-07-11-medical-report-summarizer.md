---
layout: post
title: Medical Report Summarizer (Gen AI)
image: "/posts/medical-summary-title-img.png"
tags: [OpenAI, LangChain, Streamlit, GPT-4, Summarization, Healthcare, PDF, LLM, NLP]
---

**Medical Report Summarizer (Gen AI)** is a clinical NLP application that helps convert lengthy and technical medical reports into patient-friendly summaries. It empowers patients and doctors to communicate better by making complex clinical narratives easier to understand using generative AI. This project demonstrates how to build a reliable medical summarization assistant using OpenAI, LangChain, and Streamlit.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Tools & Technologies Used](#2-tools--technologies-used)
- [3. Project Workflow](#3-project-workflow)
- [4. System Architecture](#4-system-architecture)
- [5. Full Source Code](#5-full-source-code)
  - [a. Setup & UI](#a-setup--ui)
  - [b. PDF Text Extraction](#b-pdf-text-extraction)
  - [c. Text Chunking](#c-text-chunking)
  - [d. Summarization Pipeline](#d-summarization-pipeline)
- [6. Future Enhancements](#6-future-enhancements)
- [7. Setup Instructions](#7-setup-instructions)

---

# 1. Project Overview <a name="1-project-overview"></a>

In this project, we build an intelligent medical assistant that automatically summarizes long clinical documents or lab reports into plain language for patients. It uses GPT-based LLMs to process PDF content, identify important findings, and rewrite them in a clear, human-readable way. We named it **Medical Report Summarizer** to reflect its goal of improving healthcare communication through Gen AI.

---

# 2. Tools & Technologies Used <a name="2-tools--technologies-used"></a>

- Streamlit – Interactive front-end UI
- PyPDF2 – PDF content extraction
- LangChain – LLM integration and pipeline chaining
- OpenAI GPT-4 – LLM for medical summarization

---

# 3. Project Workflow <a name="3-project-workflow"></a>

1. User uploads a medical report in PDF format
2. Text is extracted using PyPDF2
3. The document is split into logical chunks
4. Each chunk is passed to an LLM summarization chain
5. Patient-friendly summary is returned and displayed

---

# 4. System Architecture <a name="4-system-architecture"></a>

This system uses a simple extract-transform-generate flow:

1. **Document ingestion** – PDF loaded and parsed
2. **Chunking** – Logical separation for length control
3. **LLM prompt & summarization** – Prompted using a LangChain summarization chain
4. **Result aggregation** – Chunks recombined into a final summary

The Medical Report Summarizer system performs fully automated summarization of uploaded medical PDFs without requiring any user questions. Once a user uploads a report, the system extracts the text, divides it into manageable chunks, and sequentially passes each chunk through a large language model (such as GPT-4). The model then generates a simplified, patient-friendly summary, which is presented directly to the user. This streamlined, no-input-required process is illustrated in the system architecture figure below, highlighting the flow from document ingestion to summary generation.

![alt text](/img/posts/medical-report-summarizer-architecture.png "Architecture diagram")

---

# 5. Full Source Code <a name="5-full-source-code"></a>

## a. Setup & UI <a name="a-setup--ui"></a>
```python
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "your-api-key"

st.set_page_config(page_title="Medical Report Summarizer")
st.header("Medical Report Summarizer")

with st.sidebar:
    st.title("Upload Clinical Report")
    file = st.file_uploader("Upload a medical PDF report", type="pdf")
```

## b. PDF Text Extraction <a name="b-pdf-text-extraction"></a>
```python
if file is not None:
    pdf_reader = PdfReader(file)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()
```

## c. Text Chunking <a name="c-text-chunking"></a>
```python
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=1500,
        chunk_overlap=200
    )
    chunks = text_splitter.create_documents([raw_text])
```

## d. Summarization Pipeline <a name="d-summarization-pipeline"></a>
```python
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0.3,
        model_name="gpt-4"
    )

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(chunks)
    st.subheader("Patient-Friendly Summary")
    st.write(summary)
```

---

# 6. Future Enhancements <a name="6-future-enhancements"></a>

- Add multilingual summarization support
- Fine-tune on actual discharge summaries (e.g., MIMIC-III)
- Add explanation highlights with expandable sections
- Incorporate question-answering mode for deeper exploration
- Integrate voice output for accessibility

---

# 7. Setup Instructions <a name="7-setup-instructions"></a>

1. Clone this repository
2. Install dependencies:
```bash
pip install streamlit PyPDF2 langchain openai
```
3. Add your OpenAI API key
4. Run the app:
```bash
streamlit run app.py
