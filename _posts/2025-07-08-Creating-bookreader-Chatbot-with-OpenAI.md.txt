---
layout: post
title: Bookreader Chatbot (Gen AI)
image: "/posts/OpenAI-img.png"
tags: [OpenAI, LangChain, FAISS, GPT-3.5, Vector Search, PDF QA, Document Chatbot, Streamlit, Embeddings, NLP]
---

This project demonstrates how to build a **chatbot that can answer questions from PDF documents** using OpenAI's GPT model, FAISS for vector search, and Streamlit for the user interface.

---

## Table of Contents

- [00. Project Overview](#00-project-overview)
- [01. Tools & Technologies Used](#01-tools--technologies-used)
- [02. Project Workflow](#02-project-workflow)
- [03. System Architecture](#03-system-architecture)
- [04. Full Source Code](#04-full-source-code)
- [05. Future Enhancements](#05-future-enhancements)
- [06. Setup Instructions](#06-setup-instructions)

---

# 00. Project Overview <a name="00-project-overview"></a>

This app allows users to upload a PDF document and ask natural language questions about its content. The application performs the following steps:

1. Extracts text from the uploaded PDF.
2. Splits the text into chunks.
3. Generates vector embeddings using OpenAI.
4. Stores vectors in a FAISS index.
5. Retrieves relevant chunks based on the user's question.
6. Uses an OpenAI LLM (GPT-3.5-turbo) to answer the question based on retrieved context.

---

# 01. Tools & Technologies Used <a name="01-tools--technologies-used"></a>

- Streamlit – Web app UI
- PyPDF2 – PDF text extraction
- LangChain – Framework to handle LLM chains
- OpenAI GPT-3.5 – LLM for answer generation
- FAISS – Vector similarity search
- OpenAIEmbeddings – For embedding generation

---

# 02. Project Workflow <a name="02-project-workflow"></a>

1. User uploads a PDF via the sidebar
2. PDF text is extracted using PyPDF2
3. The text is split into manageable chunks
4. Chunks are embedded and indexed using FAISS
5. User types a question
6. Relevant chunks are retrieved
7. A prompt is constructed and passed to the OpenAI LLM
8. The generated answer is displayed

---

# 03. System Architecture <a name="03-system-architecture"></a>

The system follows a classic Retrieval-Augmented Generation (RAG) architecture:

1. **Document ingestion and preprocessing** – extract text and chunk it.
2. **Embedding and vector indexing** – transform chunks into embeddings and store in FAISS.
3. **Question embedding + retrieval** – user query is embedded and compared to the vector index.
4. **Context construction + LLM query** – top-k results are inserted into a prompt.
5. **Response generation** – the LLM (OpenAI GPT-3.5) answers the question using the retrieved context.

Below is the architecture diagram:

![alt text](/img/posts/Creating Chatbot - Architecture.png "Architecture diagram")

---

# 04. Full Source Code <a name="04-full-source-code"></a>

```python
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "your-api-key"  # Replace with your own secure method

st.header("My bookreader Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Type your question here")

    if user_question:
        match = vector_store.similarity_search(user_question)

        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)

        st.write(response)
```

---

# 05. Future Enhancements <a name="05-future-enhancements"></a>

- Add file type support for `.docx`, `.txt`, `.csv`
- Secure API key handling with `.env`
- Display source chunk for transparency
- Enable multi-file document analysis
- Cache embeddings to reduce cost and latency

---

# 06. Setup Instructions <a name="06-setup-instructions"></a>

1. Clone the repository
2. Install dependencies:

```bash
pip install streamlit PyPDF2 langchain openai faiss-cpu
```

3. Set your OpenAI key in the code or `.env` file
4. Run the app:

```bash
streamlit run app.py
```
