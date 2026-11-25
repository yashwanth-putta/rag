# RAG

## Naive RAG

1. Ingestion Pipeline
2. Inference Pipeline
3. Evaluation Pipeline

#### Pre-requisites

- data files. Have at least 5 text files.
- Have one golden set file with at least 10 questions

### 1. Ingestion Pipeline

Document Loader/Parser ==> Chunk ==> Embed ==> Store

#### Libraries/Tech stack

1. abc # abstract class
2. Json storage # store in a plain json file

#### Configuration

1. embedding model # Ollama embedding model to embed and store

### 2. Inference Pipeline

Query ==> Embed ==> Retrieve Chunks ==> Rerank ==> Top_k_chunks + Query to LLM ==> Answer/Response

#### Libraries/Tech Stack

#### Configuration

1. Embedding model should be same as ingestion.
2. Reranker model # Ollama/OpenRouter
3. K
4. LLM Model

### 3. Evaluation Pipeline

Run the evaluation script against the golden data set (Inference Pipeline) ==> LLM as Judge ==> Extract correctness and completeness metrics

#### Libraries/Tech-Stack

#### Configuration

1. LLM as Judge Model # Ollama or OpenRouter model which is better than Inference LLM Model
2. Golden dataset path

