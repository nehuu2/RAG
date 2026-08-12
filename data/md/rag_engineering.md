# RAG Engineering Basics

## What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation (RAG) is a pattern in which a system first retrieves relevant passages from a document collection and then gives those passages to a language model as context for generating an answer. The goal is to ground the answer in the supplied documents rather than relying only on the model's internal knowledge.

A typical RAG flow is: load documents, split them into chunks, create embeddings, store vectors and metadata, retrieve the most relevant chunks for a query, and generate an answer from the retrieved context.

## Chunking

Chunking divides long documents into smaller pieces that can be embedded and retrieved. Chunk size should be large enough to preserve useful context but small enough to keep retrieval focused. Chunk overlap repeats a small portion between adjacent chunks so information near a boundary is less likely to be separated.

A practical starting point is a chunk size of about 800 to 1200 characters with an overlap of about 100 to 200 characters. These are starting defaults, not universal rules.

## Embeddings and Vector Search

An embedding model converts text into a numerical vector. Texts with similar semantic meaning tend to have vectors that are close under an appropriate similarity measure. A vector store keeps the embeddings and associated metadata and can return the nearest chunks for a query.

For a reproducible RAG system, record the embedding model name and vector dimensionality.

## Top-k Retrieval and Thresholds

Top-k retrieval returns the k highest-ranked candidate chunks. The value of k is a parameter because different applications need different amounts of context. A similarity threshold can be used as a second guard: if every candidate is below the threshold, the system should treat the query as having no sufficiently relevant context.

## Grounded Answers and Citations

A grounded generator should use retrieved passages as evidence. Each answer should identify the source chunks that support it. If the retrieved context does not contain enough information, the system should say that the available documents do not provide enough information instead of inventing facts.

## Metadata Filtering

Useful metadata can include source filename, file type, page number, document identifier, and chunk identifier. Metadata filters can restrict retrieval to a particular source or file type before semantic ranking.

## Evaluation

Retrieval quality can be measured using Recall@k or Hit Rate, Mean Reciprocal Rank (MRR), nDCG@k, and context precision. Answer quality can be evaluated using faithfulness or groundedness and answer relevance. A fixed evaluation set makes results reproducible.

Latency should be measured separately for retrieval and the total query. Percentiles such as p50 and p95 help show typical and slower cases.

## Cost and Scale

A cost comparison should state assumptions. Vector count, embedding dimensionality, storage, query rate, compute, and any managed-service pricing assumptions affect the result. A low-cost local vector store can be attractive when the workload is large but lightly queried, provided operational requirements remain acceptable.
