# Week 6: Augmenting AI: From Memory to Retrieval

Welcome to Week 6 of Python 200! Last week we learned to interact with LLMs through APIs and to prompt them effectively. But LLMs have a fundamental limitation: their knowledge is frozen at training time. Ask one about your company's internal documents, a recent news event, or a database it has never seen, and it will either guess or hallucinate. This week we look at the main strategies for getting around that — injecting context into prompts, fine-tuning models on new data, and retrieval-augmented generation (RAG). We will spend most of our time on RAG, which is the most practical and widely used approach in production pipelines.

## Topics

1. [Intro to LLM Knowledge Augmentation](./01_llm_augmentation_intro.md)
An overview of the three main strategies for augmenting an LLM's knowledge: context injection (adding information directly to the prompt), fine-tuning (retraining the model on new data), and retrieval-augmented generation (RAG, giving the model access to an external data store at query time). We discuss the trade-offs of each before diving into RAG for the rest of the week.

2. [Keyword-based RAG](./02_keyword_rag.md)
A from-scratch RAG pipeline that gives an LLM access to a folder of PDFs using keyword search. The goal is to make the core RAG loop — retrieve relevant documents, inject them into the prompt, generate an answer — concrete before we bring in a framework.

3. [LlamaIndex](./03_llamaindex.md)
A production-ready framework for building RAG pipelines. The lesson opens with a "What LlamaIndex Automates" section that covers the concepts keyword RAG left implicit: chunking, vector embeddings, cosine similarity, and indexing. With that grounding in place, we implement a complete semantic RAG pipeline and evaluate it using LlamaIndex's built-in FaithfulnessEvaluator and RelevancyEvaluator. An optional extension covers swapping to a persistent PostgreSQL + pgvector backend.
