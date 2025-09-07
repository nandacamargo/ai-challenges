## RAG lab

This RAG (Retrieval-Augmented Generation) project is a challenge given in Introduction to Generative AI course from Duke University.

The current lab uses the Qdrant vector database alongside with Llamafile for the LLM solution, using OpenAI APIs.

The goal was to use the code from [repo1](https://github.com/alfredodeza/azure-rag) and [repo2](https://github.com/alfredodeza/learn-retrieval-augmented-generation) as inspiration and as a starting point (the `wine-ratings.csv` is from there) and to adapt the solution to instead of using Azure, integrate with a local LLM as Llama.
It was necessary to produce the embeddings and load them to the vector database when the web application starts. And, to check everything is working, we need to start the web application and navigate to the /docs URL and interacting with the /ask endpoint, at `localhost:8000/docs/`
