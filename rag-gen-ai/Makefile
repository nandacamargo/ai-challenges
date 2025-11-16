setup:
	wget "https://huggingface.co/Mozilla/Llama-3.2-3B-Instruct-llamafile/resolve/main/Llama-3.2-3B-Instruct.Q6_K.llamafile"
	chmod +x Llama-3.2-3B-Instruct.Q6_K.llamafile

llama-local:	
	./Llama-3.2-3B-Instruct.Q6_K.llamafile --server

build:
	docker build -t rag:1.0 .
run:
	docker run --network=host -d -it --name genai-rag rag:1.0

run-jupyter:
	docker run --network=host --rm -it --name jupyterhub quay.io/jupyter/datascience-notebook

exec:
	docker exec -it  genai-rag bash

clean:
	docker container stop genai-rag && \
	docker container prune -f
	
