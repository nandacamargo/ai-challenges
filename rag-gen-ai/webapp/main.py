import os
import openai
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from math import isnan


app = FastAPI()

df = pd.read_csv('wine-ratings.csv')

# remove any NaN values as it blows up serialization
df = df[df['variety'].notna()]
# Get only 700 records. More records will make it slower to index
data = df.sample(700).to_dict('records')

# Model to create embeddings
encoder = SentenceTransformer('all-MiniLM-L6-v2') 
# create the vector database client
qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance

# Create collection to store wines
qdrant.recreate_collection(
    collection_name="top_wines",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)

# vectorize!
qdrant.upload_points(
    collection_name="top_wines",
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["notes"]).tolist(),
            payload=doc,
        ) for idx, doc in enumerate(data) # data is the variable holding all the wines
    ]
)

class Body(BaseModel):
    """
    Request body model for API endpoints that expect a query string.

    Attributes:
        query (str): The input query provided by the user.
    """
    query: str


@app.get('/')
def root():
    """
    Root endpoint that redirects to the API documentation.

    Returns:
        RedirectResponse: A 301 redirect response pointing to '/docs'.
    """
    return RedirectResponse(url='/docs', status_code=301)


@app.post('/ask')
def ask(body: Body):
    """
    Handles user queries by searching a vector database and generating a response 
    using the Llama assistant model.

    Workflow:
        1. Extracts the query string from the request body.
        2. Performs a search in the vector database using the query.
        3. Passes the query and search results to the Llama assistant.
        4. Returns the assistant's response as JSON.

    Args:
        body (Body): Request body containing the user query.

    Returns:
        dict: A JSON object with the assistant's response, in the format:
            {
                "response": <assistant_response>
            }

    Raises:
        Exception: If the search operation or assistant response fails.
    """

    try:
        search_result = search(body.query)
    except Exception as e:
        print("Could not make the search request")
        raise
    try:
        chat_bot_response = assistant(body.query, search_result)
        return {'response': chat_bot_response}
    except Exception as e:
        print("Could not get the response from the assistant")
        raise

def search(query):
    """
    Send the query to Qdrant vector database, which is used for 
    Retrieval Augmented Generation (RAG)
    Args:
        query (str): The query provided by the user
    Returns:
        search_results (list): The results of the search made in the database
    """

    hits = qdrant.search(
        collection_name="top_wines",
        query_vector=encoder.encode(query).tolist(),
        limit=3
    )
    
    # define a variable to hold the search results
    search_results = [hit.payload for hit in hits]

    print(search_results)
    return search_results


def safe_context_to_str(context):
    """
    Convert the context to a JSON formatted string or to a string directly.
    Args:
        context (list): the search results from the vector database.
    Returns:
        str: The converted string.
    """
    try:
        return json.dumps(context, ensure_ascii=False, default=str)
    except Exception:
        return str(context)


def assistant(query, context):
    """
    Creates the client that connects to Llama, as well as the text messages passed to open AI chat,
    using the roles: system, user and assistant.
    Args:
        query (str): The query provided by the user
        context (list): The search results from the vector database
    Returns:
        final_response (str): The response that will be shown to the user
    """
    messages=[
        # Set the system characteristics for this chat bot
        {"role": "system", "content": "Asisstant is a chatbot that helps you find the best wine for your taste."},

        # Set the query so that the chatbot can respond to it
        {"role": "user", "content": query},

        # Add the context from the vector search results so that the chatbot can use
        # it as part of the response for an augmented context
        {"role": "assistant", "content": safe_context_to_str(context)}
    ]

    print("==== Messages are: ===")
    print(messages)

    # Now it is time to connect to the local large language model
    from openai import OpenAI

    try: 
        client = OpenAI(
            base_url="http://localhost:8080/v1",
            api_key = "no-key-required"
        )
    except Exception as e:
        print("Problem in creating OpenAI client", e)
        raise
    
    try:
        completion = client.chat.completions.create(
            model="LLaMA-3.2-3B-Instruct.Q6_K",
            messages=messages,
            stream=False
        )
    except Exception as e:
        print("Problem when trying to use the recently created client:", e)
        raise

    print(completion)
    final_response = completion.choices[0].message.content
    print(final_response)
    return final_response
