"""
This module provides functionality to load a document, process embeddings, and query the GPT-4o 
model for answers based on retrieved document information. 

Key functionalities:
- Load documents using the `load_docs` function from an external module.
- Set up embeddings and vector store using `set_embeddings`, and add documents to the vector 
  store with `add_doc`.
- Generate answers to queries using the GPT-4o model based on the document content and embeddings.

The module integrates with OpenAI's GPT-4o for generating answers and uses environment variables 
for managing API keys and configurations.
"""
import os
import openai
from dotenv import load_dotenv
from loguru import logger

from preprocess_docs import load_docs, set_embeddings, add_doc
from utils import check_keys

load_dotenv()
check_keys()

def generate_answer_with_gpt4(query, retrieved_docs):
    """
    Generate an answer using the GPT-4o model based on the provided query and retrieved documents.

    Args:
        query (str): The question or query to be answered.
        retrieved_docs (list): A list of document objects from which information will be extracted 
        to form the context.

    This function builds a prompt by extracting content from the retrieved documents, including their 
    page numbers. The prompt is then sent to the GPT-4o model to generate a detailed and accurate 
    answer based on the provided context.

    Returns:
        str: The generated answer from the GPT-4o model.
    """
    context = "\n\n".join(
        [
            f"Page {doc.metadata.get('page_number', 'Unknown')}: {doc.page_content}"
            for doc in retrieved_docs
        ]
    )
    prompt = (
        f"You are an AI assistant helping answer questions based on provided context. The context contains information extracted from multiple pages of a document. "
        f"Please use this context to provide a detailed and accurate response to the question.\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message.content


openai.api_key = os.environ["OPENAI_API_KEY"]
FILE_PATH = "./data/relatorio_b3_2023.pdf"
logger.info(f"Loading document at {FILE_PATH}.")
doc = load_docs(FILE_PATH)

vector_store, uuids = set_embeddings(doc=doc, collection_name="qdrant_test")
logger.info("Adding Documents to vector store.")
add_doc(vector_store, doc, uuids)

# Query the model about the document in question
query_model = "How many pages does the document have?"
results = vector_store.similarity_search(query_model, k=3)

# Generate answer using GPT-4
answer = generate_answer_with_gpt4(query_model, results)
logger.success(f"Answer: {answer}")
