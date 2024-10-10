"""
This module provides functionality for loading documents, setting up embeddings, and managing
vector data in a vector store using Qdrant and Jina models. It contains three primary functions:
- `load_docs`: Loads documents from a given file using the UnstructuredLoader, designed to handle
  high-resolution document processing.
- `set_embeddings`: Initializes a Jina embeddings model and a Qdrant vector store for embedding and
  managing vector data, creating a collection and generating unique document IDs.
- `add_doc`: Adds the processed documents to the vector store while handling various exceptions
  such as connection or type errors.

The module utilizes environment variables for API keys and in-memory vector storage for managing
the vectorized representations of document content.
"""
import os
from uuid import uuid4
from loguru import logger
from langchain_unstructured import UnstructuredLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def load_docs(file_path: str):
    """
    Load documents from a file using the UnstructuredLoader.

    Args:
        file_path (str): The file path to load the documents from.

    Returns:
        list: A list of loaded document objects containing content and metadata.

    This function uses the `UnstructuredLoader` to load the documents from the specified file path.
    The loader is configured with a high-resolution strategy, partitioning via API, and includes
    coordinates.
    """
    loader = UnstructuredLoader(
        file_path, strategy="hi_res", partition_via_api=True, coordinates=True
    )
    doc = loader.load()
    logger.success(f"Document {file_path}sucessfully loaded.")

    return doc


def set_embeddings(doc, collection_name):
    """
    Set up embeddings, vector store, and client for managing and storing vector data.

    Args:
        doc (list): A list of Document objects to be embedded.
        collection_name (str): Name of the collection for storing vectors.

    The function initializes the Jina embeddings model using the API key from environment variables.
    It creates an in-memory Qdrant client, sets up a collection for storing vectors, and initializes
    a QdrantVectorStore instance. UUIDs are generated for each document in the provided list.

    Returns:
        tuple: A tuple containing the vector store and the generated UUIDs for the documents.
    """
    logger.info("Setting Jina Model.")
    embeddings = JinaEmbeddings(
        jina_api_key=os.environ["JINA_API_KEY"], model_name="jina-embeddings-v2-base-en"
    )

    logger.info("Creating collection and setting Client.")
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    logger.info("Setting Vector Store object.")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    logger.info("Generating ids for the document content.")
    uuids = [str(uuid4()) for _ in range(len(doc))]
    logger.success("Qdrant Embedding Environment set!")

    return vector_store, uuids


def add_doc(vector_store, doc, ids):
    """
    Add documents to the vector store with the given IDs.

    Args:
        vector_store: An instance of QdrantVectorStore or similar that supports adding documents.
        doc (list): A list of document objects to be added.
        ids (list): A list of unique IDs corresponding to each document in `doc`.

    Returns:
        None
    """
    file_name = doc[0].metadata.get("filename")
    try:
        vector_store.add_documents(documents=doc, ids=ids)
        logger.success(f"Document {file_name} added to vector store")
    except ValueError as e:
        logger.error(f"There's an error while handling the document list: {e}")
    except TypeError as e:
        logger.error(f"Type error while handling document list: {e}")
    except ConnectionError as e:
        logger.error(f"Connection error while communicating with Vector Store: {e}")
    except Exception as e:
        logger.error(f"Unexpected error ocurred whhile adding document {file_name}: {e}")
