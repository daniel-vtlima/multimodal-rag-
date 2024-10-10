
# Multimodal Retrieval-Augmented Generation (RAG) with Jina and GPT-4

This project implements a multimodal Retrieval-Augmented Generation (RAG) pipeline. It processes documents, embeds their content using Jina embeddings, and queries the GPT-4o model to generate accurate answers based on retrieved document information.

## Features
- **Document Loading**: Load and preprocess documents using high-resolution strategies.
- **Embeddings and Vector Store**: Generate embeddings for the document content using Jina embeddings and store them in a vector database (Qdrant).
- **GPT-4o Querying**: Ask questions about the documents and retrieve answers using OpenAI's GPT-4o model.
- **Multimodal Integration**: Built to handle both structured and unstructured data in RAG workflows.

## Project Structure

```
multimodal_rag/
├── data/                        # Directory containing data files
├── .env                         # Environment variables file (API keys)
├── multimodal_rag_jina.py        # Main Python script for running RAG with Jina embeddings
├── poetry.lock                   # Poetry lock file for package management
├── preprocess_docs.py            # Module for loading, embedding, and managing documents
├── pyproject.toml                # Poetry project configuration file
├── README.md                     # Project documentation
└── utils.py                      # Utility functions, including environment validation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/daniel-vtlima/multimodal-rag.git
   cd multimodal-rag
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up environment variables. Create a `.env` file in the project root with the following contents:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   JINA_API_KEY=your_jina_api_key
   ```

## Usage

### 1. Running the Python Script

You can also run the RAG pipeline directly via the `multimodal_rag_jina.py` script. This script performs the following operations:
- Loads and preprocesses the document.
- Embeds the document content using Jina embeddings.
- Adds the embeddings to a Qdrant vector store.
- Queries GPT-4o based on retrieved documents.

To run:
```bash
python multimodal_rag_jina.py
```

### 2. Preprocessing and Document Management

The `preprocess_docs.py` module provides functions for:
- **`load_docs(file_path)`**: Load documents from a specified file.
- **`set_embeddings(doc, collection_name)`**: Set up embeddings and store them in Qdrant.
- **`add_doc(vector_store, doc, ids)`**: Add document embeddings to the vector store.

### Example Workflow
Here’s a typical workflow:

1. Load the document using `load_docs`.
2. Set embeddings using `set_embeddings` and add them to the vector store.
3. Query GPT-4o:
   ```python
   query_model = "How many pages does the document have?"
   results = vector_store.similarity_search(query_model, k=3)
   answer = generate_answer_with_gpt4(query_model, results)
   print(answer)
   ```

## Environment Variables

The `.env` file should contain the following keys:
- **OPENAI_API_KEY**: Your API key for OpenAI GPT models.
- **JINA_API_KEY**: Your API key for Jina embeddings.

## Dependencies

All dependencies are managed with Poetry and listed in the `pyproject.toml` file. You can install them using:

```bash
poetry install
```

Key dependencies include:
- `openai`
- `loguru`
- `qdrant-client`
- `jinja-embeddings`
- `langchain`

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [OpenAI](https://openai.com/) for GPT-4o.
- [Jina AI](https://www.jina.ai/) for embedding models.
- [Qdrant](https://qdrant.tech/) for their vector store technology.
