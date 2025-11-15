# Literature Retrieval and Summarization Tool

This project provides an end-to-end pipeline for: 1. Querying Semantic
Scholar for relevant papers. 2. Downloading and extracting text from
arXiv PDFs. 3. Cleaning and embedding metadata using
SentenceTransformers. 4. Performing vector search to retrieve the most
relevant papers. 5. Running recursive LLM-based summarization using
Hugging Face Inference API.

## Key Features

-   **Semantic Scholar search** via `semanticscholar` API.
-   **PDF download + extraction** using `requests` and `pypdf`.
-   **Embeddings** generated using `all-MiniLM-L6-v2`.
-   **Recursive summarization** pipeline with chunking and BART
    summarizer.
-   **Vector database** built directly inside a DataFrame.
-   **Automatic database update** when similarity scores are low.

## Usage

1.  Set your Hugging Face token:

    ``` bash
    export HF_TOKEN="your_token_here"
    ```

2.  Run the script:

    ``` bash
    python script.py
    ```

3.  Enter a natural-language query.\
    The system will:

    -   Update metadata if needed\
    -   Retrieve top-k relevant papers\
    -   Summarize recursively until compact

## File Structure

-   `script.py` --- Main pipeline.
-   `metadata.csv` --- Appended metadata file.
-   `database/` --- Downloads PDF files here.
-   `llm.log` --- Stores generated summaries.

## Requirements

    pip install numpy pandas regex requests pypdf semanticscholar sentence-transformers transformers huggingface_hub pypandoc

## Notes

-   Replace summarization model as needed by adjusting `model_name` and
    tokenizer.
-   Supports generalization to any HF model compatible with
    `.summarization()`.
