import os
import logging

import numpy as np
import pandas as pd
import regex as re
import requests
from pypdf import PdfReader
from semanticscholar import SemanticScholar
from sentence_transformers import SentenceTransformer
from sambanova import SambaNova

logging.basicConfig(filename="llm.log", level=logging.INFO)

llm_client = SambaNova(
    api_key="somethung",
    base_url="https://api.sambanova.ai/v1",
)
embedding_model_name="all-MiniLM-L6-v2"
Embedding_Model = SentenceTransformer(embedding_model_name)

# helper functions
def clean_text(t):

    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"[^\x00-\x7F]+", " ", t)
    t = re.sub(r"[^a-z0-9\.,\n ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    t = t.replace(" .", ".")

    return t.strip()

def extract_pdf_text(pdf_path):

    try:
        reader = PdfReader(pdf_path)
        text = [
            page.extract_text() or ""
            for page in reader.pages
        ]
        return "\n".join(text)

    except Exception as exc:
        logging.warning("Failed to extract PDF text from %s: %s", pdf_path, exc)
        return ""

def read_literature(meta_dataframe):

    text_corpus = []

    for arxiv_id in meta_dataframe["arxiv"]:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        try:
            r = requests.get(pdf_url, stream=True, timeout=60)
            if r.status_code != 200:
                print(f"  -> HTTP {r.status_code}, skip")
                continue

            fname = f"{arxiv_id}.pdf"

            out_path = os.path.join("./", fname)
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            text = extract_pdf_text(out_path)
            text_corpus.append(clean_text(text))

        except Exception as e:
            print(f"  -> error: {e}")

    return text_corpus

# Building a database
def update_meta_data(query, meta_dataframe, max_results=20):

    sch = SemanticScholar()
    papers = sch.search_paper(
    query=query,
    limit=max_results,
    year="2018-",
    fields_of_study=["Computer Science", "Engineering"],
    )
    if not papers or not getattr(papers, "items", None):
        raise ValueError(f"No results returned for query: {query}")

    rows = []
    for p in papers.items:
        d = dict(p)  # force to real dict

        row = {
            "paperId": d.get("paperId"),
            "title": d.get("title"),
            "year": d.get("year"),
            "venue": d.get("venue"),
            "doi": d.get("externalIds", {}).get("DOI"),
            "arxiv": d.get("externalIds", {}).get("ArXiv"),
            "url": d.get("url"),
            "open_pdf": d.get("openAccessPdf", {}).get("url"),
            "citations": d.get("citationCount"),
            "authors": ", ".join(a["name"] for a in d.get("authors", [])),
            "abstract": d.get("abstract"),
        }
        rows.append(row)

    new_dataframe = pd.DataFrame(rows)

    new_dataframe["text_corpus"] = ("Title: " + new_dataframe["title"] \
        + "\n\n" "Abstract: " + new_dataframe["abstract"])
    new_dataframe.loc[:, "text_corpus"] = new_dataframe["text_corpus"].apply(clean_text)
    new_dataframe = new_dataframe.dropna()

    if meta_df.empty:
        return new_dataframe

    updated_meta_df = pd.concat([meta_dataframe, new_dataframe],
                                ignore_index=True).drop_duplicates(subset="paperId", keep="last")

    return updated_meta_df

def compute_embedding(text, max_tokens=200):
    """
    Returns a normalized mean embedding for the input text using specified model.
    """

    words = text.split()
    chunks = [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]
    encodings = Embedding_Model.encode(chunks, normalize_embeddings=True)
    avg_encodings = np.mean(encodings, axis=0)
    return avg_encodings / np.linalg.norm(avg_encodings)

def build_vector_database(dataframe):
    """
    Generates embeddings only from the metadata text for use in paper search and retrieval.
    """

    dataframe["encodings"] = dataframe["metadata_text"].apply(compute_embedding)

    return dataframe

def compute_similarity(doc_embed, query_vec):
    return doc_embed @ query_vec

def search_top_k_relevant_paper(query_embed, meta_dataframe, top_k=5):

    scores =  meta_dataframe["encodings"].apply(lambda emb: compute_similarity(emb, query_embed))
    idx = scores.sort_values(ascending=False).index
    meta_dataframe = meta_dataframe.loc[idx]

    text_corpus = read_literature(meta_dataframe[:top_k])

    return text_corpus

def chunk_text_for_llm(text, max_tokens=20):
    return [text]

def get_llm_response(prompt):

    resp = llm_client.chat.completions.create(
                model="Meta-Llama-3.3-70B-Instruct",
                messages=[
                    {"role": "user", "content": (prompt)
                    }],
                temperature=0.1,
                top_p=0.1
                )

    return resp

def get_paper_summary(paper_text):
    """
    Summarizes a single paper and returns the text response from LLM
    """

    chunks = chunk_text_for_llm(paper_text)

    chunk_summaries = []
    for chunk in chunks:
        instruction = f"""
        You are an expert academic assistant.
        Summarize the following chunk of a reasearch paper with 5-Sentence Summary

        Here is the content:

        {chunk}
        """

        chunk_resp = get_llm_response(instruction)
        chunk_summaries.append(chunk_resp.choices[0].message.content)

    joined_chunk_summaries = "\n\n---\n\n".join(chunk_summaries)
    final_prompt = f"""
                You are an expert academic assistant.

                Below are partial summaries of one research paper (different chunks of the same paper).

                Combine them into a single coherent summary using the same format:

                1. 5-Sentence Summary
                2. Explain Like I’m 5
                3. Key Ideas
                4. 2-Sentence for method

                Partial summaries:
                {joined_chunk_summaries}
                """
    final_resp = get_llm_response(final_prompt)

    return final_resp

def get_aggregate_summary(paper_txts):

    joined_texts = "\n\n---\n\n".join(paper_txts)

    lit_review_prompt = f"""
    You are an expert academic reviewer.

    Below is a collection of research abstracts from five papers in the same topic area.

    Your task:
    1. Read all five abstracts together.
    2. Identify the common themes, methods, and contributions.
    3. Produce a single high-level **5-sentence summary** describing the overall field.
    4. Focus on the shared concepts, not individual papers.

    Here are the abstracts:
    {joined_texts}

    Now produce the 5-sentence summary.
    """
    response = get_llm_response(lit_review_prompt)

    return response

if __name__ == "__main__":

    input_query = input('Ask me a question: ')
    logging.info(f"Query:\t {input_query}")

    # Load exsiting meta_database

    meta_path = "meta_database.csv"

    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path)
    else:
        meta_df = pd.DataFrame()

    # If similarity is very low in existing database then search online
    query_embed = compute_embedding(input_query)
    scores =  meta_df["encodings"].apply(lambda emb: compute_similarity(emb, query_embed))

    if np.min(scores) < 0.45:
        # Not enough info exist in existing database, needs to be updated

        meta_df = update_meta_data(input_query, meta_dataframe=meta_df)
        meta_df = build_vector_database(meta_df)

    # Load required knowledge useful for LLM models to generate responses
    retrieved_knowledge = search_top_k_relevant_paper(input_query, meta_df)

    # Ask models to generate response from top-k papers
    for paper_texts in range(retrieved_knowledge):
        response = get_paper_summary(retrieved_knowledge)
        logging.info(response.choices[0].message.content)
