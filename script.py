import os
import logging

import numpy as np
import pandas as pd
import regex as re
import requests
from pypdf import PdfReader
from semanticscholar import SemanticScholar
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fh = logging.FileHandler("llm.log")
logger.addHandler(fh)

Embedding_Model = SentenceTransformer("all-MiniLM-L6-v2")

model_name = "facebook/bart-large-cnn"
Tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_client = InferenceClient(model=model_name,
                            token=os.getenv("HF_TOKEN"),
                            provider="hf-inference")


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
        logger.warning("Failed to extract PDF text from %s: %s", pdf_path, exc)
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

            out_path = os.path.join("./database/", fname)
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

    new_dataframe["metadata_text"] = ("Title: " + new_dataframe["title"] \
        + "\n\n" "Abstract: " + new_dataframe["abstract"])
    new_dataframe = new_dataframe.dropna()
    new_dataframe.loc[:, "metadata_text"] = new_dataframe["metadata_text"].apply(clean_text)

    if meta_dataframe.empty:
        return new_dataframe

    updated_meta_df = pd.concat([meta_dataframe, new_dataframe],
                                ignore_index=True).drop_duplicates(subset="paperId", keep="last")

    return updated_meta_df

def compute_embedding(text, max_words=200):
    """
    Returns a normalized mean embedding for the input text using specified model.
    """

    words = text.split()
    chunks =  [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

    # get average embeding
    embeds = Embedding_Model.encode(chunks, normalize_embeddings=True)
    avg_encodings = np.mean(embeds, axis=0)

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

def chunk_text_for_llm(text, max_tokens=512, overlap=64):

    tokens = Tokenizer.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = Tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

        start += max_tokens - overlap  # advance with overlap

    return chunks

def get_llm_response(prompt):

    resp = llm_client.summarization(prompt)

    return resp

def get_paper_summary(paper_txt, max_round=5, max_char=3000):
    """
    Summarizes a single paper and returns the text response from LLM
    """

    joint_txt = paper_txt
    for _ in range(max_round):
        chunks = chunk_text_for_llm(joint_txt)
        chunk_summaries = []
        for chunk in chunks:
            instruction = f"""{chunk}"""

            chunk_resp = get_llm_response(instruction)
            chunk_summaries.append(chunk_resp["summary_text"])

        joint_txt = "\n".join(chunk_summaries)

        if len(joint_txt) <=max_char:
            break

    return joint_txt

if __name__ == "__main__":

    input_query = input('Ask me a question: ')
    logger.info("\n\nQuery:\t %s\n", input_query)

    # # Load exsiting meta_database

    meta_path = "meta_database.csv"

    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path)
    else:
        meta_df = pd.DataFrame()

    # If similarity is very low in existing database then search online
    query_embed = compute_embedding(input_query)

    scores = -1
    if not meta_df.empty:
        scores =  meta_df["encodings"].apply(lambda emb: compute_similarity(emb, query_embed))

    if np.min(scores) < 0.45:
        # Not enough info exist in existing database, needs to be updated
        meta_df = update_meta_data(input_query, meta_dataframe=meta_df)
        meta_df = build_vector_database(meta_df)
    meta_df.to_csv("metadata.csv", mode="a", header=False, index=False, encoding="utf-8")

    # Load required knowledge useful for LLM models to generate responses
    retrieved_knowledge = search_top_k_relevant_paper(query_embed, meta_df, top_k=1)

    # Ask models to generate response from top-k papers
    for n, paper_texts in enumerate(retrieved_knowledge):
        response = get_paper_summary(paper_texts)
        logger.info(f"Summary of Paper: {n}")
        logger.info(response)
