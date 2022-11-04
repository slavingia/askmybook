import pandas as pd
import openai
import csv
import numpy as np
import pickle
from transformers import GPT2TokenizerFast

openai.api_key = "sk-DOiDZHHE1f1tvxnO5zs103vHelanA6BVBVO44cN7"

COMPLETIONS_MODEL = "text-davinci-002"

df = pd.read_csv('pages.csv')
df = df.set_index(["title"])
print(f"{len(df)} rows in the data.")
print(df.sample(5))

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_doc_embedding(r.content) for idx, r in df.iterrows()
    }

#    CSV with exactly these named columns:
#        "title", "0", "1", ... up to the length of the embedding vectors.

doc_embeddings = compute_doc_embeddings(df)

with open('embeddings.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["title"] + list(range(4096)))
    for i, embedding in list(doc_embeddings.items()):
        writer.writerow(["Page " + str(i + 1)] + embedding)
