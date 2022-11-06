import pandas as pd
import openai
import numpy as np
from transformers import GPT2TokenizerFast

import boto3


openai.api_key = "sk-DOiDZHHE1f1tvxnO5zs103vHelanA6BVBVO44cN7"

COMPLETIONS_MODEL = "text-davinci-002"

s3 = boto3.client(
    's3',
    aws_access_key_id="AKIAYXDBVZ7BCW3GF67W",
    aws_secret_access_key="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"
)

s3.download_file('askbook', 'pages.csv', 'pages.csv')
s3.download_file('askbook', 'embeddings.csv', 'embeddings.csv')

df = pd.read_csv('pages.csv')
df = df.set_index(["title"])

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title"])
    return {
           (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

document_embeddings = load_embeddings("embeddings.csv")

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))

f"Context separator contains {separator_len} tokens"

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        print(section_index)
        print(df.loc[section_index])
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content)
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    header = """Sahil Lavingia is the founder and CEO of Gumroad, a company he has been running since 2011. In 2021, he published a book titled The Minimalist Entrepreneur. He is often asked questions about it from aspiring entrepreneurs and business owners. Answer the question as best as you can using the provided context from this book about entrepreneurship, and if the answer is not contained within the provided context, do your best to extrapolate the answer from other intelligent technology leaders and thinkers. Answer from the perspective of Sahil Lavingia. Please keep your answers to three sentences maximum, and speak in complete sentences.\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Question: " + question + "\n Answer from Sahil Lavingia:"

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 75,
    "model": COMPLETIONS_MODEL,
}

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    print("===\n", prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")

print(answer_query_with_context("How to choose what business to start?", df, document_embeddings))
# print(answer_query_with_context("Should we start the business on the side first or should we put full effort right from the start?", df, document_embeddings))
# print(answer_query_with_context("Should we sell first than build or the other way around?", df, document_embeddings))
# print(answer_query_with_context("Andrew Chen has a book on this so maybe touché, but how should founders think about the cold start problem? Businesses are hard to start, and even harder to sustain but the latter is somewhat defined and structured, whereas the former is the vast unknown. Not sure if it’s worthy, but this is something I have personally struggled with.", df, document_embeddings))
# print(answer_query_with_context("What is one business that you think is ripe for a minimalist Entrepreneur innovation that isn’t currently being pursued by your community?", df, document_embeddings))
# print(answer_query_with_context("How can you tell if your pricing is right? If you are leaving money on the table?", df, document_embeddings))
# print(answer_query_with_context("Why is the name of your book 'the minimalist entrepreneur' ?", df, document_embeddings))
# print(answer_query_with_context("How long it takes to write TME?", df, document_embeddings))
# print(answer_query_with_context("What is the best way to distribute surveys to test my product idea?", df, document_embeddings))
# print(answer_query_with_context("How do you know, when to quit?", df, document_embeddings))