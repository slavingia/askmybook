from django.shortcuts import render
from django.http import HttpResponse

from .models import Question

import pandas as pd
import openai
import numpy as np

from resemble import Resemble

Resemble.api_key('0vWhLtB2fmjVIE0Nuzic5wtt')
openai.api_key = "sk-DOiDZHHE1f1tvxnO5zs103vHelanA6BVBVO44cN7"

COMPLETIONS_MODEL = "text-davinci-002"

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
separator_len = 3

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 75,
    "model": COMPLETIONS_MODEL,
}

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

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content)
        chosen_sections_indexes.append(str(section_index))

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    header = """Context:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

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

def index(request):
    return render(request, "index.html")

def ask(request):
    question = request.POST.get("question", "")

    previous_question = Question.objects.filter(question=question).first()
    previous_answer = previous_question.answer if previous_question else None

    if previous_answer:
        previous_question.ask_count = previous_question.ask_count + 1
        previous_question.save()
        return render(request, "answer.html", { "answer": previous_answer, "question": question, "audio_src_url": previous_question.audio_src_url })

    df = pd.read_csv("https://askbook.s3.us-east-1.amazonaws.com/pages.csv?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJHMEUCIQCosaeIJrUiCuzmLKO2PgXleWPusM85rA0iDsUqZmUNWQIgPwJAqgNu%2BtJyEfD0m0zvIWpgBWyrwtNH449DpPbHJDEq7QII3f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgw1OTkzNTI3OTA5NzgiDMyp840t9%2FPinfvxQSrBAh0gXgLija9B8WABaoVjfJlR%2B11eQiks2cxjW0YVeJ4KJYqaS8gkO0tZ%2FcoSnsoQ%2FdmlLuv6z7qHAN3L5s1Yd%2BP4m65JQ%2B0icnQnh8SEE3K0cx2JDlQ9Uf3EVciwujswpYtVu8JbRQAE40rYcHlhkHbA9r5FGw13NbL7Y9Z9dpuBcnLDSgbxo%2BQQSKE9dkLe8C70T9oF1A5ejREpZ2dG4xUZTUouiNFB%2BW586C23hWMsn0kP1z5esWxFFuGIzxY8HwVvpj%2Fe8IBU%2BkO1cabXjNAygx8iKLU1nLQcor1QmZYw1F7CgqLrm22oQT%2FGf%2FGGvp5AL2SGAryQxHeyf2Hz3%2BOABGJYUd5DE7FO4PyyGBHJw1kLcZUflNOCbUIVyGF2dWi6sATZJcxz06i%2FUb6z06z2fEWeNQiNMOHOFxCQ%2FvSvXjD93pWbBjqzAsnHm1Wz7Gk5L6FR4v%2FyrgVDiKaow6b7G5hmLd4xba%2FBuiwfuntO1oSew99oUy0i0aigsjFoAN%2FEvIcXzvFiP%2BqHnBsVjiOB%2BkAP%2F1oxd3fTJvZ6ezzp3DHwb%2BHnod%2FwA93A0Bh6%2FT4WBDFNkBNRxfFb%2F9M4xba477vg9mg%2Fb0%2F7excYq7RY8AcZ%2F%2FAxBeMHZjhwpRyKZC4FBB3JYTjoGEbDSF7NhijX%2FZKmNQcN0HQ7n6%2FOs1%2F0%2F3nRUCtUTJUgL%2FQypxsDL55%2FW9jM5Wv0f7nyopaGDQZ2mKMZwyeUSpL%2B%2BVP3wsapWU6jQnfvWvCa%2F4qEd5flSt11yj8O90iuGweKz22f6AfJhOMF6aFh1Pu97e85AbzVsPVO%2FdzaXqz2mnY9gxb4CPezeHGScqaS3kkWE30%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221104T200413Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAYXDBVZ7BOCJHYKXV%2F20221104%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=cbabc981e72dab93354b87021a8d84cd49df9bd4cb43980ea789ef666262f22f")
    df = df.set_index(["title"])

    document_embeddings = load_embeddings("https://askbook.s3.us-east-1.amazonaws.com/embeddings.csv?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJHMEUCIQCosaeIJrUiCuzmLKO2PgXleWPusM85rA0iDsUqZmUNWQIgPwJAqgNu%2BtJyEfD0m0zvIWpgBWyrwtNH449DpPbHJDEq7QII3f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgw1OTkzNTI3OTA5NzgiDMyp840t9%2FPinfvxQSrBAh0gXgLija9B8WABaoVjfJlR%2B11eQiks2cxjW0YVeJ4KJYqaS8gkO0tZ%2FcoSnsoQ%2FdmlLuv6z7qHAN3L5s1Yd%2BP4m65JQ%2B0icnQnh8SEE3K0cx2JDlQ9Uf3EVciwujswpYtVu8JbRQAE40rYcHlhkHbA9r5FGw13NbL7Y9Z9dpuBcnLDSgbxo%2BQQSKE9dkLe8C70T9oF1A5ejREpZ2dG4xUZTUouiNFB%2BW586C23hWMsn0kP1z5esWxFFuGIzxY8HwVvpj%2Fe8IBU%2BkO1cabXjNAygx8iKLU1nLQcor1QmZYw1F7CgqLrm22oQT%2FGf%2FGGvp5AL2SGAryQxHeyf2Hz3%2BOABGJYUd5DE7FO4PyyGBHJw1kLcZUflNOCbUIVyGF2dWi6sATZJcxz06i%2FUb6z06z2fEWeNQiNMOHOFxCQ%2FvSvXjD93pWbBjqzAsnHm1Wz7Gk5L6FR4v%2FyrgVDiKaow6b7G5hmLd4xba%2FBuiwfuntO1oSew99oUy0i0aigsjFoAN%2FEvIcXzvFiP%2BqHnBsVjiOB%2BkAP%2F1oxd3fTJvZ6ezzp3DHwb%2BHnod%2FwA93A0Bh6%2FT4WBDFNkBNRxfFb%2F9M4xba477vg9mg%2Fb0%2F7excYq7RY8AcZ%2F%2FAxBeMHZjhwpRyKZC4FBB3JYTjoGEbDSF7NhijX%2FZKmNQcN0HQ7n6%2FOs1%2F0%2F3nRUCtUTJUgL%2FQypxsDL55%2FW9jM5Wv0f7nyopaGDQZ2mKMZwyeUSpL%2B%2BVP3wsapWU6jQnfvWvCa%2F4qEd5flSt11yj8O90iuGweKz22f6AfJhOMF6aFh1Pu97e85AbzVsPVO%2FdzaXqz2mnY9gxb4CPezeHGScqaS3kkWE30%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221104T200509Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43199&X-Amz-Credential=ASIAYXDBVZ7BOCJHYKXV%2F20221104%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=ee12dc9fa9e57c1e740bdd243e5b4e4376425e60ae064b4ed7b57d28df2dac24")

    answer = answer_query_with_context(question, df, document_embeddings)
    print(answer)

    question = Question(question=question, answer=answer)
    question.save()

    project_uuid = '6314e4df'
    voice_uuid = '0eb3a3f1'

    response = Resemble.v2.clips.create_sync(
        project_uuid,
        voice_uuid,
        answer,
        title=None,
        sample_rate=None,
        output_format=None,
        precision=None,
        include_timestamps=None,
        is_public=None,
        is_archived=None,
        raw=None
    )

    question.audio_src_url = response['item']['audio_src']
    question.save()

    return render(request, "answer.html", { "answer": answer, "question": question.question, "audio_src_url": question.audio_src_url })

def db(request):
    questions = Question.objects.all().order_by('-ask_count')

    return render(request, "db.html", { "questions": questions })
