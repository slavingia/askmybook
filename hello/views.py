from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required

from .models import Question

import pandas as pd
import openai
import numpy as np


import os

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

COMPLETIONS_MODEL = "text-davinci-003"

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
separator_len = 3

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 250,
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
    Find the query embedding for the supplied query, and compare it against all the
    pre-calculated document embeddings to find the most relevant sections.

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


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> tuple[str, str]:
    """Fetch relevant embeddings"""
    most_relevant_document_sections = order_document_sections_by_query_similarity(
        question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[df['title'] == section_index].iloc[0]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            space_left = MAX_SECTION_LEN - chosen_sections_len - len(SEPARATOR)
            chosen_sections.append(SEPARATOR + document_section.content[:space_left])
            chosen_sections_indexes.append(str(section_index))
            break

        chosen_sections.append(SEPARATOR + document_section.content)
        chosen_sections_indexes.append(str(section_index))

    header = """Max Pumperla is a Software Engineer at Anyscale and author of Learning Ray. Please keep your answers to four sentences maximum, and speak in complete sentences. Stop speaking once your point is made.\n\nContext that may be useful, pulled from the book Learning Ray:\n"""

    # TODO add better questions for context
    # question_1 = "\n\n\nQ: How to choose what business to start?\n\nA: First off don't be in a rush. Look around you, see what problems you or other people are facing, and solve one of these problems if you see some overlap with your passions or skills. Or, even if you don't see an overlap, imagine how you would solve that problem anyway. Start super, super small."
    # question_2 = "\n\n\nQ: Q: Should we start the business on the side first or should we put full effort right from the start?\n\nA:   Always on the side. Things start small and get bigger from there, and I don't know if I would ever “fully” commit to something unless I had some semblance of customer traction. Like with this product I'm working on now!"
    # question_3 = "\n\n\nQ: Should we sell first than build or the other way around?\n\nA: I would recommend building first. Building will teach you a lot, and too many people use “sales” as an excuse to never learn essential skills like building. You can't sell a house you can't build!"

    return (header + "".join(chosen_sections)
            # + question_1 + question_2 + question_3
            + "\n\n\nQ: " + question + "\n\nA: "), ("".join(chosen_sections))

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
) -> tuple[str, str]:
    prompt, context = construct_prompt(
        query,
        document_embeddings,
        df
    )

    print("===\n", prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n"), context


def index(request):
    return render(request, "index.html", {"default_question": "What is Ray?"})


@csrf_exempt
def ask(request):
    question_asked = request.POST.get("question", "")

    if not question_asked.endswith('?'):
        question_asked += '?'

    df = pd.read_csv('book.pdf.pages.csv')
    document_embeddings = load_embeddings('book.pdf.embeddings.csv')
    answer, context = answer_query_with_context(question_asked, df, document_embeddings)

    question = Question(question=question_asked, answer=answer,
                        context=context, audio_src_url=None)
    question.save()

    return JsonResponse({"question": question.question, "answer": answer,
                         "audio_src_url": question.audio_src_url, "id": question.pk})


@login_required
def db(request):
    questions = Question.objects.all().order_by('-ask_count')

    return render(request, "db.html", { "questions": questions })


def question(request, id):
    question = Question.objects.get(pk=id)
    return render(request, "index.html", {"default_question": question.question,
                                          "answer": question.answer,
                                          "audio_src_url": question.audio_src_url })
