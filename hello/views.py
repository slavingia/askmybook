from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import redirect

from .models import Question

import pandas as pd
import openai
import numpy as np

from resemble import Resemble

import boto3

Resemble.api_key('0vWhLtB2fmjVIE0Nuzic5wtt')
openai.api_key = "sk-DOiDZHHE1f1tvxnO5zs103vHelanA6BVBVO44cN7"
DEEPGRAM_API_KEY = "b4a5ce14299719a271113fe23bf9316a41ea8119"

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

    header = """Sahil Lavingia is the founder and CEO of Gumroad, and the author of the book The Minimalist Entrepreneur (also known as TME). These are questions and answers by him. Please keep your answers to three sentences maximum, and speak in complete sentences. Stop speaking once your point is made.\n\nContext that may be useful, pulled from The Minimalist Entrepreneur:\n"""

    question_1 = "\n\n\nQ: How to choose what business to start?\n\nA: First off don't be in a rush. Look around you, see what problems you or other people are facing, and solve one of these problems if you see some overlap with your passions or skills. Or, even if you don't see an overlap, imagine how you would solve that problem anyway. Start super, super small."
    question_2 = "\n\n\nQ: Q: Should we start the business on the side first or should we put full effort right from the start?\n\nA:   Always on the side. Things start small and get bigger from there, and I don't know if I would ever “fully” commit to something unless I had some semblance of customer traction. Like with this product I'm working on now!"
    question_3 = "\n\n\nQ: Should we sell first than build or the other way around?\n\nA: I would recommend building first. Building will teach you a lot, and too many people use “sales” as an excuse to never learn essential skills like building. You can't sell a house you can't build!"
    question_4 = "\n\n\nQ: Andrew Chen has a book on this so maybe touché, but how should founders think about the cold start problem? Businesses are hard to start, and even harder to sustain but the latter is somewhat defined and structured, whereas the former is the vast unknown. Not sure if it's worthy, but this is something I have personally struggled with\n\nA: Hey, this is about my book, not his! I would solve the problem from a single player perspective first. For example, Gumroad is useful to a creator looking to sell something even if no one is currently using the platform. Usage helps, but it's not necessary."
    question_5 = "\n\n\nQ: What is one business that you think is ripe for a minimalist Entrepreneur innovation that isn't currently being pursued by your community?\n\nA: I would move to a place outside of a big city and watch how broken, slow, and non-automated most things are. And of course the big categories like housing, transportation, toys, healthcare, supply chain, food, and more, are constantly being upturned. Go to an industry conference and it's all they talk about! Any industry…"
    question_6 = "\n\n\nQ: How can you tell if your pricing is right? If you are leaving money on the table\n\nA: I would work backwards from the kind of success you want, how many customers you think you can reasonably get to within a few years, and then reverse engineer how much it should be priced to make that work."
    question_7 = "\n\n\nQ: Why is the name of your book 'the minimalist entrepreneur' \n\nA: I think more people should start businesses, and was hoping that making it feel more “minimal” would make it feel more achievable and lead more people to starting-the hardest step."
    question_8 = "\n\n\nQ: How long it takes to write TME\n\nA: About 500 hours over the course of a year or two, including book proposal and outline."
    question_9 = "\n\n\nQ: What is the best way to distribute surveys to test my product idea\n\nA: I use Google Forms and my email list / Twitter account. Works great and is 100% free."
    question_10 = "\n\n\nQ: How do you know, when to quit\n\nA: When I'm bored, no longer learning, not earning enough, getting physically unhealthy, etc… loads of reasons. I think the default should be to “quit” and work on something new. Few things are worth holding your attention for a long period of time."

    return header + "".join(chosen_sections) + question_1 + question_2 + question_3 + question_4 + question_5 + question_6 + question_7 + question_8 + question_9 + question_10 + "\n\n\nQ: " + question + "\n\nA: "

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

@csrf_exempt
def ask(request):
    question_asked = request.POST.get("question", "")

    if not question_asked.endswith('?'):
        question_asked += '?'

    previous_question = Question.objects.filter(question=question_asked).first()
    audio_src_url = previous_question and previous_question.audio_src_url if previous_question else None

    if audio_src_url:
        print("previously asked and answered: " + previous_question.answer + " ( " + previous_question.audio_src_url + ")")
        previous_question.ask_count = previous_question.ask_count + 1
        previous_question.save()
        return JsonResponse({ "question": previous_question.question, "audio_src_url": audio_src_url })

    s3 = boto3.client(
        's3',
        aws_access_key_id="AKIAYXDBVZ7BCW3GF67W",
        aws_secret_access_key="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"
    )

    s3.download_file('askbook', 'pages.csv', 'pages.csv')
    s3.download_file('askbook', 'embeddings.csv', 'embeddings.csv')

    df = pd.read_csv('pages.csv')
    df = df.set_index(["title"])

    document_embeddings = load_embeddings('embeddings.csv')

    answer = answer_query_with_context(question_asked, df, document_embeddings)
    print(answer)

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

    question = Question(question=question_asked, answer=answer, audio_src_url=response['item']['audio_src'])
    question.save()

    return JsonResponse({ "question": question.question, "audio_src_url": question.audio_src_url })

def db(request):
    questions = Question.objects.all().order_by('-ask_count')

    return render(request, "db.html", { "questions": questions })

def real_answer(request):
    id = request.POST.get("id", "")
    answer = request.POST.get("answer", "")

    question = Question.objects.get(pk=id)
    question.real_answer = answer

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

    return JsonResponse({ "question": question.question, "answer": question.real_answer })

def delete_question(request):
    id = request.POST.get("id", "")
    Question.objects.get(pk=id).delete()

    return redirect("/db/")

def question(request, id):
    question = Question.objects.get(pk=id)
    return render(request, "answer.html", { "question": question.question, "audio_src_url": question.audio_src_url })
