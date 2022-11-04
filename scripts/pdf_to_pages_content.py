import pandas as pd
from typing import Set
from transformers import GPT2TokenizerFast

import numpy as np

from PyPDF2 import PdfReader

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def extract_pages(
    page_text: str,
    index: int,
) -> str:
    """
    Extract the text from the page
    """
    if len(page_text) == 0:
        return []

    content = " ".join(page_text.split())
    print("page text: " + content)
    outputs = [("Page " + str(index), content, count_tokens(content)+4)]

    return outputs

reader = PdfReader("book.pdf")

res = []
i = 1
for page in reader.pages:
    res += extract_pages(page.extract_text(), i)
    i += 1
df = pd.DataFrame(res, columns=["title", "content", "tokens"])
df = df.reset_index().drop('index',axis=1) # reset index
df.head()

df.to_csv('pages.csv', index=False)
