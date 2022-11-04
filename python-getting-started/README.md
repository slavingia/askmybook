# askbook

### Getting started

- Add `OPENAI_API_KEY=...` to `.env`

### Website / API

- `heroku local` from `python-getting-started` folder to run locally

This application supports the [Getting Started with Python on Heroku](https://devcenter.heroku.com/articles/getting-started-with-python) article - check it out for instructions on how to deploy this app to Heroku and also run it locally.

### Scripts

- `python pdf_to_pages_content.py` to turn local PDF into structured page content
- `python content_to_embeddings.py` to turn structured page content into embeddings
- `python ask_questions.py` to ask GPT-3 questions using embeddings
