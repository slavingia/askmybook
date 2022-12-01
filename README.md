# askbook

### Website / API

- Fill in `.env` with env vars
- `python manage.py makemigrations` and `python manage.py migrate` to setup db tables
- `python manage.py runserver` to run locally

Deploys the `main` branch automatically to Heroku.

### Scripts

Add book PDF to `static` folder.

- `python pdf_to_pages_content.py` to turn local PDF into structured page content
- `python content_to_embeddings.py` to turn structured page content into embeddings
- `python ask_questions_to_model.py` to ask GPT-3 questions using embeddings

Upload these files to S3 and update the URLs.
