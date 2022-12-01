## Setup

1. Fill in `.env` with your proper env variables for OpenAI, AWS, and Resemble.

2. Turn your PDF into embeddings for GPT-3:

```
python scripts/pdf_to_pages_embeddings.py --pdf book.pdf
```

3. Upload these files to S3 and reference these within `views.py`.

4. Set up database tables:

```
python manage.py makemigrations
python manage.py migrate
```

### Run locally

```
python manage.py runserver
```

## Deployment

This repo automatically deploys the `main` branch to Heroku.
