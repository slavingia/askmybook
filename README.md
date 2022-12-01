## Setup

1. Fill in `.env` with your proper env vars.

2. Turn your PDF into embeddings for GPT-3:

```
python pdf_to_pages_embeddings.py --pdf book.pdf
```

- `python manage.py makemigrations` and `python manage.py migrate` to setup DB tables
- `python manage.py runserver` to run locally

Deploys the `main` branch automatically to Heroku.



Upload these files to S3 and update the URLs.