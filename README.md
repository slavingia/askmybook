# askbook

### Website / API

- Fill in `.env` with env vars
- `python manage.py makemigrations` and `python manage.py migrate` to setup db tables
- `python manage.py runserver` to run locally

Deploys the `main` branch automatically to Heroku.

### Scripts

- `python pdf_to_pages_content.py` to turn local PDF into structured page content
- `python content_to_embeddings.py` to turn structured page content into embeddings
- `python ask_questions_to_model.py` to ask GPT-3 questions using embeddings

Upload these files to S3 and update the URLs.

### Fine-tuning model

- Go to https://askmybook.com/queue and answer questions
- Go to https://askmymybook.com/metadata.jsonl to download metadata (answers, or real answers where provided)
  - Provide ?real_only=1 if you only want real answers to fine-tune the model
- `openai tools fine_tunes.prepare_data -f metadata.jsonl` to prepare metadata
- `openai api fine_tunes.create -t "metadata_prepared.jsonl` to fine-tune on prepared metadata