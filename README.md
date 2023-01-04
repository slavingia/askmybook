## Setup

1. Create and fill in `.env` using `.env.example` as an example.

2. Install required Python packages

```
pip install -r requirements.txt
```

Mac M1 / OS X Note: if you get an error installing psycopg2, you may need:

```
brew install postgresql
```

See https://github.com/psycopg/psycopg2/issues/1200


2. Turn your PDF into embeddings for GPT-3:

```
python scripts/pdf_to_pages_embeddings.py --pdf book.pdf
```

3. Set up database tables:

```
python manage.py makemigrations
python manage.py migrate
```

4. Other things to update:

- Book title
- Book cover image
- URL to purchase book
- Author name and bio

## Deploy to Heroku

1. Create a Heroku app:

```
heroku create askmybook
```

Set config variables on Heroku to match `.env`.

2. Push to Heroku:

```
git push heroku main
heroku ps:scale web=1
heroku run python manage.py migrate
heroku open
heroku domains:add askmybook.com
```

Note that this repo does not contain the `pages.csv` and `embeddings.csv` you'll need, generated above. You can remove `.csv` from your own `.gitignore` and push them manually via `git push heroku main`.

### Run locally

```
heroku local
```
