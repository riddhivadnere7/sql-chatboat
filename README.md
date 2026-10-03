# AskQL — chat with your data in plain English

Upload a CSV or XLSX, ask a question, get the SQL, the results and a short explanation.

## Run it

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then put your GOOGLE_API_KEY in .env
python app.py
```

Open http://127.0.0.1:5000, click **Upload CSV / XLSX** (try `sample_data/titanic.csv`) and ask
e.g. *"How many passengers survived in each class?"*

## How it works

- Uploads become tables in `data/askql.db` (SQLite). Each XLSX sheet becomes its own table.
- The model sees your real schema plus 3 sample rows per table, so it uses real column names.
- Only a single `SELECT`/`WITH` runs, on a **read-only** connection with a 5 s timeout and a 200-row cap.
  If the query errors, the error is sent back to the model for one automatic fix.
- The SQL box is editable: change it and press **Run**.
- 👍 saves a question/SQL pair; similar future questions get it as an example. 👎 asks for a different query.
- Trino / Spark / Standard dialects only generate SQL (they can't run on SQLite).

## Files

`app.py` backend · `static/index.html` UI · `data/` databases (git-ignored) · `.env` your key and optional `GEMINI_MODEL`.
