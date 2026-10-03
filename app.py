"""AskQL - chat with your CSV/XLSX data in plain English.

Upload files -> they become SQLite tables -> ask a question -> Gemini writes a
SELECT query -> it runs read-only -> you see the SQL, the results and an explanation.
"""
import io
import os
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "askql.db"
FEEDBACK_DB = DATA_DIR / "feedback.db"

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_ROWS = 200
QUERY_TIMEOUT_S = 5

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB uploads

DIALECTS = {
    "SQLite": "Use SQLite syntax.",
    "Standard": "Use standard ANSI SQL.",
    "Trino": (
        "Use Trino syntax: double-quote identifiers, array_agg() instead of GROUP_CONCAT, "
        "CAST() instead of ::, UNNEST for arrays, explicit ANSI JOINs, approx_distinct() for big data."
    ),
    "Spark": (
        "Use Spark SQL syntax: backtick identifiers, collect_list() instead of GROUP_CONCAT, "
        "EXPLODE for arrays, concat_ws() for joining, approx_count_distinct() for big data."
    ),
}


# ---------------------------------------------------------------- database ---
def connect_rw():
    return sqlite3.connect(DB_PATH)


def connect_ro():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    deadline = time.time() + QUERY_TIMEOUT_S
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 100_000)
    return conn


def q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def get_schema():
    """[{name, columns:[{name,type}], sample:[[...]]}] for every user table."""
    if not DB_PATH.exists():
        return []
    conn = connect_ro()
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        schema = []
        for t in tables:
            cols = [{"name": c[1], "type": c[2] or "TEXT"} for c in conn.execute(f"PRAGMA table_info({q(t)})")]
            sample = [list(r) for r in conn.execute(f"SELECT * FROM {q(t)} LIMIT 3")]
            schema.append({"name": t, "columns": cols, "sample": sample})
        return schema
    finally:
        conn.close()


def schema_prompt(schema) -> str:
    lines = []
    for t in schema:
        cols = ", ".join(f'{q(c["name"])} {c["type"]}' for c in t["columns"])
        lines.append(f"TABLE {q(t['name'])} ({cols})")
        for row in t["sample"]:
            lines.append("  sample row: " + str([str(v)[:40] for v in row]))
    return "\n".join(lines)


def clean_name(name: str) -> str:
    name = re.sub(r"\W+", "_", name).strip("_")
    return name if name and not name[0].isdigit() else f"t_{name}"


# --------------------------------------------------------------- SQL safety ---
def check_select(sql: str) -> str:
    """Return a cleaned single SELECT/WITH statement or raise ValueError."""
    sql = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S).strip().rstrip(";").strip()
    if not sql:
        raise ValueError("Empty query.")
    if ";" in sql:
        raise ValueError("Only one statement at a time is allowed.")
    if not re.match(r"(?is)^(select|with)\b", sql):
        raise ValueError("Only read-only SELECT queries can be run.")
    if re.search(r"(?i)\b(attach|detach|pragma|load_extension)\b", sql):
        raise ValueError("That statement is not allowed.")
    return sql


def run_select(sql: str):
    sql = check_select(sql)
    if not DB_PATH.exists():
        raise ValueError("No data yet - upload a CSV or XLSX first.")
    conn = connect_ro()
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.OperationalError as e:
        raise ValueError("Query timed out." if "interrupted" in str(e) else str(e))
    finally:
        conn.close()
    truncated = len(rows) > MAX_ROWS
    return cols, [list(r) for r in rows[:MAX_ROWS]], truncated


# ----------------------------------------------------------------- feedback ---
def feedback_conn():
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL, sql TEXT NOT NULL,
        is_correct INTEGER NOT NULL, ts DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(question, sql))""")
    return conn


def words(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def approved_examples(question: str, limit: int = 3):
    """Past thumbs-up answers whose question looks similar (word overlap)."""
    conn = feedback_conn()
    rows = conn.execute("SELECT question, sql FROM feedback WHERE is_correct = 1").fetchall()
    conn.close()
    qw, scored = words(question), []
    for rq, rs in rows:
        rw = words(rq)
        score = len(qw & rw) / max(len(qw | rw), 1)
        if score >= 0.3:
            scored.append((score, rq, rs))
    return [(rq, rs) for _, rq, rs in sorted(scored, reverse=True)[:limit]]


# --------------------------------------------------------------------- LLM ---
_client = None


def call_llm(prompt: str) -> str:
    global _client
    if _client is None:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")
        from google import genai
        _client = genai.Client(api_key=key)
    resp = _client.models.generate_content(model=MODEL, contents=prompt)
    return resp.text or ""


def build_prompt(message, history, dialect, schema, avoid_sql=None, error=None):
    parts = [
        "You are AskQL, an expert SQL assistant.",
        DIALECTS.get(dialect, DIALECTS["Standard"]),
    ]
    if schema:
        parts.append("The user's database (SQLite):\n" + schema_prompt(schema))
        parts.append("Use ONLY the tables and columns listed above. Quote identifiers with double quotes.")
    else:
        parts.append("No data is uploaded yet. Answer as general SQL help using sensible example table names.")
    parts.append("Write exactly ONE read-only SELECT (or WITH ... SELECT) query. Never modify data. "
                 "Add LIMIT 100 unless the query aggregates.")
    ex = approved_examples(message)
    if ex:
        parts.append("Previously approved answers to similar questions (may not match the current schema):\n" +
                     "\n".join(f"Q: {a}\nSQL: {b}" for a, b in ex))
    if history:
        parts.append("Conversation so far:\n" + "\n".join(
            f"{'User' if h.get('role') == 'user' else 'Assistant'}: {str(h.get('content', ''))[:600]}"
            for h in history[-6:]))
    parts.append(f"Question: {message}")
    if avoid_sql:
        parts.append(f"The user rejected this query, so write a meaningfully different, correct one:\n{avoid_sql}")
    if error:
        parts.append(f"Your previous query failed with: {error}\nFix it.")
    parts.append("Reply in exactly this format:\n```sql\n<query>\n```\nExplanation: <1-3 plain sentences>\n"
                 "If the message is not a data or SQL question (e.g. a greeting), reply briefly in plain text with no code block.")
    return "\n\n".join(parts)


def parse_reply(text: str):
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    sql = m.group(1).strip() if m else None
    expl = re.sub(r"```.*?```", "", text, flags=re.S).strip()
    expl = re.sub(r"(?i)^explanation:\s*", "", expl).strip()
    return sql, expl


# ------------------------------------------------------------------- routes ---
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/schema")
def api_schema():
    return jsonify(schema=[{"name": t["name"], "columns": t["columns"]} for t in get_schema()])


@app.post("/api/upload")
def api_upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="No files received."), 400
    created, problems = [], []
    conn = connect_rw()
    try:
        for f in files:
            stem, ext = os.path.splitext(f.filename or "")
            ext = ext.lower()
            if ext not in (".csv", ".xlsx"):
                problems.append(f"{f.filename}: only .csv and .xlsx are supported")
                continue
            try:
                raw = f.read()
                if ext == ".csv":
                    try:
                        frames = {stem: pd.read_csv(io.BytesIO(raw))}
                    except UnicodeDecodeError:
                        frames = {stem: pd.read_csv(io.BytesIO(raw), encoding="latin-1")}
                else:
                    sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
                    frames = ({stem: next(iter(sheets.values()))} if len(sheets) == 1
                              else {f"{stem}_{s}": d for s, d in sheets.items()})
                for name, df in frames.items():
                    df.columns = [str(c).strip() or f"col_{i}" for i, c in enumerate(df.columns, 1)]
                    table = clean_name(name)
                    df.to_sql(table, conn, if_exists="replace", index=False)
                    created.append({"table": table, "rows": len(df)})
            except Exception as e:  # bad file shouldn't kill the whole upload
                problems.append(f"{f.filename}: {e}")
        conn.commit()
    finally:
        conn.close()
    return jsonify(created=created, problems=problems)


@app.post("/api/chat")
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify(error="Message is empty."), 400
    dialect = data.get("dialect", "SQLite")
    history = data.get("history") or []
    schema = get_schema()
    can_run = dialect == "SQLite" and bool(schema)

    try:
        reply = call_llm(build_prompt(message, history, dialect, schema, avoid_sql=data.get("avoid_sql")))
        sql, expl = parse_reply(reply)
        out = {"sql": sql, "explanation": expl, "columns": None, "rows": None,
               "truncated": False, "run_error": None, "ran": False}
        if sql and can_run:
            try:
                out["columns"], out["rows"], out["truncated"] = run_select(sql)
                out["ran"] = True
            except ValueError as e:  # one automatic repair attempt
                reply = call_llm(build_prompt(message, history, dialect, schema, error=str(e)))
                sql2, expl2 = parse_reply(reply)
                if sql2:
                    out["sql"], out["explanation"] = sql2, expl2 or expl
                    try:
                        out["columns"], out["rows"], out["truncated"] = run_select(sql2)
                        out["ran"] = True
                    except ValueError as e2:
                        out["run_error"] = str(e2)
                else:
                    out["run_error"] = str(e)
        return jsonify(out)
    except RuntimeError as e:
        return jsonify(error=str(e)), 500
    except Exception as e:
        return jsonify(error=f"Model call failed: {e}"), 502


@app.post("/api/execute")
def api_execute():
    sql = (request.get_json(force=True, silent=True) or {}).get("sql", "")
    try:
        cols, rows, truncated = run_select(sql)
        return jsonify(columns=cols, rows=rows, truncated=truncated)
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.post("/api/feedback")
def api_feedback():
    d = request.get_json(force=True, silent=True) or {}
    if not d.get("question") or not d.get("sql"):
        return jsonify(error="question and sql are required"), 400
    conn = feedback_conn()
    conn.execute("INSERT OR REPLACE INTO feedback (question, sql, is_correct) VALUES (?,?,?)",
                 (d["question"], d["sql"], 1 if d.get("is_correct") else 0))
    conn.commit()
    conn.close()
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=False)
