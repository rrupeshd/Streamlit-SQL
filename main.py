# app.py
# Streamlit SQL Playground (OneCompiler-style)
# -------------------------------------------------------------
# Features
# - In-memory DuckDB engine (fast, zero-setup, very capable SQL)
# - Optional SQL dialect transpilation (SQL Server T‑SQL, MySQL, Postgres, SQLite → DuckDB) via sqlglot
# - Multi-statement execution (split safely with sqlparse)
# - Default sample tables you can instantly query
# - Upload CSV/Parquet to create tables on the fly
# - Schema browser & quick previews
# - Column tools in sidebar: rename columns, change column types
# - Optional sanitize uploaded column names to snake_case
# - Per-session isolation; "Reset" to start fresh
# -------------------------------------------------------------

import time
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
import io
import re

# Lazy-import optional deps so the app can start and show helpful tips
_missing = []
try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None
    _missing.append("duckdb")

try:
    import sqlparse  # type: ignore
except Exception:
    sqlparse = None
    _missing.append("sqlparse")

try:
    import sqlglot  # type: ignore
    from sqlglot import transpile
except Exception:
    sqlglot = None
    transpile = None
    _missing.append("sqlglot")

# Optional: on-the-fly encoding detection for CSV uploads
try:
    from charset_normalizer import from_bytes as _cn_from_bytes  # lightweight alternative to chardet
except Exception:
    _cn_from_bytes = None

st.set_page_config(page_title="SQL Playground", layout="wide")
st.title("⚙️ SQL Playground")
st.caption("Run SQL instantly on an in‑memory database. Create your own tables or start with sample data.")

# -------------------------------------------------------------
# Helpful install message
# -------------------------------------------------------------
if _missing:
    st.error(
        """
        The following Python packages are required but not installed: **{pkgs}**.
        
        In your environment or terminal, run:
        
        ```bash
        pip install streamlit duckdb sqlparse sqlglot pandas
        ```
        
        Then start the app with:
        
        ```bash
        streamlit run app.py
        ```
        """.format(pkgs=", ".join(_missing))
    )
    st.stop()

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
DIALECTS = [
    "auto",
    "duckdb",
    "tsql",       # SQL Server / T‑SQL
    "postgres",
    "mysql",
    "sqlite",
    "snowflake",
    "bigquery",
    "hive",
    "spark",
    "presto",
    "trino",
    "oracle",
]

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def safe_ident(name: str) -> str:
    """Quote an identifier for DuckDB safely."""
    return '"' + str(name).replace('"', '""') + '"'

def snake_case(name: str) -> str:
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s and s[0].isdigit():
        s = "_" + s
    return s.lower()

def sanitize_columns(cols) -> list:
    seen = {}
    out = []
    for c in cols:
        base = snake_case(str(c))
        name = base or "col"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out

def split_statements(sql_text: str) -> List[str]:
    """Split SQL into individual statements using sqlparse, preserving order."""
    if not sql_text.strip():
        return []
    if sqlparse is None:
        # Fallback: naive split — less robust with semicolons in strings
        return [s.strip() for s in sql_text.split(";") if s.strip()]
    parts = [s.strip() for s in sqlparse.split(sql_text) if s and s.strip()]
    return parts


def to_duckdb(stmt: str, chosen_dialect: str) -> str:
    """Transpile a single statement to DuckDB SQL using sqlglot if needed."""
    if not stmt.strip():
        return stmt
    if chosen_dialect in ("duckdb", "auto") and sqlglot is None:
        return stmt  # sqlglot unavailable; run as-is (likely DuckDB-compatible)

    if chosen_dialect == "duckdb":
        return stmt

    if sqlglot is None or transpile is None:
        return stmt

    # Try requested dialect first; if auto, try several common ones.
    candidates = (
        [chosen_dialect]
        if chosen_dialect != "auto"
        else [
            "tsql",
            "postgres",
            "mysql",
            "sqlite",
            "duckdb",
            "snowflake",
            "bigquery",
            "presto",
            "trino",
            "oracle",
            "hive",
            "spark",
        ]
    )
    for d in candidates:
        try:
            return transpile(stmt, read=d, write="duckdb")[0]
        except Exception:
            continue
    # Last resort: run original; DuckDB handles a lot of ANSI SQL directly
    return stmt


def run_sql(sql_text: str, read_dialect: str) -> List[Dict]:
    """Execute multi-statement SQL. Returns a list of result dicts.
    Each dict contains: { 'statement', 'ok', 'elapsed_ms', 'rows'(optional DataFrame), 'error'(optional) }
    """
    results: List[Dict] = []
    for raw_stmt in split_statements(sql_text):
        stmt = to_duckdb(raw_stmt, read_dialect)
        start = time.perf_counter()
        try:
            cur = con.execute(stmt)
            df: Optional[pd.DataFrame] = None
            # Try fetching rows (SELECT/SHOW/EXPLAIN, etc.)
            try:
                df = cur.fetch_df()
            except Exception:
                df = None
            elapsed_ms = (time.perf_counter() - start) * 1000
            out = {"statement": raw_stmt, "duckdb": stmt, "ok": True, "elapsed_ms": elapsed_ms}
            if df is not None:
                out["rows"] = df
                st.session_state.last_result = df
            results.append(out)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            results.append({
                "statement": raw_stmt,
                "duckdb": stmt,
                "ok": False,
                "elapsed_ms": elapsed_ms,
                "error": str(e),
            })
    return results


def list_tables() -> pd.DataFrame:
    q = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type = 'BASE TABLE'
      AND table_schema IN ('main','temp')
    ORDER BY 1,2;
    """
    return con.execute(q).fetch_df()


def table_columns(schema: str, table: str) -> pd.DataFrame:
    q = """
    SELECT column_name, data_type, ordinal_position
    FROM information_schema.columns
    WHERE table_schema = ? AND table_name = ?
    ORDER BY ordinal_position;
    """
    return con.execute(q, [schema, table]).fetch_df()


def preview_table(schema: str, table: str, n: int = 25) -> pd.DataFrame:
    q = f"SELECT * FROM {safe_ident(schema)}.{safe_ident(table)} LIMIT {n};"
    return con.execute(q).fetch_df()


# -------------------------------------------------------------
# Column tools: type/rename ops
# -------------------------------------------------------------
def build_cast_expr(column: str, target_type: str, *, strip_non_numeric: bool = True, comma_decimal: bool = False) -> str:
    col = safe_ident(column)
    tt = target_type.upper()
    if tt in {"INTEGER", "BIGINT", "DOUBLE"} or tt.startswith("DECIMAL"):
        expr = col
        if strip_non_numeric:
            expr = f"REGEXP_REPLACE({expr}, '[^0-9,.-]', '')"
        if comma_decimal and (tt in {"DOUBLE", "INTEGER", "BIGINT"} or tt.startswith("DECIMAL")):
            expr = f"REPLACE({expr}, ',', '.')"
        return f"TRY_CAST({expr} AS {tt})"
    elif tt in {"VARCHAR", "DATE", "TIMESTAMP", "BOOLEAN"}:
        return f"TRY_CAST({col} AS {tt})"
    else:
        return f"TRY_CAST({col} AS {tt})"


def rename_column(schema: str, table: str, old: str, new: str):
    con.execute(f"ALTER TABLE {safe_ident(schema)}.{safe_ident(table)} RENAME COLUMN {safe_ident(old)} TO {safe_ident(new)};")


def alter_column_type(schema: str, table: str, column: str, target_type: str, *, strip_non_numeric: bool = True, comma_decimal: bool = False):
    tmp = f"__tmp__{snake_case(column)}"
    expr = build_cast_expr(column, target_type, strip_non_numeric=strip_non_numeric, comma_decimal=comma_decimal)
    con.execute(f"ALTER TABLE {safe_ident(schema)}.{safe_ident(table)} ADD COLUMN {safe_ident(tmp)} {target_type};")
    con.execute(f"UPDATE {safe_ident(schema)}.{safe_ident(table)} SET {safe_ident(tmp)} = {expr};")
    con.execute(f"ALTER TABLE {safe_ident(schema)}.{safe_ident(table)} DROP COLUMN {safe_ident(column)};")
    con.execute(f"ALTER TABLE {safe_ident(schema)}.{safe_ident(table)} RENAME COLUMN {safe_ident(tmp)} TO {safe_ident(column)};")


# -------------------------------------------------------------
# Session-scoped DB connection & state
# -------------------------------------------------------------
if "con" not in st.session_state:
    st.session_state.con = duckdb.connect(database=":memory:")
    st.session_state.query_history: List[Dict] = []
    st.session_state.last_result: Optional[pd.DataFrame] = None
    st.session_state.loaded_samples = False
    st.session_state.show_history = False

con: "duckdb.DuckDBPyConnection" = st.session_state.con


# -------------------------------------------------------------
# Sample data
# -------------------------------------------------------------
def load_sample_data():
    if st.session_state.loaded_samples:
        return

    # Employees & Departments
    depts = pd.DataFrame(
        [
            {"dept_id": 10, "dept_name": "Engineering"},
            {"dept_id": 20, "dept_name": "Sales"},
            {"dept_id": 30, "dept_name": "Marketing"},
            {"dept_id": 40, "dept_name": "Finance"},
        ]
    )

    employees = []
    rng = pd.Series(range(1, 51))
    first_names = [
        "Alex","Jordan","Taylor","Sam","Casey","Riley","Avery","Quinn","Morgan","Jamie"
    ]
    last_names = [
        "Smith","Lee","Patel","Garcia","Khan","Zhang","Silva","Nguyen","Kumar","Brown"
    ]
    for i in rng:
        employees.append({
            "emp_id": int(i),
            "first_name": first_names[(i-1) % len(first_names)],
            "last_name": last_names[(i-1) % len(last_names)],
            "dept_id": [10,20,30,40][(i-1) % 4],
            "hire_date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(i)*37),
        })
    employees = pd.DataFrame(employees)

    salaries = pd.DataFrame({
        "emp_id": employees.emp_id,
        "salary": (employees.emp_id * 3500 + 60000).astype(int),
        "effective_date": pd.Timestamp("2024-01-01"),
    })

    # Products & Orders
    products = pd.DataFrame([
        {"product_id": 1, "product_name": "Laptop Pro", "unit_price": 1500.0},
        {"product_id": 2, "product_name": "Laptop Air", "unit_price": 1100.0},
        {"product_id": 3, "product_name": "4K Monitor", "unit_price": 400.0},
        {"product_id": 4, "product_name": "Mechanical Keyboard", "unit_price": 120.0},
        {"product_id": 5, "product_name": "Wireless Mouse", "unit_price": 60.0},
        {"product_id": 6, "product_name": "USB-C Hub", "unit_price": 80.0},
    ])

    orders = []
    order_items = []
    oid = 1
    for emp in employees.emp_id.sample(30, random_state=42):
        orders.append({"order_id": oid, "emp_id": int(emp), "order_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=oid)})
        # each order has 1-3 items
        for pid in products.product_id.sample((oid % 3) + 1, random_state=oid):
            qty = (pid % 4) + 1
            price = float(products.loc[products.product_id == pid, "unit_price"].values[0])
            order_items.append({
                "order_id": oid,
                "product_id": int(pid),
                "quantity": int(qty),
                "unit_price": price,
            })
        oid += 1
    orders = pd.DataFrame(orders)
    order_items = pd.DataFrame(order_items)

    # Load into DuckDB
    con.register("reg_depts", depts)
    con.register("reg_emps", employees)
    con.register("reg_sals", salaries)
    con.register("reg_products", products)
    con.register("reg_orders", orders)
    con.register("reg_order_items", order_items)

    con.execute("CREATE OR REPLACE TABLE departments AS SELECT * FROM reg_depts;")
    con.execute("CREATE OR REPLACE TABLE employees AS SELECT * FROM reg_emps;")
    con.execute("CREATE OR REPLACE TABLE salaries AS SELECT * FROM reg_sals;")
    con.execute("CREATE OR REPLACE TABLE products AS SELECT * FROM reg_products;")
    con.execute("CREATE OR REPLACE TABLE orders AS SELECT * FROM reg_orders;")
    con.execute("CREATE OR REPLACE TABLE order_items AS SELECT * FROM reg_order_items;")

    st.session_state.loaded_samples = True


# -------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    dialect = st.selectbox("Your SQL dialect", DIALECTS, index=0, help="If your query is T‑SQL/MySQL/Postgres/etc., pick it here. We'll transpile to DuckDB where possible.")

    st.divider()
    st.subheader("Data sources")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Load sample data", use_container_width=True):
            load_sample_data()
            st.success("Sample tables created: employees, departments, salaries, products, orders, order_items")
    with col_b:
        if st.button("Reset database", type="secondary", use_container_width=True):
            # Recreate fresh in-memory DB
            con.close()
            st.session_state.con = duckdb.connect(database=":memory:")
            st.session_state.loaded_samples = False
            st.session_state.query_history = []
            st.session_state.last_result = None
            st.session_state.show_history = False
            st.rerun()

    st.caption("Sample data are optional. You can also upload your own files or create a table via SQL.")

    uploaded = st.file_uploader("Upload CSV or Parquet to create a table", type=["csv", "parquet"], accept_multiple_files=False)
    if uploaded is not None:
        default_name = uploaded.name.rsplit(".", 1)[0].replace("-", "_").replace(" ", "_")
        new_table_name = st.text_input("Table name", value=default_name)
        sanitize_cols = st.checkbox("Sanitize column names (snake_case)", value=True, help="Make identifiers SQL-friendly so you don't need double quotes.")
        if st.button("Create table from file"):
            try:
                if uploaded.name.endswith(".csv"):
                    raw = uploaded.getvalue()  # bytes
                    enc_candidates = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
                    detected = None
                    if _cn_from_bytes is not None:
                        try:
                            res = _cn_from_bytes(raw).best()
                            if res and res.encoding:
                                enc_candidates.insert(0, res.encoding)
                        except Exception:
                            pass
                    last_err = None
                    df = None
                    for enc in enc_candidates:
                        try:
                            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                            detected = enc
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                    if df is None:
                        raw_fixed = raw.replace(b"\xa0", b" ")
                        for enc in enc_candidates:
                            try:
                                df = pd.read_csv(io.BytesIO(raw_fixed), encoding=enc, engine="python")
                                detected = f"{enc} (NBSP fixed)"
                                last_err = None
                                break
                            except Exception as e:
                                last_err = e
                    if df is None and last_err is not None:
                        raise last_err
                    st.caption(f"Detected/used encoding: {detected}")
                else:
                    df = pd.read_parquet(uploaded)

                if sanitize_cols:
                    old_cols = list(df.columns)
                    new_cols = sanitize_columns(df.columns)
                    df.columns = new_cols
                    if old_cols != new_cols:
                        try:
                            pairs = ", ".join([f"{o}→{n}" for o, n in zip(old_cols, new_cols)])
                            st.caption("Sanitized columns: " + (pairs[:3000] + (" …" if len(pairs) > 3000 else "")))
                        except Exception:
                            pass

                con.register("_uploaded_df", df)
                con.execute(f"CREATE OR REPLACE TABLE {safe_ident(new_table_name)} AS SELECT * FROM _uploaded_df;")
                st.success(f"Created/updated table `{new_table_name}` with {len(df)} rows.")
            except Exception as e:
                st.error(f"Could not create table: {e}")

    st.divider()
    st.subheader("Schema browser")
    try:
        tbls = list_tables()
        if len(tbls) == 0:
            st.info("No tables yet. Load samples, upload a file, or run CREATE TABLE.")
        else:
            for _, row in tbls.iterrows():
                with st.expander(f"{row.table_schema}.{row.table_name}"):
                    cols = table_columns(row.table_schema, row.table_name)
                    st.dataframe(cols, use_container_width=True, hide_index=True)
                    st.caption("Preview")
                    st.dataframe(preview_table(row.table_schema, row.table_name), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Schema unavailable: {e}")

    # Column tools
    st.divider()
    st.subheader("🛠 Column tools")
    try:
        tbls_tools = list_tables()
        if len(tbls_tools) == 0:
            st.caption("No tables yet to edit.")
        else:
            table_options = [f"{r.table_schema}.{r.table_name}" for _, r in tbls_tools.iterrows()]
            selected_table = st.selectbox("Select table", table_options, key="ct_table")
            if selected_table:
                schema, table = selected_table.split(".", 1)
                cols_df = table_columns(schema, table)
                st.dataframe(cols_df, use_container_width=True, hide_index=True)
                st.caption("Rename columns or change their data types (safe in-place swap).")

                colL, colR = st.columns(2)
                with colL:
                    col_to_rename = st.selectbox("Column to rename", cols_df["column_name"].tolist(), key="ct_rename_col")
                    suggested = snake_case(col_to_rename)
                    new_name = st.text_input("New column name", value=suggested, key="ct_new_name")
                    if st.button("Rename column", use_container_width=True):
                        try:
                            rename_column(schema, table, col_to_rename, new_name)
                            st.success(f"Renamed {col_to_rename} → {new_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Rename failed: {e}")
                with colR:
                    col_to_cast = st.selectbox("Column to change type", cols_df["column_name"].tolist(), key="ct_cast_col")
                    target_type = st.selectbox(
                        "Target type",
                        ["INTEGER","BIGINT","DOUBLE","DECIMAL(18,2)","VARCHAR","DATE","TIMESTAMP","BOOLEAN"],
                        key="ct_target_type"
                    )
                    strip = st.checkbox("Strip non‑numeric characters", value=True, key="ct_strip")
                    comma_dec = st.checkbox("Treat comma as decimal", value=False, key="ct_commadec")
                    if st.button("Apply type change", use_container_width=True):
                        try:
                            alter_column_type(schema, table, col_to_cast, target_type, strip_non_numeric=strip, comma_decimal=comma_dec)
                            st.success(f"Changed type of {col_to_cast} to {target_type}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Type change failed: {e}")
    except Exception as e:
        st.warning(f"Column tools unavailable: {e}")

# -------------------------------------------------------------
# Main editor & runner
# -------------------------------------------------------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("SQL Editor")
    example = "SELECT * FROM employees LIMIT 10;"
    sql_text = st.text_area(
        "Write SQL here (multiple statements allowed)",
        height=260,
        placeholder=example,
    )

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        run_clicked = st.button("▶ Run (Ctrl/Cmd+Enter)", type="primary", use_container_width=True)
    with c2:
        if st.button("Load a random sample query", use_container_width=True):
            # rotate samples
            key = list(SAMPLE_QUERIES.keys())[len(st.session_state.query_history) % len(SAMPLE_QUERIES)]
            st.session_state.sample_sql = SAMPLE_QUERIES[key]
            st.session_state.sample_name = key
            st.rerun()
    with c3:
        if st.button("Show history", use_container_width=True):
            st.session_state.show_history = True

    # If user asked to load a sample, populate the editor
    if "sample_sql" in st.session_state and not sql_text.strip():
        sql_text = st.session_state.sample_sql
        st.info(f"Loaded sample: {st.session_state.get('sample_name','sample query')}")

    if run_clicked and sql_text.strip():
        with st.spinner("Running SQL…"):
            results = run_sql(sql_text, dialect)
        st.session_state.query_history.append({
            "sql": sql_text,
            "dialect": dialect,
            "results": results,
            "ts": pd.Timestamp.utcnow(),
        })

    # Display latest results immediately
    if st.session_state.query_history:
        latest = st.session_state.query_history[-1]["results"]
        # Separate result-sets and messages
        result_sets = [r for r in latest if r.get("rows") is not None]
        messages = [r for r in latest if r.get("rows") is None]

        if result_sets:
            tabs = st.tabs([f"Result {i+1}" for i in range(len(result_sets))])
            for tab, res in zip(tabs, result_sets):
                with tab:
                    st.caption(f"⏱ {res['elapsed_ms']:.1f} ms")
                    st.dataframe(res["rows"], use_container_width=True, hide_index=True)
                    csv = res["rows"].to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", csv, file_name="result.csv")
        if messages:
            st.divider()
            for r in messages:
                if r["ok"]:
                    st.success(f"OK — executed in {r['elapsed_ms']:.1f} ms\n\n**Statement:** {r['statement']}")
                else:
                    with st.expander("❌ Error (click for details)", expanded=True):
                        st.error(r["error"]) 
                        st.code(r["statement"], language="sql")
                        if r.get("duckdb") and r["duckdb"] != r["statement"]:
                            st.caption("Transpiled to DuckDB:")
                            st.code(r["duckdb"], language="sql")

with right:
    st.subheader("Tips")
    st.markdown(
        """
        - Use **Load sample data** from the sidebar to get instant practice tables.
        - You can **CREATE TABLE**, **INSERT**, **UPDATE**, **DELETE**, **ALTER**, **CREATE VIEW**, **CTEs**, **JOINs**, **WINDOW functions**, etc.
        - If you're pasting **SQL Server**/**MySQL**/**Postgres** code, set *Your SQL dialect* to help auto‑convert.
        - Upload your own **CSV/Parquet** to create a table with one click.
        - Click **Download CSV** under results to export.
        """
    )

    st.subheader("Sample queries")
    SAMPLE_QUERIES = {
        "Select employees earning > 100k": "SELECT e.emp_id, e.first_name, e.last_name, s.salary FROM employees e JOIN salaries s USING(emp_id) WHERE s.salary > 100000 ORDER BY s.salary DESC;",
        "Department headcount": "SELECT d.dept_name, COUNT(*) AS headcount FROM employees e JOIN departments d USING(dept_id) GROUP BY d.dept_name ORDER BY headcount DESC;",
        "Top product revenue": "SELECT p.product_id, p.product_name, SUM(oi.quantity*oi.unit_price) AS revenue FROM order_items oi JOIN products p USING(product_id) GROUP BY 1,2 ORDER BY revenue DESC LIMIT 5;",
        "Window fn: salary rank per dept": """
            SELECT dept_id, emp_id, salary,
                   RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
            FROM salaries
            ORDER BY dept_id, rnk;
        """.strip(),
        "CTE example": """
            WITH big_spenders AS (
                SELECT emp_id, salary FROM salaries WHERE salary > 120000
            )
            SELECT e.first_name, e.last_name, b.salary
            FROM employees e JOIN big_spenders b USING(emp_id)
            ORDER BY b.salary DESC;
        """.strip(),
    }
    for name, q in SAMPLE_QUERIES.items():
        with st.expander(name):
            st.code(q, language="sql")

    if st.session_state.query_history and st.session_state.get("show_history"):
        st.divider()
        st.subheader("History (this session)")
        for i, h in enumerate(reversed(st.session_state.query_history[-20:]), start=1):
            st.caption(f"{i}. {h['ts']} — dialect: {h['dialect']}")
            st.code(h["sql"], language="sql")

# End of app
