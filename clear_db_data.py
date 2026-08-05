"""
Clears all ROWS from every table in the 'public' schema (visitors, ai_team,
visitor_logs, etc.) on the Neon Postgres DB, while keeping tables, columns,
constraints, and schema fully intact.

Works regardless of FK relationships (visitor_logs -> visitors / ai_team)
because CASCADE handles dependency order automatically.

Setup:
    pip install psycopg2-binary python-dotenv

.env file (in project root) should contain:
    DATABASE_URL=postgresql://neondb_owner:xxxx@xxxx.neon.tech/neondb?sslmode=require

Usage:
    python clear_db_data.py
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not found in .env")
    sys.exit(1)

TRUNCATE_ALL_SQL = """
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public')
    LOOP
        EXECUTE 'TRUNCATE TABLE public.' || quote_ident(r.tablename) || ' RESTART IDENTITY CASCADE';
    END LOOP;
END $$;
"""


def get_row_counts(cur, tables):
    counts = {}
    for t in tables:
        cur.execute(f'SELECT COUNT(*) FROM public."{t}"')
        counts[t] = cur.fetchone()[0]
    return counts


def clear_all_data():
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        tables = [row[0] for row in cur.fetchall()]

        if not tables:
            print("No tables found in the public schema. Nothing to do.")
            return

        before_counts = get_row_counts(cur, tables)

        print("The following tables will have ALL their data cleared:")
        for t in tables:
            print(f"  - {t} ({before_counts[t]} rows)")

        confirm = input("\nType 'yes' to proceed: ").strip().lower()
        if confirm != "yes":
            print("Aborted. No changes made.")
            return

        cur.execute(TRUNCATE_ALL_SQL)

        after_counts = get_row_counts(cur, tables)
        print("\n✅ Done. Row counts after truncate:")
        for t in tables:
            print(f"  - {t}: {after_counts[t]} rows")

        print("\nSchema, columns, PKs/FKs/UKs are untouched.")

        cur.close()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    clear_all_data()