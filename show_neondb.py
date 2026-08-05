"""
Shows the complete PostgreSQL database structure and all stored data.

Install:
    pip install psycopg2-binary python-dotenv

.env
DATABASE_URL=postgresql://username:password@host/dbname?sslmode=require
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")


from psycopg2 import sql


def write_line(file, text=""):
    print(text)
    file.write(text + "\n")


def show_database():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    with open("database_report.txt", "w", encoding="utf-8") as report:

        cur.execute("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname='public'
            ORDER BY tablename;
        """)

        tables = [row[0] for row in cur.fetchall()]

        if not tables:
            write_line(report, "No tables found.")
            return

        write_line(report, "=" * 100)
        write_line(report, "POSTGRES DATABASE REPORT")
        write_line(report, "=" * 100)

        for table in tables:

            try:

                write_line(report, "")
                write_line(report, "#" * 100)
                write_line(report, f"TABLE : {table}")
                write_line(report, "#" * 100)

                # ----------------------------------------------------
                # Columns
                # ----------------------------------------------------
                cur.execute("""
                    SELECT
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns
                    WHERE table_schema='public'
                    AND table_name=%s
                    ORDER BY ordinal_position;
                """, (table,))

                write_line(report, "\nColumns:\n")

                for column in cur.fetchall():
                    write_line(
                        report,
                        f"{column[0]:25} "
                        f"{column[1]:20} "
                        f"Nullable={column[2]:3} "
                        f"Default={column[3]}"
                    )

                # ----------------------------------------------------
                # Primary Keys
                # ----------------------------------------------------
                cur.execute("""
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name=kcu.constraint_name
                    WHERE tc.table_schema='public'
                    AND tc.table_name=%s
                    AND tc.constraint_type='PRIMARY KEY';
                """, (table,))

                pks = [r[0] for r in cur.fetchall()]
                write_line(report, "\nPrimary Keys:")
                write_line(report, str(pks) if pks else "None")

                # ----------------------------------------------------
                # Foreign Keys
                # ----------------------------------------------------
                cur.execute("""
                    SELECT
                        kcu.column_name,
                        ccu.table_name,
                        ccu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name=kcu.constraint_name
                    JOIN information_schema.constraint_column_usage ccu
                        ON tc.constraint_name=ccu.constraint_name
                    WHERE tc.constraint_type='FOREIGN KEY'
                    AND tc.table_schema='public'
                    AND tc.table_name=%s;
                """, (table,))

                fks = cur.fetchall()

                write_line(report, "\nForeign Keys:")

                if fks:
                    for fk in fks:
                        write_line(report, f"{fk[0]} -> {fk[1]}.{fk[2]}")
                else:
                    write_line(report, "None")

                # ----------------------------------------------------
                # Row Count
                # ----------------------------------------------------
                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}").format(
                        sql.Identifier("public", table)
                    )
                )

                count = cur.fetchone()[0]

                write_line(report, f"\nTotal Rows : {count}")

                # ----------------------------------------------------
                # Data
                # ----------------------------------------------------
                if count > 0:

                    cur.execute(
                        sql.SQL("SELECT * FROM {}").format(
                            sql.Identifier("public", table)
                        )
                    )

                    columns = [desc[0] for desc in cur.description]

                    write_line(report, "\nColumn Names:")
                    write_line(report, ", ".join(columns))

                    write_line(report, "\nRows:\n")

                    for i, row in enumerate(cur.fetchall(), start=1):
                        write_line(report, f"Row {i}")
                        for col, value in zip(columns, row):
                            write_line(report, f"   {col}: {value}")
                        write_line(report)

                else:
                    write_line(report, "\nTable is empty.")

            except Exception as e:
                write_line(report, f"\nError reading table '{table}': {e}")

        write_line(report, "\n\nDatabase export completed successfully.")

    cur.close()
    conn.close()

    print("\nReport saved as: database_report.txt")

if __name__ == "__main__":
    show_database()