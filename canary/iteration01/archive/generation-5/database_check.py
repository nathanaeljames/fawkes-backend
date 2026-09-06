#!/usr/bin/env python3
"""
Check database schema and identify any issues
"""
import duckdb

DUCKDB_PATH = "./speakers/speakers.duckdb"

con = duckdb.connect(DUCKDB_PATH)

print("=== Pangrams Table Schema ===")
schema = con.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'pangrams'
    ORDER BY ordinal_position
""").fetchall()

for col in schema:
    print(f"  {col[0]}: {col[1]}")

print("\n=== Speakers Table Schema ===")
schema = con.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'speakers'
    ORDER BY ordinal_position
""").fetchall()

for col in schema:
    print(f"  {col[0]}: {col[1]}")

print("\n=== Pangrams Count ===")
count = con.execute("SELECT COUNT(*) FROM pangrams").fetchone()[0]
print(f"Total pangrams: {count}")

if count > 0:
    print("\n=== Pangrams Preview ===")
    pangrams = con.execute("SELECT id, LEFT(text, 60) || '...' FROM pangrams").fetchall()
    for p in pangrams:
        print(f"  ID {p[0]}: {p[1]}")

con.close()