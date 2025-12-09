"""
Database Setup Script
This script creates the DuckDB database structure and syncs pangrams and passages with CSV files.
CSV files are the source of truth - items not in CSV will be removed from database.
For speakers, it only creates the table structure - use populate_speakers.py to add initial speakers.
"""

import duckdb
import csv
import os
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "database.duckdb"
PANGRAMS_CSV = SCRIPT_DIR / "pangrams.csv"
PASSAGES_CSV = SCRIPT_DIR / "passages.csv"


def setup_database():
    """
    Sets up the DuckDB tables for storing speaker, pangram, and passage data.
    Speakers table: Creates structure only (use populate_speakers.py to add speakers)
    Pangrams/Passages: Creates and syncs with CSV files (CSV is source of truth)
    """
    print("=" * 80)
    print("DATABASE SETUP")
    print("=" * 80)
    print(f"Database: {DB_PATH}\n")
    
    con = duckdb.connect(str(DB_PATH))
    
    try:
        # Check if speakers table exists
        speakers_exists = con.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = 'speakers'
        """).fetchone()[0] > 0
        
        if speakers_exists:
            print("✓ Speakers table already exists (not modified)")
        else:
            # Create speakers table structure only
            con.execute("""
                CREATE SEQUENCE IF NOT EXISTS seq_uid START 1;
                CREATE TABLE IF NOT EXISTS speakers (
                    uid INTEGER PRIMARY KEY DEFAULT nextval('seq_uid'),
                    firstname VARCHAR NOT NULL,
                    surname VARCHAR,
                    gpt_cond_latent FLOAT[],
                    gpt_shape VARCHAR,
                    xtts_embedding FLOAT[],
                    xtts_shape VARCHAR,
                    ecapa_embedding FLOAT[],
                    total_duration_sec FLOAT DEFAULT 0.0,
                    sample_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    pangrams INTEGER[] DEFAULT []
                );
            """)
            print("✓ Created speakers table (empty - use populate_speakers.py to add speakers)")
        
        # Create pangrams table
        con.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_pangram_id START 1;
            CREATE TABLE IF NOT EXISTS pangrams (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_pangram_id'),
                text VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("✓ Pangrams table ready")
        
        # Create passages table
        con.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_passage_id START 1;
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_passage_id'),
                source VARCHAR NOT NULL,
                quote TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("✓ Passages table ready\n")
        
        # Populate pangrams
        populate_pangrams(con)
        
        # Populate passages
        populate_passages(con)
        
    finally:
        con.close()
    
    print("\n" + "=" * 80)
    print("✓ Database setup complete!")
    print("=" * 80)
    if not speakers_exists:
        print("\n💡 Note: Speakers table is empty. To add initial speakers, run:")
        print("   python populate_speakers.py")
    print()


def populate_pangrams(con):
    """
    Sync pangrams table with pangrams.csv (CSV is source of truth).
    Inserts new pangrams and removes pangrams not in CSV.
    Handles duplicate pangrams in the database.
    """
    print("=== Syncing Pangrams ===")
    
    if not PANGRAMS_CSV.exists():
        print(f"⚠ Warning: {PANGRAMS_CSV} not found, skipping pangrams")
        return
    
    # Read pangrams from CSV
    csv_pangrams = set()
    with open(PANGRAMS_CSV, 'r', encoding='utf-8-sig') as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f)
        for row in reader:
            csv_pangrams.add(row['text'].strip())
    
    print(f"Found {len(csv_pangrams)} pangram(s) in CSV")
    
    # Get ALL pangrams from database (including duplicates)
    all_db_pangrams = con.execute("SELECT id, text FROM pangrams").fetchall()
    db_pangram_ids = {}  # text -> list of IDs
    for row in all_db_pangrams:
        text = row[1]
        if text not in db_pangram_ids:
            db_pangram_ids[text] = []
        db_pangram_ids[text].append(row[0])
    
    print(f"Found {len(all_db_pangrams)} existing pangram(s) in database")
    if len(all_db_pangrams) > len(db_pangram_ids):
        print(f"  ⚠ Warning: {len(all_db_pangrams) - len(db_pangram_ids)} duplicate(s) detected")
    
    # Remove pangrams not in CSV (including ALL duplicate instances)
    removed_count = 0
    for text, ids in db_pangram_ids.items():
        if text not in csv_pangrams:
            # Delete all instances of this pangram
            for pangram_id in ids:
                con.execute("DELETE FROM pangrams WHERE id = ?", [pangram_id])
                removed_count += 1
                print(f"  - Removed ID#{pangram_id}: {text[:60]}...")
        elif len(ids) > 1:
            # Keep one, remove duplicates
            for pangram_id in ids[1:]:
                con.execute("DELETE FROM pangrams WHERE id = ?", [pangram_id])
                removed_count += 1
                print(f"  - Removed duplicate ID#{pangram_id}: {text[:60]}...")
    
    if removed_count > 0:
        print(f"✓ Removed {removed_count} pangram(s)")
    
    # Get unique pangrams currently in database after cleanup
    existing_pangrams = set(db_pangram_ids.keys())
    
    # Insert new pangrams
    to_insert = csv_pangrams - existing_pangrams
    inserted_count = 0
    for pangram_text in to_insert:
        con.execute("INSERT INTO pangrams (text) VALUES (?)", [pangram_text])
        inserted_count += 1
        print(f"  + Inserted: {pangram_text[:60]}...")
    
    if inserted_count == 0 and removed_count == 0:
        print("  No changes needed")
    elif inserted_count > 0:
        print(f"✓ Inserted {inserted_count} new pangram(s)")

def populate_passages(con):
    """
    Sync passages table with passages.csv (CSV is source of truth).
    Inserts new passages and removes passages not in CSV.
    Handles duplicate passages in the database.
    """
    print("\n=== Syncing Passages ===")
    
    if not PASSAGES_CSV.exists():
        print(f"⚠ Warning: {PASSAGES_CSV} not found, skipping passages")
        return
    
    # Read passages from CSV
    csv_passages = set()
    with open(PASSAGES_CSV, 'r', encoding='utf-8-sig') as f:  # utf-8-sig strips BOM
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            source = row[0].strip()
            quote = row[1].strip()
            csv_passages.add((source, quote))
    
    print(f"Found {len(csv_passages)} passage(s) in CSV")
    
    # Get ALL passages from database (including duplicates)
    all_db_passages = con.execute("SELECT id, source, quote FROM passages").fetchall()
    db_passage_ids = {}  # (source, quote) -> list of IDs
    for row in all_db_passages:
        key = (row[1], row[2])
        if key not in db_passage_ids:
            db_passage_ids[key] = []
        db_passage_ids[key].append(row[0])
    
    print(f"Found {len(all_db_passages)} existing passage(s) in database")
    if len(all_db_passages) > len(db_passage_ids):
        print(f"  ⚠ Warning: {len(all_db_passages) - len(db_passage_ids)} duplicate(s) detected")
    
    # Remove passages not in CSV (including ALL duplicate instances)
    removed_count = 0
    for key, ids in db_passage_ids.items():
        if key not in csv_passages:
            source, quote = key
            # Delete all instances of this passage
            for passage_id in ids:
                con.execute("DELETE FROM passages WHERE id = ?", [passage_id])
                removed_count += 1
                print(f"  - Removed ID#{passage_id} from '{source}': {quote[:50]}...")
        elif len(ids) > 1:
            # Keep one, remove duplicates
            for passage_id in ids[1:]:
                source, quote = key
                con.execute("DELETE FROM passages WHERE id = ?", [passage_id])
                removed_count += 1
                print(f"  - Removed duplicate ID#{passage_id} from '{source}': {quote[:50]}...")
    
    if removed_count > 0:
        print(f"✓ Removed {removed_count} passage(s)")
    
    # Get unique passages currently in database after cleanup
    existing_passages = set(db_passage_ids.keys())
    
    # Insert new passages
    to_insert = csv_passages - existing_passages
    inserted_count = 0
    for source, quote in to_insert:
        con.execute("INSERT INTO passages (source, quote) VALUES (?, ?)", [source, quote])
        inserted_count += 1
        print(f"  + Inserted from '{source}': {quote[:50]}...")
    
    if inserted_count == 0 and removed_count == 0:
        print("  No changes needed")
    elif inserted_count > 0:
        print(f"✓ Inserted {inserted_count} new passage(s)")

if __name__ == "__main__":
    setup_database()
