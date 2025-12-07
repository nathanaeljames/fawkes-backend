"""
Database Setup Script
This script creates the DuckDB database structure and populates pangrams and passages.
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
    Pangrams/Passages: Creates and populates from CSV files
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
    Populate pangrams table from pangrams.csv.
    Only inserts new pangrams that don't already exist.
    """
    print("=== Populating Pangrams ===")
    
    if not PANGRAMS_CSV.exists():
        print(f"⚠ Warning: {PANGRAMS_CSV} not found, skipping pangrams")
        return
    
    # Get existing pangrams
    existing_pangrams = set()
    existing = con.execute("SELECT text FROM pangrams").fetchall()
    for row in existing:
        existing_pangrams.add(row[0])
    
    print(f"Found {len(existing_pangrams)} existing pangram(s)")
    
    # Read and insert new pangrams
    inserted_count = 0
    with open(PANGRAMS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pangram_text = row['text'].strip()
            
            if pangram_text not in existing_pangrams:
                con.execute("""
                    INSERT INTO pangrams (text) VALUES (?)
                """, [pangram_text])
                inserted_count += 1
                print(f"  + Inserted: {pangram_text[:60]}...")
    
    if inserted_count == 0:
        print("  No new pangrams to insert")
    else:
        print(f"✓ Inserted {inserted_count} new pangram(s)")


def populate_passages(con):
    """
    Populate passages table from passages.csv.
    Only inserts new passages that don't already exist.
    """
    print("\n=== Populating Passages ===")
    
    if not PASSAGES_CSV.exists():
        print(f"⚠ Warning: {PASSAGES_CSV} not found, skipping passages")
        return
    
    # Get existing passages (using source + quote as unique identifier)
    existing_passages = set()
    existing = con.execute("SELECT source, quote FROM passages").fetchall()
    for row in existing:
        existing_passages.add((row[0], row[1]))
    
    print(f"Found {len(existing_passages)} existing passage(s)")
    
    # Read and insert new passages
    inserted_count = 0
    with open(PASSAGES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            
            source = row[0].strip()
            quote = row[1].strip()
            
            if (source, quote) not in existing_passages:
                con.execute("""
                    INSERT INTO passages (source, quote) VALUES (?, ?)
                """, [source, quote])
                inserted_count += 1
                print(f"  + Inserted from '{source}': {quote[:50]}...")
    
    if inserted_count == 0:
        print("  No new passages to insert")
    else:
        print(f"✓ Inserted {inserted_count} new passage(s)")


if __name__ == "__main__":
    setup_database()
