"""
Database Validation Script
This script checks the database structure, verifies it's up to date with CSV files,
and displays the contents in a readable format.
"""

import duckdb
import csv
from pathlib import Path
from datetime import datetime

# Configuration
SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "database.duckdb"
PANGRAMS_CSV = SCRIPT_DIR / "pangrams.csv"
PASSAGES_CSV = SCRIPT_DIR / "passages.csv"


def check_database():
    """
    Validate database structure and contents.
    """
    print("=" * 80)
    print("DATABASE VALIDATION REPORT")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if not DB_PATH.exists():
        print("\n❌ ERROR: Database file not found!")
        print(f"   Expected location: {DB_PATH}")
        print("   Please run setup_database.py first.")
        return
    
    con = duckdb.connect(str(DB_PATH))
    
    try:
        # Check table structure
        print("\n### TABLE STRUCTURE CHECK ###\n")
        check_table_structure(con)
        
        # Check sync status with CSVs
        print("\n### CSV SYNC STATUS ###\n")
        check_csv_sync(con)
        
        # Display table contents
        print("\n### TABLE CONTENTS ###\n")
        display_speakers(con)
        display_pangrams(con)
        display_passages(con)
        
    finally:
        con.close()
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


def check_table_structure(con):
    """
    Verify all required tables exist and have correct structure.
    """
    required_tables = ['speakers', 'pangrams', 'passages']
    
    for table_name in required_tables:
        exists = con.execute(f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        """).fetchone()[0] > 0
        
        if exists:
            row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"✓ {table_name.upper()} table exists ({row_count} records)")
        else:
            print(f"❌ {table_name.upper()} table missing!")


def check_csv_sync(con):
    """
    Check if database is in sync with CSV files.
    """
    # Check pangrams
    if PANGRAMS_CSV.exists():
        csv_pangrams = []
        with open(PANGRAMS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            csv_pangrams = [row['text'].strip() for row in reader]
        
        db_pangrams = [row[0] for row in con.execute("SELECT text FROM pangrams").fetchall()]
        
        csv_set = set(csv_pangrams)
        db_set = set(db_pangrams)
        
        missing_in_db = csv_set - db_set
        extra_in_db = db_set - csv_set
        
        if not missing_in_db and not extra_in_db:
            print(f"✓ Pangrams in sync ({len(csv_pangrams)} records)")
        else:
            print(f"⚠ Pangrams out of sync:")
            if missing_in_db:
                print(f"  - {len(missing_in_db)} in CSV but not in DB")
            if extra_in_db:
                print(f"  - {len(extra_in_db)} in DB but not in CSV")
    else:
        print(f"⚠ {PANGRAMS_CSV} not found")
    
    # Check passages
    if PASSAGES_CSV.exists():
        csv_passages = []
        with open(PASSAGES_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            csv_passages = [(row[0].strip(), row[1].strip()) for row in reader if len(row) >= 2]
        
        db_passages = [(row[0], row[1]) for row in con.execute("SELECT source, quote FROM passages").fetchall()]
        
        csv_set = set(csv_passages)
        db_set = set(db_passages)
        
        missing_in_db = csv_set - db_set
        extra_in_db = db_set - csv_set
        
        if not missing_in_db and not extra_in_db:
            print(f"✓ Passages in sync ({len(csv_passages)} records)")
        else:
            print(f"⚠ Passages out of sync:")
            if missing_in_db:
                print(f"  - {len(missing_in_db)} in CSV but not in DB")
            if extra_in_db:
                print(f"  - {len(extra_in_db)} in DB but not in CSV")
    else:
        print(f"⚠ {PASSAGES_CSV} not found")


def display_speakers(con):
    """
    Display speakers table contents.
    """
    print("─" * 80)
    print("SPEAKERS TABLE")
    print("─" * 80)
    
    results = con.execute("""
        SELECT 
            uid, 
            firstname, 
            surname, 
            total_duration_sec, 
            sample_count, 
            pangrams,
            last_updated
        FROM speakers
        ORDER BY uid
    """).fetchall()
    
    if not results:
        print("  (No speakers in database)")
    else:
        for row in results:
            uid, firstname, surname, duration, samples, pangrams, updated = row
            
            print(f"\nSpeaker #{uid}: {firstname} {surname or ''}")
            print(f"  Total Duration: {duration:.2f} seconds")
            print(f"  Sample Count: {samples}")
            print(f"  Pangrams: {pangrams if pangrams else '[]'}")
            print(f"  Last Updated: {updated}")


def display_pangrams(con):
    """
    Display pangrams table contents.
    """
    print("\n" + "─" * 80)
    print("PANGRAMS TABLE")
    print("─" * 80)
    
    results = con.execute("""
        SELECT id, text, created_at
        FROM pangrams
        ORDER BY id
    """).fetchall()
    
    if not results:
        print("  (No pangrams in database)")
    else:
        for row in results:
            pangram_id, text, created = row
            print(f"\nPangram #{pangram_id}:")
            print(f"  {text}")
            print(f"  Created: {created}")


def display_passages(con):
    """
    Display passages table contents.
    """
    print("\n" + "─" * 80)
    print("PASSAGES TABLE")
    print("─" * 80)
    
    results = con.execute("""
        SELECT id, source, quote, created_at
        FROM passages
        ORDER BY source, id
    """).fetchall()
    
    if not results:
        print("  (No passages in database)")
    else:
        current_source = None
        for row in results:
            passage_id, source, quote, created = row
            
            # Print source header when it changes
            if source != current_source:
                print(f"\n[{source}]")
                current_source = source
            
            # Wrap long quotes for readability
            if len(quote) > 70:
                print(f"  #{passage_id}: {quote[:67]}...")
            else:
                print(f"  #{passage_id}: {quote}")


if __name__ == "__main__":
    check_database()
