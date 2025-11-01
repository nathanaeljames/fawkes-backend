#!/usr/bin/env python3
"""
One-time database update script for DuckDB speakers database
This script:
1. Updates existing speaker names (nathanael/courtney)
2. Creates the pangrams table
3. Adds pangrams column to speakers table
4. Inserts initial pangrams

Run this ONCE before updating server01e.py with the new schema
"""

import duckdb

# Path to your DuckDB database
DUCKDB_PATH = "./speakers/speakers.duckdb"

def update_speaker_names(con):
    """Update existing speaker names to proper capitalization"""
    print("\n=== Updating Speaker Names ===")
    
    # Update nathanael warren
    result = con.execute("""
        UPDATE speakers 
        SET firstname = 'Nathanael', surname = 'Warren'
        WHERE firstname = 'nathanael' AND surname = 'warren'
    """)
    print(f"Updated nathanael warren → Nathanael Warren: {result.fetchone()[0]} rows affected")
    
    # Update courtney mosierwarren
    result = con.execute("""
        UPDATE speakers 
        SET firstname = 'Courtney', surname = 'Mosier Warren'
        WHERE firstname = 'courtney' AND surname = 'mosierwarren'
    """)
    print(f"Updated courtney mosierwarren → Courtney Mosier Warren: {result.fetchone()[0]} rows affected")
    
    # Verify the changes
    print("\nVerifying updated names:")
    results = con.execute("""
        SELECT uid, firstname, surname 
        FROM speakers 
        WHERE (firstname = 'Nathanael' AND surname = 'Warren')
           OR (firstname = 'Courtney' AND surname = 'Mosier Warren')
    """).fetchall()
    
    for row in results:
        print(f"  UID {row[0]}: {row[1]} {row[2]}")

def create_pangrams_table(con):
    """Create the pangrams table with auto-incrementing ID"""
    print("\n=== Creating Pangrams Table ===")
    
    # Create sequence for pangram IDs
    con.execute("CREATE SEQUENCE IF NOT EXISTS seq_pangram_id START 1")
    
    # Create pangrams table
    con.execute("""
        CREATE TABLE IF NOT EXISTS pangrams (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_pangram_id'),
            text VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("Pangrams table created successfully")

def add_pangrams_column_to_speakers(con):
    """Add pangrams column to speakers table to track recited pangrams"""
    print("\n=== Adding Pangrams Column to Speakers Table ===")
    
    # Check if column already exists
    columns = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'speakers'
    """).fetchall()
    
    column_names = [col[0] for col in columns]
    
    if 'pangrams' not in column_names:
        con.execute("""
            ALTER TABLE speakers 
            ADD COLUMN pangrams INTEGER[] DEFAULT []
        """)
        print("Added 'pangrams' column to speakers table")
    else:
        print("'pangrams' column already exists in speakers table")

def insert_initial_pangrams(con):
    """Insert the initial set of pangrams"""
    print("\n=== Inserting Initial Pangrams ===")
    
    pangrams = [
        "The beige hue on the waters of the loch impressed all, including the French queen, before she heard that symphony again, just as young Arthur wanted.",
        
        "The pleasure of Shawn's company is what I most enjoy. He put a tack on Ms. Yancey's chair when she called him a horrible boy. At the end of the month he was flinging two kittens across the width of the room. I count on his schemes to show me a way now of getting away from my gloom.",
        
        "Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
    ]
    
    # Check if pangrams already exist
    existing_count = con.execute("SELECT COUNT(*) FROM pangrams").fetchone()[0]
    
    if existing_count > 0:
        print(f"Pangrams table already contains {existing_count} pangram(s). Skipping insertion.")
        return
    
    # Insert each pangram
    for i, pangram in enumerate(pangrams, 1):
        con.execute("""
            INSERT INTO pangrams (text) VALUES (?)
        """, [pangram])
        print(f"Inserted pangram {i}")
    
    # Verify insertion
    results = con.execute("SELECT id, LEFT(text, 50) || '...' as preview FROM pangrams").fetchall()
    print(f"\nSuccessfully inserted {len(results)} pangrams:")
    for row in results:
        print(f"  ID {row[0]}: {row[1]}")

def verify_database_state(con):
    """Print current database state for verification"""
    print("\n=== Final Database State ===")
    
    # Count speakers
    speaker_count = con.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]
    print(f"Total speakers: {speaker_count}")
    
    # Count pangrams
    pangram_count = con.execute("SELECT COUNT(*) FROM pangrams").fetchone()[0]
    print(f"Total pangrams: {pangram_count}")
    
    # Show speakers with their pangram arrays
    print("\nSpeakers with pangram tracking:")
    results = con.execute("""
        SELECT uid, firstname, surname, pangrams 
        FROM speakers 
        ORDER BY uid
    """).fetchall()
    
    for row in results:
        pangrams_str = str(row[3]) if row[3] else "[]"
        print(f"  UID {row[0]}: {row[1]} {row[2]} - Recited: {pangrams_str}")

def main():
    """Main execution function"""
    print("=" * 60)
    print("DuckDB Database Update Script")
    print("=" * 60)
    
    # Connect to database
    print(f"\nConnecting to database: {DUCKDB_PATH}")
    con = duckdb.connect(DUCKDB_PATH)
    
    try:
        # Execute all updates
        update_speaker_names(con)
        create_pangrams_table(con)
        add_pangrams_column_to_speakers(con)
        insert_initial_pangrams(con)
        verify_database_state(con)
        
        print("\n" + "=" * 60)
        print("Database update completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during database update: {e}")
        raise
    
    finally:
        con.close()
        print("\nDatabase connection closed")

if __name__ == "__main__":
    main()
