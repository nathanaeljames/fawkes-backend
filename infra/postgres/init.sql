-- Runs once, on first start of an empty data volume. Schema itself lives in migrations/.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
