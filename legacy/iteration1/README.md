# Iteration 1 (frozen)

The Rasa-based voice assistant that preceded the LLM + statechart rebuild. It is kept runnable through Phase 1 of Iteration 2 and retired in Phase 2, when its enrollment and voice-cloning workflows are ported to the new core and its DuckDB tables migrate to Postgres.

- `canary/`: the voice server (`server03f.py`), its Dockerfile, the speaker-database tooling, and `iteration01/` with the frozen generations of earlier code and toolchain scripts.
- `rasa-nlp/`, `rasa-actions/`: the Rasa dialogue layer and custom actions.
- Workflow diagrams for these flows live in `docs/diagrams/iteration1/`.
- The experimental archive of earlier code generations is preserved at tag iteration-1-final under `canary/iteration01/archive/`.

Run it from the repository root with the same commands as before; the compose service names (`canary`, `rasa-nlp`, `rasa-actions`) are unchanged:

```bash
docker compose up -d canary rasa-nlp rasa-actions
docker compose exec canary python3 server03f.py
```

The server reads the speaker database from `/workspace-private/speakers/database.duckdb`, which is the `fawkes_private/speakers/` directory mounted from outside the repository.
