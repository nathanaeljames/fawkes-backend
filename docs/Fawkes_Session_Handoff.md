# Fawkes Session Handoff
A living document for resuming work in any new conversation window. Update at the end of any session that changes state. Companion documents: Wishlist_v2, Guiding_Principles, Systems_Inventory, Implementation_Plan, Toolbox, Architecture_Specification, README_draft.

## Current state (as of 2026-09-15)

- **Iteration 1** (Rasa-based) complete: enrollment, speaker recognition, voice cloning, refactored `server03f.py` class architecture. No conversation logging exists in iteration 1. Keeps running in parallel through Phase 1; retired in Phase 2. Repo history is the resume artifact.
- **Iteration 2** (LLM + statechart) in final planning; **no code written yet**. Architecture settled; document set current; an Architecture Specification now exists for independent multi-model review (decorrelated judgment) before Phase 0.
- Hardware: one RTX 3090 (the GTX 1660 Super is gone); second RTX 3090 planned by end of year.
- Next action: independent review of the Architecture Specification, then **Phase 0 / Milestone Zero** — Postgres+pgvector Compose service, migration 001, MemoryStore skeleton, one green pytest. Say "go."

## Decisions log (major, most recent first)

- 2026-09-15: GTX 1660 Super removed from all layouts; Phase 1 runs on one 3090 (27B under vLLM at ~0.80 utilization + 4B under llama.cpp + CPU embedder), with a split-mode fallback. **Input routing doctrine** fixed: ASR text → 4B router → {ignore | FSM | escalate to voice-slot model}, never directly to research; text channels → 4B arbiter → {dispatch | queue | interrupt | FSM command}, never discard, ~5 s default-to-queue timeout; voice-slot model may start/queue/interrupt research and write the blackboard. Voice pipeline reads down (status board + deep transcript access); research receives only notes-up and imperatives. **External API bridge** (Phase 3): local-first routing policy, remote tier selection, OCR-first text/JSON/Markdown payloads, budget gate, cost accounting. **Token/cost accounting** added to the trace schema (Phase 1). **Panel mode** (decorrelated judgment, Phase 4) and **code-knowledge-graph MCP** (adopt at the edge, Phase 3-4) added. Back-catalog import = Claude/ChatGPT exports only (Phase 5); DuckDB retires completely after the Phase 2 migration. Timestamps: UTC `timestamptz`, per-session IANA time zone, local rendering. Screen-capture conversation harvested: ingestion source adapters (Phase 3), screen-context subsystem (Phase 6+), KVM/HID control as a separate track. Compression proxies deferred (measured savings marginal); structured-extraction payloads are the real lever. Diagram color semantics assigned; README diagram corrected (router drives the FSM; voice-slot model named; ingest() shown).
- 2026-09-06: Clean-slate ruling; operational DuckDB migration (Phase 2); Canary-Qwen-as-router scratched; rubber-duck interplay v1 on single served Qwen (Phase 3), Gemma Phase 4, heads-down mode (Phase 4+); tools bound to slots; time/elapsed-time injection; CI/testing strategy adopted (per-commit functional walkthroughs with a shared FakeLLM fixture; nightly GPU evals; test-first deterministic core); SSM ASR swap separated from Cocktail Party (~3.5 likely); Q5_K_S disavowed; arbiter protocol (asyncio inbox + pending counter + boundary condition + watcher cancel). Assistant failure recorded: documents described without delivery — corrected.
- 2026-08-31: Primary model **Qwen3.8-27B**; runtime **vLLM now**; small-model router fronts the FSM; Postgres SoR from v1; OpenCode adopted (Phase 3); DeepSeek Harness watchlisted; wiki **OKF-conformant**; Claude integration = API-key escalation + MemoryStore-MCP (Phase 3); standing rule: full revised docs with explanation at end of any changing turn. SimonScrapes review → memory-promotion hook, context expansion, RLS validation.
- Earlier: two-pipeline latency split is structural; verbatim-first; bi-temporal facts; ontology v1 before data; eval harness in Phase 1; tunnels lifecycle; deferred risk-based voice auth; no reasoning-retrieval on voice path; adopt-at-edges/build-the-core; monorepo (frontend + separate tracks split later); Adi Insights discredited; Harness-1 supersedes Context-1.

## Open questions (parked, mostly measurables)

- Qwen3.8 independent benchmark confirmation; Int4 checkpoint availability/quality for vLLM; single-card Phase 1 co-residency in practice.
- Gemma 4 vs tuned Qwen for voice register; value of decorrelated judgment in panel mode.
- Routing thresholds for the external API bridge (calibration of the confidence field against evals).
- Wiki vs Cog-RAG; hand-rolled loop vs deepagents; Harness-1; code-graph MCP vs grep-only.
- Reranker on the voice path: worth 30-100 ms?
- OpenCode vs DeepSeek Harness maturity at Phase-3 start; GLM-class coding model for heads-down mode.
- When bi-temporal SQL → Graphiti; when the SSM ASR swap (~3.5) and which checkpoints.

## Standing rules (recorded in assistant memory as well)

- Never proceed if a referenced file/URL/document is missing — stop and ask.
- Expand uncommon acronyms on first use.
- Multi-user support at every stage; never regresses.
- Any input meriting a change to project files returns the full revised file(s) in the same turn, with an explanation of exactly what was edited and why.
- Patch-style code edits with surrounding-context anchors; no unsolicited refactors; no emoji; Mermaid house colorway with the semantics recorded in the toolbox.

## How to resume a session

Paste or reference this document, name the phase you're in, and state the immediate goal. If code exists, attach the current file(s) being modified. If the session changes any decision, update the Decisions log here and regenerate affected companion documents.
