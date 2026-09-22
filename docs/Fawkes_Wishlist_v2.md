# Fawkes Wishlist v2
Updated 2026-09-20. Supersedes the original wishlist. Every feature mentioned across all conversations to date, prioritized by emphasis and mapped to the implementation phase where it lands. Phases refer to Fawkes_Implementation_Plan.md. Not every item ships in its nominal phase; the standing rule is that infrastructure never forecloses an item (schema first, analytics later).

## Tier 1 — Core commitments (Phases 0-2, high priority)

- LLM + hierarchical state machine (statechart) dialogue core replacing Rasa; the LLM proposes, a Python validator ratifies, and a rejected proposal returns to its proposer with a constraint hint a bounded number of times. [Phase 1-2]
- Silo layer, Tier A, from migration 001: a `silos` table with one seeded default, non-null `silo_id` on every content and derived row, projects keyed `(silo_id, project_id)`, `silo_id` in every unique constraint and index prefix, tunnels and blocked pairs unable to represent a cross-silo link, content hash unique per silo, semantic cache and traces keyed by silo, an immutable silo context on every MemoryStore call, permanent leakage tests. Identity lives outside silos with a membership table and a default silo per user. Tier B (partitioning by silo, silo-keyed row-level security, per-silo roles, instance pin) waits for the first real second silo. Fawkes runs one silo; the schema is ready for many-per-database or one-per-database. [Phase 0-1; Tier B Phase 5+]
- Input routing doctrine (three channels). Voice: all ASR text goes to the 4B router, which discards noise, advances FSM workflows, or escalates everything else to the voice-slot model; the voice-slot model may also propose FSM transitions that need conversational context or that the router missed. Text: all text goes to the 4B acting as arbiter — start, queue, interrupt, or FSM command — never discards; illegal commands return to the arbiter. Rubber-duck powers: the voice-slot model may answer, write a blackboard note, or issue start/queue/interrupt imperatives straight to the research command queue. Response modality follows input modality. [Phase 1-3]
- Small-model router (Qwen3-4B-Instruct class) as FSM front line and arbiter; upgrade only if the 4B fails the routing eval slice. [Phase 1-2]
- MemoryStore on Postgres + pgvector + full-text search: verbatim timestamped transcripts, documents and document trees, bi-temporal facts table. [Phase 1]
- Single `ingest()` write path: content hashes, idempotency, provenance, silo/project stamps, fan-out to derived stores; turns, traces, and model-call cost rows included. [Phase 1]
- Hand-written versioned ontology enforced at runtime by enum tables and constraints. [Phase 1]
- Evaluation harness from day one with deterministic scoring: labeled routing verdicts, seeded recall facts with expected citations, fixed-corpus multi-hop citation ids, small failing-test repositories for coding-patch pass rate, schema validation, loop/stop/latency observation; a reference-anchored LLM judge only for free-text slices; runs on fingerprint change in idle windows plus weekly; real-usage corrections become new labeled cases; capability cards (versioned rows per model, quantization, serving config) read by the router and the external bridge; paid remote tiers scored once per release and spot-checked monthly on a capped sample. [Phase 1]
- Serving-recipe experiments measured, not assumed: stock vLLM vs tuned vLLM fork vs tuned llama.cpp fork. [Phase 1]
- Continuous-integration functional test suite on every commit: unit, integration, FSM lifecycle walkthroughs with a shared mock-LLM fixture, silo leakage tests, secret scanning; every phase exit test codified as a permanent regression marker. [Phase 1]
- Structured observability with token and cost accounting: one JSON trace per turn/call; per-task rollups flag expensive work. [Phase 1]
- vLLM as the serving runtime now, behind the OpenAI-compatible boundary; Qwen3.8-27B (Int4/AWQ-class) as primary model; 4B router via llama.cpp on the same card; embedder on CPU until the second GPU. [Phase 1]
- Multi-user and multi-endpoint support at every stage (hard requirement). [All phases]
- Tools bound to pipeline slots, not to models. [All phases]
- Tiered voice-path context loading: Layers 0-3. [Phase 2]
- Sub-second voice-path retrieval: deterministic SQL fact lookup, semantic cache, hybrid BM25+vector with Reciprocal Rank Fusion; no extra LLM round-trips on the hot path. [Phase 2]
- Deferred, risk-based voice authentication with passphrase FSM and retroactive turn attribution. [Phase 2]
- Enrollment and voice-clone FSMs ported to the new statechart. [Phase 2]
- Operational data migration: iteration-1 DuckDB tables into Postgres; DuckDB retired; ECAPA matrix built from Postgres at startup. [Phase 2]
- Timestamps on every turn, fact, resource, and modification, stored as UTC `timestamptz`; per-session IANA time zone; local time and elapsed time rendered per session. [All phases]
- Everyday question answering and single-shot tools. [Phase 2]
- Stall-word / provisional-answer support with a correction-event hook that delivers research results and self-corrections back by voice. [Phase 2]
- Sandboxed (Docker) tool execution. [Phase 2]
- Per-turn memory-promotion hook. [Phase 2]
- Interim single-GPU fallback: FSM voice flows on the 4B alone whenever the 27B cannot be co-resident. [Phase 2]

## Tier 2 — Research, coding, knowledge, external hands (Phases 3-4)

- Research pipeline: agentic loop, CRAG grading, web fallback, multi-hop, parallel fan-out (~4 medium-depth workers), job checkpoints, text responses to text clients. [Phase 3]
- OpenCode adopted as the coding surface; transcripts flowing back into MemoryStore. (DeepSeek Harness on watchlist.) [Phase 3]
- Code tools: ripgrep + fuzzy + tree-sitter tiers; adopted code-graph MCP servers used as shipped with their own storage — code-review-graph for blast radius and PR-grade impact, graphify for the multimodal bird's-eye view (docs, PDFs, images, community detection) — updated per commit. [Phase 3-4]
- MemoryStore exposed as an MCP server with scoped auth (silo, user, project) and audit. [Phase 3]
- External API bridge: local-first routing policy; tier selection from capability cards; OCR-first payloads; budget gate; cost accounting; silo-stamped audit. [Phase 3]
- Ingestion source adapters: file drops and coding-session transcripts. [Phase 3] (Browser and terminal adapters deferred — Tier 4.)
- Ingestion router with structure scoring and Qwen-vision OCR at ingest; document-selection layer; recall ladder. [Phase 3]
- Rubber-duck dual-pipeline interplay v1: blackboard (notes-up), imperative channel direct to the research command queue, read-down status board and deep transcript access, arbiter triage with ~5 s default-to-queue timeout; research model resteers itself. [Phase 3]
- Background consolidation and compaction with manifests, idle-time scheduling. [Phase 3]
- Wiki distillation layer, OKF-conformant, one per project, lifecycle states, lint cron. [Phase 4]
- Project tunnels with full lifecycle, within a silo only. [Phase 3-4]
- Ingest-time contradiction detection; salience weights; reconciliation crons. [Phase 3-4]
- Gemma 4 as the voice-slot model with the second RTX 3090, one model per card by default. [Phase 4]
- Panel mode (decorrelated judgment): identical-spec parallel execution with arbiter strategies, and decision-point consensus; available inside heads-down mode. [Phase 4]
- Heads-down mode: task-mode slot swap for a second coding model or Harness-1. [Phase 4+]
- Autonomous deep-research tasks with sub-task decomposition and synthesis reports. [Phase 4]
- Benchmark experiments: Cog-RAG vs wiki; hand-rolled loop vs deepagents; Harness-1; Qwen-vs-Gemma register; code-graph vs grep-only; tensor-parallel vs one-model-per-card. [Phase 4]

## Tier 3 — Interfaces and reach (Phase 5)

- Web interface with authentication, uploads, project management, optional model-target dropdown. [Phase 5]
- Voice-driven project management. [Phase 5]
- Back-catalog import of Claude/ChatGPT exports through `ingest()` with project assignment. (No iteration-1 conversation logs exist.) [Phase 5]
- Mobile/watch/car clients; multi-room deployment. [Phase 5+]
- Claude project-context sync via API context packaging. [Phase 5]
- Row-level security keyed to user for additional users; silo Tier B if a second silo is wanted. [Phase 5]

## Tier 4 — Eventual / far future (Phase 6+)

- SSM/Mamba ASR replacement — likely pulled forward to roughly Phase 3.5.
- Browser text-extraction sidecar and terminal capture adapters — deferred; the motivating goal (reading screens without installing anything on the host) belongs to the KVM/HID track.
- Screen-context subsystem; HUD foundation.
- Prompt/context compression for external traffic — deferred indefinitely on measured evidence.
- Microsoft To Do; Google Maps/Waze; home lighting/automation.
- Newsletter/email ingestion and scheduled scraping → digests.
- Computer vision pipeline.
- vLLM scale-out; Graphiti if bi-temporal SQL hits its ceiling.
- Power management; fine-tuning experiments only if prompting demonstrably falls short.

## Separate tracks (own repositories, own milestone zero)

- Cocktail Party speaker-separation research.
- KVM/HID computer control: HDMI-capture plus USB-HID injection; held separate until Fawkes Phase 3 exists.

## Explicitly rejected (see Fawkes_Toolbox.md for reasons)

- Summary-first memory — never.
- Closed-set intent classification as a conversation gate.
- Hand-built coding harness on the critical path.
- LangChain/LangGraph/deepagents in the voice core.
- Reasoning-driven retrieval inside the voice hot path.
- Canary-Qwen LLM mode as the router.
- Routing ASR text directly to the research pipeline; discarding text-channel messages; voice output for text input or text output for voice input.
- Tunnels, scope weights, caches, or prompts that cross a silo.
