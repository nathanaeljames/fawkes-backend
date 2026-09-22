# Fawkes Wishlist v2
Updated 2026-09-19. Supersedes the original wishlist. Every feature mentioned across all conversations to date, prioritized by emphasis and mapped to the implementation phase where it lands. Phases refer to Fawkes_Implementation_Plan.md. Not every item ships in its nominal phase; the standing rule is that infrastructure never forecloses an item (schema first, analytics later).

## Tier 1 — Core commitments (Phases 0-2, high priority)

- LLM + hierarchical state machine (statechart) dialogue core replacing Rasa; the LLM proposes, a Python validator ratifies, and a rejected proposal returns to its proposer with a constraint hint a bounded number of times. Phase 1 builds the machinery in text mode; Phase 2 recreates the Rasa workflows over voice and retires Rasa. [Phase 1-2]
- Silo layer from migration 001: a `silos` table, every content and derived row stamped with a non-null `silo_id`, projects keyed `(silo_id, project_id)`, tunnels and blocked pairs unable to represent a cross-silo link, content hash unique per silo, derived tables list-partitioned by silo, semantic cache and traces keyed by silo, an immutable silo context on every MemoryStore call, forced row-level security under a non-owner role, and permanent leakage tests. Identity (speakers, imprints, devices) lives outside silos with a membership table and a default silo per user. Fawkes runs one silo; the schema is ready for many-per-database or one-per-database. [Phase 0-1]
- Input routing doctrine (three channels). Voice: all ASR text goes to the 4B router, which discards noise, advances FSM workflows (rendering tiny in-workflow replies itself), or escalates everything else to the voice-slot model; nothing from ASR reaches the research pipeline except through the voice pipeline. Text (web/app): all text goes to the 4B acting as arbiter, which starts a research task immediately, queues it for the next loop boundary, or interrupts a running task — never discards; text may also trigger an FSM command, and an illegal command returns to the arbiter. Rubber-duck powers: the voice-slot model may answer, write a blackboard note, or issue start/queue/interrupt imperatives straight to the research command queue. Response modality follows input modality. [Phase 1-3]
- Small-model router (Qwen3-4B-Instruct class) as FSM front line and arbiter; upgrade to a 14B-class utility model only if the 4B fails the routing eval slice. [Phase 1-2]
- MemoryStore on Postgres + pgvector + full-text search: verbatim timestamped transcripts (every turn, both roles), documents and document trees, bi-temporal facts table. [Phase 1]
- Single `ingest()` write path: content hashes, idempotency, provenance stamps, silo/project stamps, fan-out to derived stores; turns, traces, and model-call cost rows enter through it too. [Phase 1]
- Hand-written versioned ontology (entity types, relation types, facet taxonomy: fact / preference / decision / event / task-state) enforced at runtime by enum tables and constraints. [Phase 1]
- Evaluation harness from day one: ~30 seeded questions across voice recall, research, coding; behavioral checks; latency timing at the API boundary; per-model capability cards (versioned rows per model, quantization, and serving config with per-slice accuracy, latency percentiles, tool-call validity, loop and stop rates, cost per task) rewritten nightly and read by the router and the external bridge. [Phase 1]
- Serving-recipe experiments measured, not assumed: stock vLLM vs tuned vLLM fork (draft-model speculative decoding, requantized embeddings) vs tuned llama.cpp fork (long context), scored by the harness. [Phase 1]
- Continuous-integration functional test suite on every commit (GitHub Actions): unit tests for the deterministic core, integration tests against Postgres, full FSM lifecycle walkthroughs driven by a shared mock-LLM fixture, silo leakage tests, secret scanning; every phase exit test codified as a permanent regression marker. [Phase 1]
- Structured observability with token and cost accounting: one JSON trace record per turn/call (stage timings, retrieval results, FSM state, model, tokens in/out/cached, external cost, task id, silo); per-task rollups flag expensive work for review. [Phase 1]
- vLLM as the serving runtime now, behind the OpenAI-compatible `/v1/chat/completions` boundary; Qwen3.8-27B (Int4/AWQ-class quant) as primary model; 4B router via llama.cpp on the same card; embedder on CPU until the second GPU. [Phase 1]
- Multi-user and multi-endpoint support at every stage (hard requirement, not a feature). [All phases]
- Tools bound to pipeline slots, not to models. [All phases]
- Tiered voice-path context loading: Layer 0 persona (prompt-cached), Layer 1 per-user standing context on ECAPA resolution, Layer 2 per-turn topic pre-retrieval, Layer 3 tool-called deep search. [Phase 2]
- Sub-second voice-path retrieval: deterministic SQL fact lookup, semantic cache, hybrid BM25+vector with Reciprocal Rank Fusion; no extra LLM round-trips on the hot path. [Phase 2]
- Deferred, risk-based voice authentication: unidentified → voice-identified → authenticated tiers; recency-modulated ECAPA thresholds; passphrase FSM; retroactive attachment of buffered turns on identification. [Phase 2]
- Enrollment and voice-clone FSMs ported to the new statechart. [Phase 2]
- Operational data migration: iteration-1 DuckDB tables migrated into Postgres; DuckDB retired completely; the in-memory ECAPA matrix is built from Postgres at startup. [Phase 2]
- Timestamps (date and time) on every turn, fact, resource, and modification, stored as UTC `timestamptz`; each device/session registers an IANA time zone; the prompt builder renders local time and elapsed time per session. Time is not optional. [All phases]
- Everyday question answering and single-shot tools (reminders, fact storage/recall, weather, timers). [Phase 2]
- Stall-word / provisional-answer support in the voice register, with a correction-event hook that also delivers voice-initiated research results back by voice. [Phase 2]
- Sandboxed (Docker) tool execution; no agent access to project root. [Phase 2]
- Per-turn memory-promotion hook. [Phase 2]
- Interim single-GPU fallback: FSM voice flows on the 4B router alone whenever the 27B cannot be co-resident with the speech stack. [Phase 2]

## Tier 2 — Research, coding, knowledge, external hands (Phases 3-4)

- Research pipeline: agentic loop (grep/FTS/tree/web tools), CRAG-style relevance grading, web-search fallback, multi-hop reasoning, parallel fan-out for breadth-first tasks (~4 concurrent workers at medium depth as the design point); job checkpoints for crash recovery; text responses to text clients. [Phase 3]
- OpenCode adopted as the coding surface, pointed at the local vLLM endpoint, transcripts flowing back into MemoryStore. (DeepSeek Harness on watchlist as alternative.) [Phase 3]
- Code tools: ripgrep + fuzzy + tree-sitter tiers; an adopted code-knowledge-graph MCP server (code-review-graph or graphify class, used as shipped with its own storage) updated incrementally per commit, providing callers/callees, blast radius, and community structure to OpenCode and the research pipeline. [Phase 3-4]
- MemoryStore exposed as an MCP (Model Context Protocol) server with scoped read/write auth (silo, user, project) and audit log. [Phase 3]
- External API bridge (Claude and other remote models): local-first routing policy; remote-model tier selection from capability cards; OCR-first payloads (extracted text/JSON/Markdown, never PDFs or screenshots unless visual judgment is required); budget gate; per-call token/cost accounting; silo-stamped audit log. [Phase 3]
- Ingestion source adapters: file drops and coding-session transcripts. [Phase 3] (Browser text-extraction sidecar and terminal capture deferred — see Tier 4.)
- Ingestion router: document structure scoring; route to PageIndex-style trees (stored in Postgres), hybrid search, or lazy-only handling; OCR via Qwen vision at ingest. [Phase 3]
- Document-selection layer: per-document summaries, BM25+vector searchable. [Phase 3]
- Recall ladder: standing context → hybrid search → rerank → context expansion → synthesized answer with citations, or an honest "not found." [Phase 3]
- Rubber-duck dual-pipeline interplay v1 on the single served Qwen: shared blackboard (notes only); imperative channel (start/queue/interrupt) direct to the research command queue; read-down via a status board refreshed at every loop boundary and injected into voice Layer 1, plus deep transcript access through voice Layer 3; arbiter triage with a ~5 s default-to-queue timeout; research model performs its own resteer. [Phase 3]
- Background consolidation and compaction: summaries-with-manifests, archival with stub pointers, idle-time scheduling. [Phase 3]
- Wiki distillation layer, OKF-conformant, one per project (therefore per silo), lifecycle states, lint cron. [Phase 4]
- Project tunnels with full lifecycle, within a silo only. [Phase 3-4]
- Ingest-time contradiction detection; severity-gated: log, or clarification FSM. [Phase 3-4]
- Salience/importance weights with stored components and decay. [Phase 3]
- Reconciliation crons and wiki lint. [Phase 4]
- Gemma 4 as the voice-slot model alongside Qwen (arrives with second RTX 3090, one model per card by default): conversational register plus decorrelated judgment; listening tests decide. [Phase 4]
- Panel mode (decorrelated judgment): identical-spec parallel execution by two models with an arbiter strategy (judge-selects, vote, synthesize, test-harness-wins), and decision-point consensus; reserved for complex, critical, or sensitive tasks; available inside heads-down mode. [Phase 4]
- Heads-down mode: a task-mode switch that swaps the conversational model's GPU slot for a second coding model (GLM-class, size permitting) or a retrieval model (Harness-1), with the voice pipeline falling back to a Qwen-driven register. [Phase 4+]
- Autonomous research tasks (deep-research style) with sub-task decomposition, parallel execution, and synthesis reports. [Phase 4]
- Benchmark experiments: Cog-RAG vs wiki; hand-rolled loop vs deepagents; Harness-1 as retrieval subagent; Qwen-vs-Gemma voice register; code-graph MCP vs grep-only; tensor-parallel vs one-model-per-card. [Phase 4]

## Tier 3 — Interfaces and reach (Phase 5)

- Web interface: interactive chat, file upload, project creation/management, notes and instructions, authentication and sessions; optional model-target dropdown. [Phase 5]
- Voice-driven project management. [Phase 5]
- Back-catalog import: exported Claude/ChatGPT history through `ingest()` with verbatim transcripts, per-day summaries, embeddings, and project assignment. (No iteration-1 conversation logs exist.) [Phase 5]
- Mobile/watch/car clients; multi-room deployment; possible dedicated device. [Phase 5+]
- Claude project-context sync for handoffs (via API context packaging). [Phase 5]
- Row-level-security scoping for additional users; first non-default silo exercised end to end. [Phase 5]

## Tier 4 — Eventual / far future (Phase 6+)

- SSM/Mamba ASR replacement for the interim and final slots — likely pulled forward to roughly Phase 3.5.
- Browser text-extraction sidecar (DevTools protocol / content script) and terminal capture adapters — deferred; the goal that motivated them (reading screens without installing anything on the host) belongs to the KVM/HID track.
- Screen-context subsystem: accessibility-API-first screen reading with event-triggered OCR fallback; HUD foundation.
- Prompt/context compression for external traffic — deferred indefinitely on measured evidence (single-digit-percent real savings); revisit only if external spend becomes material.
- Microsoft To Do; Google Maps/Waze routing; home lighting/automation.
- Newsletter/email ingestion and scheduled site scraping → digests and verbal notifications.
- Computer vision pipeline: object recognition/tracking.
- vLLM scale-out; Graphiti temporal knowledge graph if bi-temporal SQL hits its ceiling.
- Power management: idle GPU power limits, scheduled indexing windows, standby modes.
- Fine-tuning experiments only if prompting demonstrably falls short.

## Separate tracks (own repositories, own milestone zero; not scheduled inside Fawkes)

- Cocktail Party speaker-separation research; results folded into Fawkes speech later.
- KVM/HID computer control: HDMI-capture plus USB-HID injection so Fawkes can observe and operate machines with no AI tooling installed (IP-KVM devices first; output-rate control over capture-rate; event-driven agent loop). Held separate until Fawkes Phase 3 exists.

## Explicitly rejected (see Fawkes_Toolbox.md for reasons)

- Summary-first memory (summaries replacing verbatim text) — never.
- Closed-set intent classification as a conversation gate (the Rasa failure mode).
- Hand-built coding harness on the critical path.
- LangChain/LangGraph/deepagents in the voice core.
- Reasoning-driven retrieval inside the voice hot path.
- Canary-Qwen LLM mode as the router.
- Routing ASR text directly to the research pipeline; discarding text-channel messages; voice output for text input or text output for voice input.
- Tunnels, scope weights, caches, or prompts that cross a silo.
