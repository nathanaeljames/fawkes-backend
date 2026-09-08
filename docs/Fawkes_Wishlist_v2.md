# Fawkes Wishlist v2
Updated 2026-09-06. Supersedes the original wishlist. Every feature mentioned across all conversations to date, prioritized by emphasis and mapped to the implementation phase where it lands. Phases refer to Fawkes_Implementation_Plan.md.

## Tier 1 — Core commitments (Phases 1-2, high priority)

- LLM + hierarchical state machine (statechart) dialogue core replacing Rasa; the LLM proposes, a Python validator ratifies. Phase 1 builds the machinery in text mode; Phase 2 recreates the Rasa workflows over voice and retires Rasa. [Phase 1-2]
- Small-model router as FSM front line: intent recognition, slot capture, and noise/backchannel gating on a 4B-class model with confidence-based escalation to the 27B. Escalate to a larger utility model only if the 4B fails the routing eval slice. [Phase 1-2]
- MemoryStore on Postgres + pgvector + full-text search: verbatim timestamped transcripts (every turn, both roles), documents, bi-temporal facts table. [Phase 1]
- Single `ingest()` write path: content hashes, idempotency, provenance stamps. [Phase 1]
- Hand-written versioned ontology (entity types, relation types, facet taxonomy: fact / preference / decision / event / task-state) enforced at runtime by enum tables and constraints; evolved only through versioned migrations. [Phase 1]
- Evaluation harness from day one: ~30 seeded questions across voice recall, research, coding; behavioral checks (valid tool calls, no loops, facts maintained, clean stops); latency timing at the API boundary. Runs nightly/on-demand on the workstation and gates phase exits. [Phase 1]
- Continuous-integration functional test suite on every commit (GitHub Actions): unit tests for the deterministic core, integration tests against Postgres, and full FSM lifecycle walkthroughs (enrollment, voice clone, auth) driven by a shared mock-LLM fixture; every phase exit test codified as a permanent regression marker. [Phase 1]
- Structured observability: one JSON trace record per turn (stage timings, retrieval results, FSM state, model, tokens). [Phase 1]
- vLLM as the serving runtime now, behind the OpenAI-compatible `/v1/chat/completions` boundary; Qwen3.8-27B (Int4/AWQ-class quant) as primary model on the RTX 3090; 4B router + embedding model on the GTX 1660 Super via llama.cpp until the second 3090 arrives. [Phase 1]
- Multi-user and multi-endpoint support at every stage (hard requirement, not a feature). [All phases]
- Tools bound to pipeline slots, not to models: the tool registry, MemoryStore access, transcript access, and web search are available to whatever model occupies the voice or research slot on a given day (Qwen, Gemma, or another). [All phases]
- Tiered voice-path context loading: Layer 0 persona (prompt-cached), Layer 1 per-user standing context on ECAPA resolution, Layer 2 per-turn topic pre-retrieval, Layer 3 tool-called deep search. [Phase 2]
- Sub-second voice-path retrieval: deterministic SQL fact lookup, semantic cache, hybrid BM25+vector with Reciprocal Rank Fusion; no extra LLM round-trips on the hot path. [Phase 2]
- Deferred, risk-based voice authentication: unidentified → voice-identified → authenticated tiers; recency-modulated ECAPA thresholds; passphrase FSM; retroactive attachment of buffered turns on identification. [Phase 2]
- Enrollment and voice-clone FSMs ported to the new statechart. [Phase 2]
- Operational data migration: iteration-1 DuckDB tables (speakers with ECAPA/XTTS imprints, pangrams, passages) migrated into Postgres; DuckDB retired. [Phase 2]
- Timestamps on every turn, every fact, every resource, every modification, and the prompt builder injects current time plus elapsed time since the previous turn. Time is not optional. [All phases]
- Everyday question answering and single-shot tools (reminders, fact storage/recall, weather, timers). [Phase 2]
- Stall-word / provisional-answer support in the voice register, with a correction-event hook for background verification results. [Phase 2]
- Sandboxed (Docker) tool execution; no agent access to project root. [Phase 2]
- Per-turn memory-promotion hook: after each turn a small model judges whether anything is a durable fact/preference/decision worth promoting to standing context and the facts table. [Phase 2]
- Interim single-GPU fallback: FSM voice flows run on the 4B router alone (iteration-1 parity) whenever the 27B cannot be co-resident with the speech stack. [Phase 2]

## Tier 2 — Research, coding, knowledge (Phases 3-4)

- Research pipeline: agentic loop (grep/FTS/tree/web tools), CRAG-style relevance grading, web-search fallback, multi-hop reasoning, parallel fan-out for breadth-first tasks. [Phase 3]
- OpenCode adopted as the coding surface, pointed at the local vLLM endpoint, transcripts flowing back into MemoryStore. (DeepSeek Harness on watchlist as alternative.) [Phase 3]
- MemoryStore exposed as an MCP (Model Context Protocol) server with scoped read/write auth and audit log, consumable by OpenCode, Claude, and other agent hosts. [Phase 3]
- Claude escalation tool: budget-gated Anthropic API (API-key) calls for low-confidence or very complex tasks. [Phase 3]
- Rubber-duck dual-pipeline interplay v1, built against the single served Qwen: shared blackboard in MemoryStore; notes-up from voice to research; read-down via a status board (research summary refreshed at every loop boundary, injected into voice Layer 1) plus deep access to the full research transcript through voice Layer 3 tools; imperative channel; arbiter (4B router) triaging mid-flight input into discard / queue / interrupt; research model performs its own resteer. [Phase 3]
- Ingestion router: document structure scoring; route to PageIndex-style trees, hybrid search, or lazy-only handling; OCR via Qwen vision at ingest. [Phase 3]
- Document-selection layer: per-document summaries, BM25+vector searchable. [Phase 3]
- Recall ladder: standing context → hybrid search → rerank → context expansion to neighboring turns → synthesized answer with citations, or an honest "not found." [Phase 3]
- Background consolidation and compaction: summaries-with-manifests, archival with stub pointers, never on the hot path; idle-time scheduling. [Phase 3]
- Robust context/compaction pipeline for long sessions (Claude-Code-class); long research jobs checkpoint plan and progress to MemoryStore for crash recovery. [Phase 3]
- Wiki distillation layer, OKF-conformant (Open Knowledge Format bundles: markdown + YAML frontmatter + index.md), one per project, lifecycle states (draft/verified/stale/archived), lint cron. [Phase 4]
- Project tunnels with full lifecycle: autonomous proposal (attributed), explicit-command crossing, reinforcement on use, decay on veto, destruction on request; Layer-1 standing-context matches as tunnel triggers. [Phase 3-4]
- Ingest-time contradiction detection against the facts table; severity-gated: log, or trigger a clarification FSM over TTS/ASR. [Phase 3-4]
- Salience/importance weights: explicit marking, recency, retrieval frequency, goal linkage; component scores stored, decay over time. [Phase 3]
- Reconciliation crons (store agreement sampling) and wiki lint (orphans, staleness, gaps). [Phase 4]
- Gemma 4 as conversational/second-opinion model alongside Qwen (arrives with second RTX 3090): conversational register plus decorrelated judgment; listening tests decide. [Phase 4]
- Heads-down mode: a task-mode switch that swaps the conversational model's GPU slot for a second coding model (GLM-class, size permitting) or a retrieval model (Harness-1), with the voice pipeline falling back to a Qwen-driven register. [Phase 4+]
- Autonomous research tasks (deep-research style) with sub-task decomposition, parallel execution, and synthesis reports. [Phase 4]
- Benchmark experiments: Cog-RAG vs wiki on research corpus; hand-rolled loop vs deepagents; Harness-1 as retrieval subagent; Qwen-vs-Gemma voice register. [Phase 4]

## Tier 3 — Interfaces and reach (Phase 5)

- Web interface: interactive chat, file upload, project creation/management, notes and instructions, authentication and sessions. [Phase 5]
- Voice-driven project management ("create a new project called X; remember Y"). [Phase 5]
- Back-catalog import: iteration-1 conversation logs and exported Claude/ChatGPT history ingested through `ingest()` with verbatim transcripts, per-day summaries, embeddings, and project assignment — after project scopes exist so structure is preserved. [Phase 5]
- Mobile/watch/car clients; multi-room speaker+mic deployment; possible dedicated device. [Phase 5+]
- Claude project-context sync for handoffs (via API context packaging; claude.ai Projects have no public API). [Phase 5]
- Row-level-security scoping: per-user, per-project memory visibility for additional users/team members. [Phase 5]

## Tier 4 — Eventual / far future (Phase 6+)

- SSM/Mamba ASR replacement for the interim (FastConformer) and final (Canary-Qwen) slots — likely pulled forward to roughly Phase 3.5 for VRAM relief before the second GPU.
- Cocktail Party speaker-separation research: separate track and repository; results folded into Fawkes speech later.
- Microsoft To Do integration; Google Maps/Waze routing with LLM-improved recommendations; home lighting/automation.
- Newsletter/email ingestion and scheduled site scraping → goal-cross-referenced digests and verbal notifications.
- Computer vision pipeline: object recognition/tracking, garage-door and doorway reminders (separate real-time CV stack; vision-LLM describes, CV pipeline watches).
- vLLM scale-out / second-opinion serving as hardware grows; Graphiti temporal knowledge graph if bi-temporal SQL hits its ceiling.
- Power management: idle GPU power limits, scheduled indexing windows, standby modes (24/7 availability preserved).
- Fine-tuning experiments (voice register, retrieval policies) only if prompting demonstrably falls short.

## Explicitly rejected (see Fawkes_Toolbox.md for reasons)

- Summary-first memory (summaries replacing verbatim text) — never.
- Closed-set intent classification as a conversation gate (the Rasa failure mode).
- Hand-built coding harness on the critical path.
- LangChain/LangGraph/deepagents in the voice core.
- Reasoning-driven retrieval (tree traversal, multi-hop, CRAG loops) inside the voice hot path.
- Canary-Qwen LLM mode as the router (scratched 2026-09-06).
