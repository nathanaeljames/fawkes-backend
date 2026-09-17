# Fawkes Wishlist v2
Updated 2026-09-15. Supersedes the original wishlist. Every feature mentioned across all conversations to date, prioritized by emphasis and mapped to the implementation phase where it lands. Phases refer to Fawkes_Implementation_Plan.md.

## Tier 1 — Core commitments (Phases 1-2, high priority)

- LLM + hierarchical state machine (statechart) dialogue core replacing Rasa; the LLM proposes, a Python validator ratifies. Phase 1 builds the machinery in text mode; Phase 2 recreates the Rasa workflows over voice and retires Rasa. [Phase 1-2]
- Input routing doctrine (three channels). Voice: all ASR text goes to the 4B router, which discards noise, advances FSM workflows (rendering tiny in-workflow replies itself), or escalates everything else to the voice-slot model; nothing from ASR reaches the research pipeline except through the voice pipeline. Text (web/app): all text goes to the 4B acting as arbiter, which dispatches to the research pipeline immediately, queues it for the next loop boundary, or interrupts a running task — never discards; text may also trigger an FSM command. Rubber-duck powers: the voice-slot model may answer, start a background research task, write to the blackboard, queue an injection, or interrupt research. [Phase 1-3]
- Small-model router (Qwen3-4B-Instruct class) as FSM front line and arbiter; upgrade to a 14B-class utility model only if the 4B fails the routing eval slice. [Phase 1-2]
- MemoryStore on Postgres + pgvector + full-text search: verbatim timestamped transcripts (every turn, both roles), documents, bi-temporal facts table. [Phase 1]
- Single `ingest()` write path: content hashes, idempotency, provenance stamps, fan-out to derived stores. [Phase 1]
- Hand-written versioned ontology (entity types, relation types, facet taxonomy: fact / preference / decision / event / task-state) enforced at runtime by enum tables and constraints; evolved only through versioned migrations. [Phase 1]
- Evaluation harness from day one: ~30 seeded questions across voice recall, research, coding; behavioral checks (valid tool calls, no loops, facts maintained, clean stops); latency timing at the API boundary; per-model capability cards derived from segmented results. Runs nightly/on-demand and gates phase exits. [Phase 1]
- Continuous-integration functional test suite on every commit (GitHub Actions): unit tests for the deterministic core, integration tests against Postgres, full FSM lifecycle walkthroughs driven by a shared mock-LLM fixture; every phase exit test codified as a permanent regression marker. [Phase 1]
- Structured observability with token and cost accounting: one JSON trace record per turn/call (stage timings, retrieval results, FSM state, model, tokens in/out/cached, external cost, task id); per-task rollups flag expensive work for review. [Phase 1]
- vLLM as the serving runtime now, behind the OpenAI-compatible `/v1/chat/completions` boundary; Qwen3.8-27B (Int4/AWQ-class quant) as primary model; 4B router via llama.cpp on the same card; embedder on CPU until the second GPU. [Phase 1]
- Multi-user and multi-endpoint support at every stage (hard requirement, not a feature). [All phases]
- Tools bound to pipeline slots, not to models: the tool registry, MemoryStore access, transcript access, and web search are available to whatever model occupies the voice or research slot on a given day. [All phases]
- Tiered voice-path context loading: Layer 0 persona (prompt-cached), Layer 1 per-user standing context on ECAPA resolution, Layer 2 per-turn topic pre-retrieval, Layer 3 tool-called deep search. [Phase 2]
- Sub-second voice-path retrieval: deterministic SQL fact lookup, semantic cache, hybrid BM25+vector with Reciprocal Rank Fusion; no extra LLM round-trips on the hot path. [Phase 2]
- Deferred, risk-based voice authentication: unidentified → voice-identified → authenticated tiers; recency-modulated ECAPA thresholds; passphrase FSM; retroactive attachment of buffered turns on identification. [Phase 2]
- Enrollment and voice-clone FSMs ported to the new statechart. [Phase 2]
- Operational data migration: iteration-1 DuckDB tables (speakers with ECAPA/XTTS imprints, pangrams, passages) migrated into Postgres; DuckDB retired completely; the in-memory ECAPA matrix is built from Postgres at startup. [Phase 2]
- Timestamps (date and time) on every turn, fact, resource, and modification, stored as UTC `timestamptz`; each device/session registers an IANA time zone; the prompt builder renders local time and elapsed time per session. Time is not optional. [All phases]
- Everyday question answering and single-shot tools (reminders, fact storage/recall, weather, timers). [Phase 2]
- Stall-word / provisional-answer support in the voice register, with a correction-event hook for background verification results. [Phase 2]
- Sandboxed (Docker) tool execution; no agent access to project root. [Phase 2]
- Per-turn memory-promotion hook: after each turn a small model judges whether anything is a durable fact/preference/decision worth promoting to standing context and the facts table. [Phase 2]
- Interim single-GPU fallback: FSM voice flows run on the 4B router alone (iteration-1 parity) whenever the 27B cannot be co-resident with the speech stack. [Phase 2]

## Tier 2 — Research, coding, knowledge, external hands (Phases 3-4)

- Research pipeline: agentic loop (grep/FTS/tree/web tools), CRAG-style relevance grading, web-search fallback, multi-hop reasoning, parallel fan-out for breadth-first tasks; job checkpoints for crash recovery. [Phase 3]
- OpenCode adopted as the coding surface, pointed at the local vLLM endpoint, transcripts flowing back into MemoryStore. (DeepSeek Harness on watchlist as alternative.) [Phase 3]
- Code tools for the research pipeline: ripgrep + fuzzy + tree-sitter tiers; an adopted code-knowledge-graph MCP server (code-review-graph or graphify class) updated incrementally per commit, consumed by OpenCode and the research pipeline. [Phase 3-4]
- MemoryStore exposed as an MCP (Model Context Protocol) server with scoped read/write auth and audit log, consumable by OpenCode, Claude, and other agent hosts. [Phase 3]
- External API bridge (Claude and other remote models): local-first routing policy (escalate only on low confidence, high complexity, criticality, or explicit request); remote-model tier selection; OCR-first payloads (Fawkes extracts text/JSON/Markdown locally and forwards text, never PDFs or screenshots, unless visual judgment is required); budget gate; per-call token/cost accounting; audit log. [Phase 3]
- Ingestion source adapters: browser text extraction sidecar (DevTools-protocol/content-script → `ingest()`), terminal capture (stdout/tmux), file drops — replacing screenshot-to-PDF workflows. [Phase 3]
- Ingestion router: document structure scoring; route to PageIndex-style trees, hybrid search, or lazy-only handling; OCR via Qwen vision at ingest. [Phase 3]
- Document-selection layer: per-document summaries, BM25+vector searchable. [Phase 3]
- Recall ladder: standing context → hybrid search → rerank → context expansion to neighboring turns → synthesized answer with citations, or an honest "not found." [Phase 3]
- Rubber-duck dual-pipeline interplay v1, built against the single served Qwen: shared blackboard in MemoryStore; notes-up; read-down via a status board (research summary refreshed at every loop boundary, injected into voice Layer 1) plus deep access to the full research transcript through voice Layer 3 tools; imperative channel; arbiter triage into discard (voice channel only) / dispatch / queue / interrupt with a ~5 s default-to-queue timeout; research model performs its own resteer. [Phase 3]
- Background consolidation and compaction: summaries-with-manifests, archival with stub pointers, never on the hot path; idle-time scheduling. [Phase 3]
- Wiki distillation layer, OKF-conformant (Open Knowledge Format bundles), one per project, lifecycle states, lint cron. [Phase 4]
- Project tunnels with full lifecycle: autonomous proposal (attributed), explicit-command crossing, reinforcement on use, decay on veto, destruction on request. [Phase 3-4]
- Ingest-time contradiction detection against the facts table; severity-gated: log, or trigger a clarification FSM. [Phase 3-4]
- Salience/importance weights: explicit marking, recency, retrieval frequency, goal linkage; component scores stored, decay over time. [Phase 3]
- Reconciliation crons (store agreement sampling) and wiki lint (orphans, staleness, gaps). [Phase 4]
- Gemma 4 as the voice-slot model alongside Qwen (arrives with second RTX 3090): conversational register plus decorrelated judgment; listening tests decide. [Phase 4]
- Panel mode (decorrelated judgment): identical-spec parallel execution by two models with an arbiter strategy (judge-selects, vote, synthesize, test-harness-wins), and decision-point consensus where only extracted key decisions are cross-checked; reserved for complex, critical, or sensitive tasks. [Phase 4]
- Heads-down mode: a task-mode switch that swaps the conversational model's GPU slot for a second coding model (GLM-class, size permitting) or a retrieval model (Harness-1), with the voice pipeline falling back to a Qwen-driven register. [Phase 4+]
- Autonomous research tasks (deep-research style) with sub-task decomposition, parallel execution, and synthesis reports. [Phase 4]
- Benchmark experiments: Cog-RAG vs wiki on research corpus; hand-rolled loop vs deepagents; Harness-1 as retrieval subagent; Qwen-vs-Gemma voice register; code-graph MCP vs grep-only code tools. [Phase 4]

## Tier 3 — Interfaces and reach (Phase 5)

- Web interface: interactive chat, file upload, project creation/management, notes and instructions, authentication and sessions; optional model-target dropdown. [Phase 5]
- Voice-driven project management ("create a new project called X; remember Y"). [Phase 5]
- Back-catalog import: exported Claude/ChatGPT history ingested through `ingest()` with verbatim transcripts, per-day summaries, embeddings, and project assignment — after project scopes exist so structure is preserved. (No iteration-1 conversation logs exist.) [Phase 5]
- Mobile/watch/car clients; multi-room speaker+mic deployment; possible dedicated device. [Phase 5+]
- Claude project-context sync for handoffs (via API context packaging; claude.ai Projects have no public API). [Phase 5]
- Row-level-security scoping: per-user, per-project memory visibility for additional users/team members. [Phase 5]

## Tier 4 — Eventual / far future (Phase 6+)

- SSM/Mamba ASR replacement for the interim (FastConformer) and final (Canary-Qwen) slots — likely pulled forward to roughly Phase 3.5 for VRAM relief before the second GPU.
- Screen-context subsystem: accessibility-API-first screen reading (Windows UI Automation / macOS AX / Linux AT-SPI) with event-triggered, perceptual-hash-deduplicated OCR fallback for un-instrumented apps; foundation for a heads-up display.
- Prompt/context compression for external traffic (headroom-class proxy, cache-aligned live-zone compression) — only if external token spend proves material after structured-extraction payloads.
- Microsoft To Do integration; Google Maps/Waze routing with LLM-improved recommendations; home lighting/automation.
- Newsletter/email ingestion and scheduled site scraping → goal-cross-referenced digests and verbal notifications.
- Computer vision pipeline: object recognition/tracking, garage-door and doorway reminders.
- vLLM scale-out / second-opinion serving as hardware grows; Graphiti temporal knowledge graph if bi-temporal SQL hits its ceiling.
- Power management: idle GPU power limits, scheduled indexing windows, standby modes (24/7 availability preserved).
- Fine-tuning experiments (voice register, retrieval policies) only if prompting demonstrably falls short.

## Separate tracks (own repositories, own milestone zero; not scheduled inside Fawkes)

- Cocktail Party speaker-separation research; results folded into Fawkes speech later.
- KVM/HID computer control: HDMI-capture plus USB-HID injection so Fawkes can observe and operate machines with no AI tooling installed (IP-KVM devices first; output-rate control over capture-rate; event-driven agent loop). Comparable in scope to Iteration 2; held separate until Fawkes Phase 3 exists.

## Explicitly rejected (see Fawkes_Toolbox.md for reasons)

- Summary-first memory (summaries replacing verbatim text) — never.
- Closed-set intent classification as a conversation gate (the Rasa failure mode).
- Hand-built coding harness on the critical path.
- LangChain/LangGraph/deepagents in the voice core.
- Reasoning-driven retrieval (tree traversal, multi-hop, CRAG loops) inside the voice hot path.
- Canary-Qwen LLM mode as the router (scratched 2026-09-06).
- Routing ASR text directly to the research pipeline; discarding text-channel messages.
