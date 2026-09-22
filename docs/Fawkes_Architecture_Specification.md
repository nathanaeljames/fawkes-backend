# Fawkes Architecture Specification
Version 1.1, 2026-09-19. A complete prose description of the envisioned final structure of Fawkes, written for machine and human review. No diagrams. Every component has an identifier (C-nn), every pathway is traced step by step (P-nn), and every invariant is stated (I-nn). Reviewers are asked to check internal consistency, missing components, unstated assumptions, and pathways that cannot work as described. Companion documents: Guiding_Principles, Systems_Inventory, Implementation_Plan, Toolbox, Wishlist_v2.

Changes from 1.0: silo layer added (R-11, C-32, D-08, P-15); response-modality invariant (I-11); unified verdict vocabulary; illegal-proposal handling defined (P-02 note); text-output pathway (P-14); capability-card process (C-29); GPU layout policy (H-03); identity placed outside silos (D-01).

## 0. Purpose and scope

Fawkes is a self-hosted, voice-native, multi-user AI assistant. It listens on any microphone or client, identifies the speaker by voice, remembers every conversation verbatim with timestamps, reasons over that memory with a hierarchical state machine driving local language models, and runs a separate research/coding pipeline alongside the voice loop. Everything runs on owned hardware; cloud calls are explicit, budget-gated, and logged.

## 1. Hard requirements

- R-01 Multi-user and multi-endpoint at every stage: memory is keyed to resolved speaker identity, never to a session.
- R-02 Verbatim, timestamped storage of every turn (both roles), every fact, every resource; time stored as UTC with time-zone-aware types; per-session time zone; elapsed time rendered into prompts.
- R-03 One ACID system of record (Postgres); all other stores derived and rebuildable.
- R-04 Two latency classes: the voice loop performs only deterministic lookups, cached answers, and single-shot hybrid retrieval; reasoning-driven retrieval belongs to the research pipeline.
- R-05 The LLM proposes; deterministic code ratifies every state transition.
- R-06 Tools, memory, transcripts, and web access are bound to pipeline slots, not to models.
- R-07 Every transformation writes a manifest; every retrieval writes a trace; every model call records tokens and cost.
- R-08 Answers about the past cite sources or admit ignorance.
- R-09 Outbound cloud payloads are extracted text and structured data unless visual judgment is the task.
- R-10 Nothing from the ASR channel reaches the research pipeline except through the voice pipeline; nothing from text channels is discarded.
- R-11 Every content row belongs to exactly one silo; silos are hard partitions that no tunnel, scope weight, cache, or prompt can cross; the schema supports many silos per database and one silo per database without code changes.

## 2. Model slots and hardware

- S-01 Router/arbiter slot: Qwen3-4B-Instruct class, served by llama.cpp. Roles: voice-channel routing, text-channel arbitration, mid-flight injection triage, slot extraction, memory-promotion judgments, background summarization. Upgrade path: 14B-class utility model if evals demand.
- S-02 Voice slot: the model that answers escalated voice turns. Qwen3.8-27B (thinking off or small budget) through Phase 3; Gemma 4 from Phase 4. Powers: answer; call single-shot tools; start a background research task; write a blackboard note; queue an injection into research; interrupt research.
- S-03 Research slot: Qwen3.8-27B (thinking high, thinking preservation), served by vLLM behind an OpenAI-compatible endpoint. Drives the research/coding pipeline; performs its own resteering.
- S-04 Embedding model: bge-m3 class (CPU until second GPU). Reranker: bge-reranker-v2-m3 class (optional).
- S-05 External models: Anthropic API models via the external API bridge (C-22); selected by tier per routing policy.
- S-06 Heads-down mode (Phase 4+): the voice-slot GPU allocation may be swapped for a second coding model or a retrieval specialist (Harness-1); the voice slot then falls back to a Qwen register.
- H-01 Phase 1 (one RTX 3090, text-mode): vLLM at ~0.80 utilization for the 27B Int4 plus KV cache; 4B under llama.cpp on the same card; embedder on CPU. Fallback: split mode, alternating loads per eval slice. Optimized serving recipes (tuned vLLM fork with a draft model; tuned llama.cpp fork for long context) are measured against stock vLLM by the eval harness, never assumed.
- H-02 Phase 2 (one card): speech stack + 4B resident; FSM voice flows on the 4B alone; 27B swapped in for open conversation when no speech session needs the stack.
- H-03 Phase 4 (two cards): default layout pins one model per card (card one = research slot; card two = speech stack + voice slot) for isolation and predictable latency. Tensor parallelism across both cards is an experiment permitted only with a verified NVLink bridge, x16 slots, and PSU transient headroom, and adopted only if the eval harness shows a single-stream or capacity gain that the voice pipeline's latency does not pay for.

## 3. Components

Each entry: responsibility; inputs; outputs; depends on; phase.

- C-01 Audio I/O and WebSocket layer. Streams audio per client; tracks device id and registered IANA time zone. Carry-over.
- C-02 VAD and endpointing (MarbleNet). Detects speech, finalizes utterances, drops most non-speech. Carry-over.
- C-03 ASR (interim FastConformer; final Canary-Qwen; SSM replacement ~Phase 3.5). Carry-over.
- C-04 Speaker identity (ECAPA-TDNN). In-memory embedding matrix built from Postgres at startup and updated on enrollment; per-second matching; emits user_id with confidence; supports deferred identification. Identity data lives outside the silo scheme (D-01). Carry-over; auth tiers Phase 2.
- C-05 Router/arbiter (S-01). Voice channel verdicts: {ignore | FSM event with slots | escalate}. Text channel verdicts: {start | queue | interrupt | FSM command}. Mid-flight injection verdicts: {discard (voice-origin only) | queue | interrupt}. Renders tiny in-workflow replies itself. Phase 1-3.
- C-06 FSM registry and validator. Statechart definitions with entry guards, states, slot schemas, legal transitions. The validator deterministically checks any proposed transition or slot update and, on rejection, returns a constraint hint to the component that proposed it (router, arbiter, or voice-slot model). Retries are bounded (default two); after the bound, the proposal is dropped with a logged trace and, on the voice channel, a clarifying reply. Phase 1-2.
- C-07 Prompt builder. Assembles state-conditioned prompts: Layer 0 persona and tool registry (byte-stable, prompt-cached), Layer 1 per-user standing context (including the research status board, refreshed per loop boundary, during research sessions), Layer 2 per-turn pre-retrieval, active FSM state and slot schemas, local time and elapsed time. Asserts every context item matches the request's silo before any model call. Phase 1.
- C-08 Structured-output contract. JSON schema: slot_updates, transition, tool_calls, response_text, confidence; constrained decoding. Only response_text reaches TTS. Phase 1.
- C-09 Tool registry and executor. Opaque slot-bound tools; Docker-sandboxed execution. Phase 2.
- C-10 MemoryStore. Postgres + pgvector + full-text search; tables: silos, projects, transcripts, facts (bi-temporal), documents, document_nodes, tasks, model_calls, manifests, traces, tunnels, blocked_pairs, capability_cards; row-level security keys (silo, user, project). Backend-agnostic interface; every method takes a silo context (C-32). Phase 1.
- C-11 Ontology and constraints. Versioned entity/relation/facet vocabulary enforced by enum tables, foreign keys, CHECK constraints; migrations per version. Phase 1.
- C-12 ingest(). Single write path: content hash (unique per silo), timestamps, provenance, silo and project stamps; idempotent fan-out handlers to FTS, vectors, document trees, wiki, facts extraction. Records turns, traces, and model-call cost rows as well as documents. Phase 1.
- C-13 Hybrid retrieval. BM25 + vector with Reciprocal Rank Fusion; scope (project/topic) weights inside the silo; optional rerank. Phase 1-2.
- C-14 Semantic cache. Query-similarity cache keyed by silo; invalidation on ingest. Phase 2.
- C-15 Memory-promotion hook. After each turn, C-05 judges durability; promotes to facts table and Layer 1. Phase 2.
- C-16 Correction/event bus and arbiter machinery. asyncio inbox, pending counter, boundary condition with timeout (~5 s default-to-queue), watcher task that cancels an in-flight generation on interrupt; carries background-verification results, research-task completions destined for the voice channel, and cross-pipeline imperatives. Phase 2 (hook), Phase 3 (arbiter).
- C-17 Ingestion source adapters and router. Adapters in Phase 3: file drops and coding-session transcripts. Adapters deferred to Phase 6+: browser text-extraction sidecar, terminal capture. Router: structure scoring → document trees (checksum-cached, stored in document_nodes) / hybrid-only / lazy-only; Qwen-vision OCR for image-bearing inputs. Phase 3.
- C-18 Research pipeline. Agentic loop over grep/FTS/tree/web/code tools; relevance grading; web fallback; multi-hop; parallel fan-out; checkpoints plan and progress to MemoryStore; status board refreshed at every loop boundary; returns text responses to text clients and correction events to the voice channel for voice-initiated tasks. Phase 3.
- C-19 Code tools and code knowledge graph. ripgrep / fuzzy / tree-sitter tiers; adopted code-graph MCP server (its own SQLite storage, rebuildable from the repository) updated per commit; serves OpenCode and C-18. Phase 3-4.
- C-20 Coding surface (OpenCode). Adopted harness on the local endpoint; transcripts ingested via C-12; consumes C-21. Phase 3.
- C-21 MCP server. Exposes MemoryStore recall/remember and selected tools to external agent hosts with scoped auth (silo, user, project) and audit. Phase 3.
- C-22 External API bridge. Routing policy (explicit request → criticality → local confidence cross-checked by cheap verification → complexity → budget); remote tier selection from capability cards; OCR-first payload builder (text/JSON/Markdown); budget gate; per-call token/cost accounting; audit log carrying the request's silo. Phase 3.
- C-23 Rubber-duck interplay. Shared blackboard (MemoryStore rows tagged to the active task and silo); notes-up (voice → research, filtered); read-down (status board into voice Layer 1; deep research-transcript access via voice Layer 3); imperative channel (start/queue/interrupt) delivered directly to the research pipeline's command queue, not via the blackboard. Phase 3 (single model), Phase 4 (Gemma).
- C-24 Consolidation, compaction, scheduler. Idle-window summaries with manifests, archival with stubs, research-job compaction; GPU budget arbitration. Phase 3.
- C-25 Recall ladder. Standing context → hybrid → rerank → context expansion (neighboring turns) → cited synthesis or honest gap. Phase 3.
- C-26 Salience weights, contradiction detector, tunnel manager. Component salience scores with decay; ingest-time fact-conflict checks; tunnels within a silo only, with propose/attribute/reinforce/decay/destroy. Phase 3-4.
- C-27 Wiki distillation layer (OKF-conformant). Per-project (therefore per-silo) compiled bundles with lifecycle states and lint. Phase 4.
- C-28 Panel mode. Parallel identical-spec execution or decision-point consensus across two models with an arbiter strategy (judge-selects, vote, synthesize, test-harness-wins); full fan-out recorded. Phase 4.
- C-29 Eval harness, CI, observability, capability cards. Nightly GPU evals over segmented slices (FSM routing, voice recall, research multi-hop, coding patch, structured-output validity, latency); per-commit functional walkthroughs with a shared FakeLLM fixture; one JSON trace per turn/call with tokens and cost; per-task rollups. A capability card is a versioned row per (model, quantization, serving config) holding per-slice accuracy, latency percentiles, tool-call validity rate, loop rate, stop-compliance rate, and cost per task; the nightly run re-scores the seed set plus newly labeled failures and writes a new card version; the router and the external bridge read the latest cards for thresholds and tier selection. Phase 1.
- C-30 Web interface and auth. Chat, uploads, project management, sessions, row-level security, optional model-target selector; back-catalog importer. Phase 5.
- C-31 TTS and stream manager (XTTS/Piper). Speaks response_text only. Carry-over.
- C-32 Silo context. An immutable value created once at request entry after identity and project are resolved (silo_id, project_id, user_id); required by every MemoryStore method; never a global; a request whose silo cannot be resolved is rejected and logged. Phase 0-1.

## 4. Data model (summary)

- D-01 Identity store, outside the silo scheme: Person (user_id, name, preferences facets), voice imprints, Device (device_id, time zone, endpoint type), user_silos membership with a default silo per user.
- D-02 Silo (silo_id). Project keyed by (silo_id, project_id); a project belongs to one silo for life. Every content and derived table carries a non-null silo_id and references projects through the composite pair. Vector and full-text tables are list-partitioned by silo_id (one default partition until a second silo exists). Row-level security is enabled and forced, keyed to a transaction-scoped setting under a non-owner application role. A deployment setting pins an instance to an allowed silo set.
- D-03 Turn: id, silo_id, project_id, user_id, device_id, role, text (verbatim), recorded_at (UTC), local_tz, model_call_id, trace_id.
- D-04 Fact: id, silo_id, subject, predicate, object, facet, valid_from, valid_to, recorded_at, source, salience components, superseded_by.
- D-05 Document, DocumentNode (tree nodes with summaries and spans, stored in Postgres), content hash unique per silo, provenance, structure score, index status.
- D-06 Task (research/coding job): silo_id, plan, status, checkpoints, status board, blackboard notes, cost rollup, originating channel (voice or text).
- D-07 ModelCall: model, slot, tokens in/out/cached, cost, duration, time-to-first-token, thinking setting, silo_id.
- D-08 Tunnels and blocked pairs: one silo_id column shared by both endpoints, so a cross-silo tunnel cannot be represented. Manifest and Trace rows carry silo_id.

## 5. Pathways

- P-01 Voice turn, no FSM active. C-01 → C-02 → C-03 → C-04 → C-32 (silo context resolved from user default or active project) → C-05. Ignore: turn persisted as noise via C-12, no response. FSM event: P-02. Escalate: C-07 builds prompt → S-02 generates structured output → if a transition is proposed, C-06 validates it → C-31 speaks response_text → C-12 persists turn, trace, and model-call cost → C-15 judges promotion.
- P-02 FSM workflow. Entry guard fires (utterance-driven via C-05, or signal-driven via C-04). C-05 handles subsequent turns; C-06 validates each transition; state and slots persist per user_id. An illegal proposal (for example, "advance to record_pangram" while the name slot is empty, or entering voice_clone while enrollment is active) returns a constraint hint to the proposer, which retries within the bound; after the bound, the proposal is dropped and traced. Exit persists results via C-12.
- P-03 Text-channel message. Message → C-32 → C-05 as arbiter. Start: create Task and start C-18. Queue: outbox for the next loop boundary. Interrupt: cancel event. FSM command: validated by C-06; illegal commands return to the arbiter, never to the voice slot. Never discarded. Response returns as text to the originating client (P-14).
- P-04 Research task lifecycle. C-18 plans; loops over tool calls (each call is a loop boundary); at each boundary: drain outbox, refresh status board, checkpoint to D-06, check pending counter (wait up to timeout). Retrieval via C-13/C-25/C-19; escalation via C-22; completion writes cited synthesis and cost rollup; delivery per P-14 or P-05.
- P-05 Voice-initiated research. During P-01, S-02 answers provisionally and starts a Task with originating channel = voice. Later voice turns may add notes (blackboard) or issue imperatives (start/queue/interrupt) delivered to C-18's command queue. On completion or a material update, C-18 raises a correction event through C-16, which the prompt builder injects into the next voice turn or a proactive interjection; the result is spoken, never rendered as text.
- P-06 Mid-flight injection and interrupt. Input arrives while C-18 is generating → inbox → pending counter increments → C-05 rules {discard (voice-origin only) | queue | interrupt}. Queue: outbox. Interrupt: cancel event → watcher cancels generation and aborts the vLLM request → partial output kept → injection appended → C-18 resumes and resteers itself. Boundary wait: if pending > 0 at a boundary, C-18 waits up to the timeout, then proceeds treating undecided items as queued.
- P-07 Read-down. C-18 refreshes the status board at every boundary; C-07 injects it into S-02's Layer 1 on every voice turn during an active research session; deep detail via Layer 3 tools reading the research transcript from MemoryStore.
- P-08 Ingestion. Source adapter → C-12 (hash per silo, dedupe, provenance, silo and project stamps) → rows → fan-out: FTS, vectors, C-17 structure scoring → tree build (idle-time) or lazy-only; C-15/C-26 fact extraction and contradiction check; wiki update queued (Phase 4).
- P-09 External escalation. C-22 policy triggers → payload builder produces extracted text/JSON/Markdown (local OCR via S-03 vision if needed) → silo assertion → budget gate → remote call → D-07 records tokens/cost → response returns into the calling pipeline with provenance marked external.
- P-10 Identity and startup. On boot: C-04 loads speaker rows into the matrix; C-07 warms Layer 0; capability cards loaded. On new client: device registers time zone. Identification tiers: unidentified, voice-identified, authenticated.
- P-11 Memory promotion and consolidation. After each turn C-15 may write facts and update Layer 1. Idle windows: C-24 summaries with manifests, archival with stubs, compaction; C-26 reconciliation sampling (Phase 4).
- P-12 Panel mode (Phase 4). Task flagged critical → both models (or Qwen self-fusion) run the identical spec or vote on extracted decisions → arbiter strategy → verdict and candidates recorded → result returns to the requesting pipeline.
- P-13 Heads-down mode (Phase 4+). Explicit mode switch → voice-slot model unloaded → second coding model or Harness-1 loaded → voice slot rebinds to a Qwen register → tools rebind by slot → mode exit reverses.
- P-14 Text response. Every text-channel task returns its result as text to the originating client (web/mobile session). Voice-initiated tasks never return text; they return through P-05. A voice question about a text-initiated task is answered by voice from the status board (P-07).
- P-15 Cross-silo movement. No in-place move. Export from the source silo, write a manifest, re-ingest into the destination silo as new rows with an opaque text pointer to the source; approval is a pluggable policy (the owner in Fawkes).

## 6. Interfaces and seams

- X-01 OpenAI-compatible `/v1/chat/completions`: the only way any pipeline talks to any model, local or remote.
- X-02 MemoryStore interface: the only way any component reads or writes memory; one backend adapter per database; every method takes a silo context.
- X-03 ingest(): the only write path into MemoryStore.
- X-04 MCP server: the only way external agent hosts reach Fawkes memory or tools.
- X-05 Tool registry: slot-bound; identical surface for any model in a slot.
- X-06 Event bus contract: inbox/outbox/pending/cancel primitives; swappable for a real message bus without changing callers.

## 7. Invariants

- I-01 Only response_text reaches TTS; reasoning never leaks to speech.
- I-02 No state mutates without validator approval; rejected proposals return to their proposer with a hint, bounded.
- I-03 No LLM round-trip inside voice retrieval.
- I-04 ASR text reaches research only via the voice pipeline; text-channel messages are never discarded.
- I-05 The arbiter triages; the research model resteers.
- I-06 Verbatim rows are never deleted or rewritten by summarization; manifests accompany every transformation.
- I-07 All timestamps are UTC time-zone-aware; rendering is per session.
- I-08 Every model call has a ModelCall row with tokens and cost.
- I-09 Derived stores may be dropped and rebuilt from the system of record at any time.
- I-10 A phase exit test never leaves the test suite.
- I-11 Response modality follows input modality: voice in, voice out; text in, text out; background results return through the channel that started the task.
- I-12 No row, cache entry, tunnel, prompt item, or outbound payload crosses a silo boundary; the permanent leakage tests (ingest in silo A, query as silo B through every path) gate every phase exit.

## 8. Known open questions for reviewers

- Q-01 Single-card co-residency of 27B + 4B + KV in Phase 1: is the memory arithmetic sound, and is split mode an adequate fallback?
- Q-02 Is the three-channel routing doctrine complete, or is there an input class with no defined path?
- Q-03 Does the arbiter protocol (inbox, pending counter, boundary wait, watcher cancel) have a race or deadlock the description misses?
- Q-04 Is the read-down design (status board + tool access) sufficient for the voice pipeline to discuss research intelligently without context bloat?
- Q-05 Are the routing-policy signals for external escalation adequate, and how should thresholds be calibrated from capability cards?
- Q-06 Is anything required by the hard requirements (R-01…R-11) unassigned to a component?
- Q-07 Does the silo layer (composite keys, partitioned derived tables, forced row-level security, identity outside silos) introduce a migration or ORM cost that outweighs its value at one silo, and is the identity placement (D-01) sound?
