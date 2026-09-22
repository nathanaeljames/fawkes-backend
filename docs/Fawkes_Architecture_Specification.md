# Fawkes Architecture Specification
Version 1.2, 2026-09-20. A complete prose description of the envisioned final structure of Fawkes, written for machine and human review. No diagrams (diagram sources at three complexity levels live beside this document). Every component has an identifier (C-nn), every pathway is traced step by step (P-nn), and every invariant is stated (I-nn). Companion documents: Guiding_Principles, Systems_Inventory, Implementation_Plan, Toolbox, Wishlist_v2.

Changes from 1.1: silo layer split into Tier A (migration 001) and Tier B (deferred to the first real second silo) in D-02 and C-32; capability-card measurement method specified in C-29; voice-slot FSM proposals clarified as a context-richer proposer, not only a safety net (C-05, P-02); blackboard and correction-event naming aligned with the diagrams (C-23, P-05).

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
- R-11 Every content row carries exactly one silo key; silos are hard partitions that no tunnel, scope weight, cache, or prompt may cross; the schema supports many silos per database and one silo per database without code changes. Enforcement machinery beyond the key (partitioning, silo-keyed row-level security) is added when a second silo exists.

## 2. Model slots and hardware

- S-01 Router/arbiter slot: Qwen3-4B-Instruct class, served by llama.cpp. Roles: voice-channel routing, text-channel arbitration, mid-flight injection triage, slot extraction, memory-promotion judgments, background summarization. Upgrade path: 14B-class utility model if evals demand.
- S-02 Voice slot: Qwen3.8-27B (thinking off or small budget) through Phase 3; Gemma 4 from Phase 4. Powers: answer; call single-shot tools; propose FSM transitions from conversational context; write a blackboard note; issue start/queue/interrupt imperatives to the research command queue.
- S-03 Research slot: Qwen3.8-27B (thinking high, thinking preservation), served by vLLM behind an OpenAI-compatible endpoint. Drives the research/coding pipeline; performs its own resteering.
- S-04 Embedding model: bge-m3 class (CPU until second GPU). Reranker: bge-reranker-v2-m3 class (optional).
- S-05 External models: Anthropic API models via the external API bridge (C-22); tier selected from capability cards.
- S-06 Heads-down mode (Phase 4+): the voice-slot GPU allocation may be swapped for a second coding model or a retrieval specialist; the voice slot falls back to a Qwen register; panel mode remains available.
- H-01 Phase 1 (one RTX 3090, text-mode): vLLM at ~0.80 utilization for the 27B Int4 plus KV cache; 4B under llama.cpp on the same card; embedder on CPU. Fallback: split mode. Serving recipes (tuned vLLM fork with a draft model; tuned llama.cpp fork for long context) are measured against stock vLLM by the eval harness.
- H-02 Phase 2 (one card): speech stack + 4B resident; FSM voice flows on the 4B alone; 27B swapped in for open conversation when no speech session needs the stack.
- H-03 Phase 4 (two cards): one model per card by default (card one = research slot; card two = speech stack + voice slot). Tensor parallelism is an experiment permitted only with a verified NVLink bridge, x16 slots, and PSU transient headroom, adopted only on measured gain that the voice pipeline's latency does not pay for.

## 3. Components

- C-01 Audio I/O and WebSocket layer. Streams audio per client; tracks device id and registered IANA time zone. Carry-over.
- C-02 VAD and endpointing (MarbleNet). Carry-over.
- C-03 ASR (interim FastConformer; final Canary-Qwen; SSM replacement ~Phase 3.5). Carry-over.
- C-04 Speaker identity (ECAPA-TDNN). In-memory matrix built from Postgres at startup and updated on enrollment; per-second matching; user_id with confidence; deferred identification tiers. Identity data lives outside the silo scheme. Carry-over; auth tiers Phase 2.
- C-05 Router/arbiter (S-01). Voice verdicts: {ignore | FSM event with slots | escalate}. Text verdicts: {start | queue | interrupt | FSM command}. Injection verdicts: {discard (voice-origin only) | queue | interrupt}. The router is the primary FSM driver and sees one utterance plus FSM state; the voice-slot model (S-02), holding Layers 0-3, may also propose transitions that need conversational context or that the router missed — the escalation valve that makes no misroute irreversible. Phase 1-3.
- C-06 FSM registry and validator. Statechart definitions with entry guards, states, slot schemas, legal transitions. The validator deterministically checks any proposed transition or slot update and, on rejection, returns a constraint hint to the component that proposed it (router, arbiter, or voice-slot model). Retries are bounded (default two); after the bound, the proposal is dropped with a logged trace and, on the voice channel, a clarifying reply. Phase 1-2.
- C-07 Prompt builder. Layer 0 persona and tool registry (byte-stable, prompt-cached), Layer 1 per-user standing context (including the research status board, refreshed per loop boundary), Layer 2 per-turn pre-retrieval, active FSM state and slot schemas, local time and elapsed time; asserts every context item matches the request's silo. Phase 1.
- C-08 Structured-output contract. JSON schema: slot_updates, transition, tool_calls, response_text, confidence; constrained decoding. Only response_text reaches TTS. Phase 1.
- C-09 Tool registry and executor. Opaque slot-bound tools; Docker-sandboxed execution. Phase 2.
- C-10 MemoryStore. Postgres + pgvector + full-text search; tables: silos, projects, user_silos, transcripts, facts (bi-temporal), documents, document_nodes, tasks, model_calls, manifests, traces, tunnels, blocked_pairs, capability_cards. Backend-agnostic interface; every method takes a silo context (C-32). Phase 1.
- C-11 Ontology and constraints. Versioned entity/relation/facet vocabulary enforced by enum tables, foreign keys, CHECK constraints. Phase 1.
- C-12 ingest(). Single write path: content hash (unique per silo), timestamps, provenance, silo and project stamps; idempotent fan-out to FTS, vectors, document trees, wiki queue, facts extraction; records turns, traces, and model-call cost rows as well as documents. Phase 1.
- C-13 Hybrid retrieval. BM25 + vector with Reciprocal Rank Fusion; scope weights inside the silo; optional rerank. Phase 1-2.
- C-14 Semantic cache. Keyed by silo; invalidated on ingest. Phase 2.
- C-15 Memory-promotion hook. After each turn, C-05 judges durability; promotes to facts table and Layer 1. Phase 2.
- C-16 Correction/event bus and arbiter machinery. asyncio inbox, pending counter, boundary condition with timeout (~5 s default-to-queue), watcher task that cancels an in-flight generation on interrupt, research command queue (start/queue/interrupt), correction hook that injects research results and self-corrections into the next voice turn or a proactive interjection. Phase 2 (hook), Phase 3 (arbiter).
- C-17 Ingestion source adapters and router. Phase 3 adapters: file drops and coding-session transcripts. Deferred to Phase 6+: browser text-extraction sidecar, terminal capture. Router: structure scoring → document trees (checksum-cached, in document_nodes) / hybrid-only / lazy-only; Qwen-vision OCR for image-bearing inputs. Phase 3.
- C-18 Research pipeline. Agentic loop over grep/FTS/tree/web/code tools; relevance grading; web fallback; multi-hop; parallel fan-out (~4 medium-depth workers); checkpoints plan and progress; status board refreshed at every loop boundary; text responses to text clients; correction events for voice-initiated tasks. Phase 3.
- C-19 Code tools and code knowledge graph. ripgrep / fuzzy / tree-sitter tiers; adopted code-graph MCP servers used as shipped with their own storage: code-review-graph for blast radius and impact, graphify for the multimodal bird's-eye view (docs, PDFs, images, community detection); updated per commit. Phase 3-4.
- C-20 Coding surface (OpenCode). Adopted harness on the local endpoint; transcripts ingested; consumes C-21. Phase 3.
- C-21 MCP server. Exposes MemoryStore recall/remember and selected tools to external agent hosts with scoped auth (silo, user, project) and audit. Phase 3.
- C-22 External API bridge. Routing policy (explicit request → criticality → local confidence cross-checked by cheap verification → complexity → budget); tier selection from capability cards; OCR-first payload builder; budget gate; per-call cost accounting; silo-stamped audit log. Phase 3.
- C-23 Rubber-duck interplay. Shared blackboard and status board (notes-up, read-down, imperatives): notes flow voice → blackboard → research; the status board flows research → voice Layer 1; imperatives (start/queue/interrupt) go directly to the research command queue, never via the blackboard; deep research-transcript access via voice Layer 3. Phase 3 (single model), Phase 4 (Gemma).
- C-24 Consolidation, compaction, scheduler. Idle-window summaries with manifests, archival with stubs, research-job compaction; GPU budget arbitration. Phase 3.
- C-25 Recall ladder. Standing context → hybrid → rerank → context expansion → cited synthesis or honest gap. Phase 3.
- C-26 Salience weights, contradiction detector, tunnel manager (within a silo only). Phase 3-4.
- C-27 Wiki distillation layer (OKF-conformant), per project and therefore per silo. Phase 4.
- C-28 Panel mode. Parallel identical-spec execution or decision-point consensus across two models with an arbiter strategy (judge-selects, vote, synthesize, test-harness-wins); full fan-out recorded; available inside heads-down mode. Phase 4.
- C-29 Eval harness, CI, observability, capability cards. Measurement is deterministic wherever possible: FSM-routing accuracy is exact match against human-labeled verdicts; voice recall seeds a test database with known facts and checks answer content and citation; research multi-hop checks expected citation ids over a fixed test corpus; coding patch pass rate applies the model's patch to a small repository with failing tests and runs the suite; structured-output validity, loop rate, stop compliance, and latency are observed by the harness; a reference-anchored LLM judge is used only for free-text quality slices. The seed set is fixed so that the system is the only variable; runs are keyed by a fingerprint of model, quantization, serving configuration, prompt version, and retrieval-index contents, skipped when unchanged, repeated over several sampling seeds with mean and confidence interval, and scheduled into idle windows with a weekly sanity run. Real-usage signals (thumbs-down, explicit corrections, re-asks) become new labeled cases. Cards for paid remote tiers are scored once per model release and spot-checked monthly on a capped sample. A capability card is a versioned row per (model, quantization, serving config) holding per-slice accuracy, latency percentiles at stated context depths, tool-call validity, loop and stop rates, and cost per task; the router reads cards for escalation thresholds and the bridge for tier selection. Panel mode is a runtime feature, not the measurement process. Phase 1.
- C-30 Web interface and auth. Phase 5.
- C-31 TTS and stream manager. Speaks response_text only. Carry-over.
- C-32 Silo context. An immutable value created once at request entry (silo_id, project_id, user_id); required by every MemoryStore method; never a global; unresolvable silo → reject and log. Phase 0-1.

## 4. Data model (summary)

- D-01 Identity store, outside the silo scheme: Person, voice imprints, Device (with time zone), user_silos membership with a default silo per user.
- D-02 Silo layer, Tier A (migration 001): `silos` table with one seeded default; every content and derived table carries a non-null `silo_id`; projects keyed `(silo_id, project_id)` and referenced by that composite pair; `silo_id` included in every unique constraint and index prefix (content hash unique per silo); tunnels and blocked pairs share a single `silo_id` column; semantic cache, traces, manifests, and outbound payloads carry the silo; leakage tests are permanent. Tier B (deferred until a second silo exists): list partitioning of vector and full-text tables by silo; row-level security keyed to silo under a non-owner role with a transaction-scoped setting (row-level security keyed to *user* arrives in Phase 5 regardless); per-silo roles; instance pin to an allowed silo set. Tier A costs about an hour and nothing at runtime; Tier B is real friction and delivers nothing at one silo.
- D-03 Turn: id, silo_id, project_id, user_id, device_id, role, text (verbatim), recorded_at (UTC), local_tz, model_call_id, trace_id.
- D-04 Fact: id, silo_id, subject, predicate, object, facet, valid_from, valid_to, recorded_at, source, salience components, superseded_by.
- D-05 Document, DocumentNode (Postgres rows), content hash unique per silo, provenance, structure score, index status.
- D-06 Task: silo_id, plan, status, checkpoints, status board, blackboard notes, cost rollup, originating channel.
- D-07 ModelCall: model, slot, tokens in/out/cached, cost, duration, time-to-first-token, thinking setting, silo_id.
- D-08 Tunnels and blocked pairs: one shared `silo_id` column. Manifest and Trace rows carry silo_id. CapabilityCard: model, quant, serving config, fingerprint, per-slice scores, latency percentiles, version, evaluated_at.

## 5. Pathways

- P-01 Voice turn, no FSM active. C-01 → C-02 → C-03 → C-04 → C-32 → C-05. Ignore: persisted as noise, no response. FSM event: P-02. Escalate: C-07 → S-02 → (any proposed transition validated by C-06) → C-31 speaks response_text → C-12 persists turn, trace, cost → C-15 judges promotion.
- P-02 FSM workflow. Entry guard fires — utterance-driven via C-05 (primary), utterance-driven via S-02 when the cue needs conversational context or the router missed it, or signal-driven via C-04. C-05 handles subsequent turns; C-06 validates each transition; state persists per user_id. An illegal proposal returns a hint to the proposer within the bound, then is dropped and traced.
- P-03 Text-channel message. → C-32 → C-05 as arbiter → start (create Task, start C-18) | queue (outbox) | interrupt (cancel event) | FSM command (validated by C-06; illegal commands return to the arbiter, never to the voice slot). Never discarded. Result returns as text (P-14).
- P-04 Research task lifecycle. C-18 plans; loops over tool calls (each a loop boundary); at each boundary: drain outbox, refresh status board, checkpoint, check pending counter (wait up to timeout). Retrieval via C-13/C-25/C-19; escalation via C-22; completion writes cited synthesis and cost rollup; delivery per P-14 or P-05.
- P-05 Voice-initiated research. S-02 answers provisionally and starts a Task with originating channel = voice. Later voice turns add notes (blackboard) or imperatives (command queue). On completion or a material update or self-correction, C-18 raises a correction event through C-16, injected into the next voice turn or a proactive interjection; spoken, never rendered as text.
- P-06 Mid-flight injection and interrupt. Input during generation → inbox → pending counter increments → C-05 rules {discard (voice-origin only) | queue | interrupt}. Queue: outbox. Interrupt: cancel event → watcher cancels generation and aborts the vLLM request → partial output kept → injection appended → C-18 resumes and resteers itself. Boundary wait: if pending > 0, wait up to the timeout, then treat undecided items as queued.
- P-07 Read-down. Status board refreshed every boundary and injected into S-02's Layer 1 on every voice turn during an active research session; deep detail via Layer 3 tools reading the research transcript.
- P-08 Ingestion. Adapter → C-12 (hash per silo, dedupe, provenance, stamps) → rows → fan-out: FTS, vectors, structure scoring → tree build (idle) or lazy-only; fact extraction and contradiction check; wiki update queued.
- P-09 External escalation. Policy triggers → payload builder (local OCR if needed) → silo assertion → budget gate → remote call → ModelCall row → response returns with provenance marked external.
- P-10 Identity and startup. Boot: speaker matrix loaded, Layer 0 warmed, capability cards loaded. New client: device registers time zone. Tiers: unidentified, voice-identified, authenticated.
- P-11 Memory promotion and consolidation. Per turn: C-15. Idle windows: C-24, C-26 reconciliation (Phase 4), eval runs on fingerprint change.
- P-12 Panel mode (Phase 4). Critical task → both models (or Qwen self-fusion) run the identical spec or vote on extracted decisions → arbiter strategy → verdict and candidates recorded → result returns to the requesting pipeline.
- P-13 Heads-down mode (Phase 4+). Mode switch → voice-slot model unloaded → coding model or Harness-1 loaded → voice slot rebinds to a Qwen register → tools rebind by slot → exit reverses.
- P-14 Text response. Every text-channel task returns text to the originating client. Voice-initiated tasks return through P-05. A voice question about a text-initiated task is answered by voice from the status board.
- P-15 Cross-silo movement. No in-place move: export, manifest, re-ingest as new rows with an opaque pointer; approval is a pluggable policy (the owner in Fawkes).

## 6. Interfaces and seams

- X-01 OpenAI-compatible `/v1/chat/completions`: the only way any pipeline talks to any model.
- X-02 MemoryStore interface: the only way any component reads or writes memory; every method takes a silo context.
- X-03 ingest(): the only write path.
- X-04 MCP server: the only way external agent hosts reach Fawkes memory or tools.
- X-05 Tool registry: slot-bound; identical surface for any model in a slot.
- X-06 Event bus contract: inbox/outbox/pending/cancel primitives; swappable for a real message bus.

## 7. Invariants

- I-01 Only response_text reaches TTS.
- I-02 No state mutates without validator approval; rejected proposals return to their proposer with a hint, bounded.
- I-03 No LLM round-trip inside voice retrieval.
- I-04 ASR text reaches research only via the voice pipeline; text-channel messages are never discarded.
- I-05 The arbiter triages; the research model resteers.
- I-06 Verbatim rows are never deleted or rewritten by summarization; manifests accompany every transformation.
- I-07 All timestamps are UTC time-zone-aware; rendering is per session.
- I-08 Every model call has a ModelCall row with tokens and cost.
- I-09 Derived stores may be dropped and rebuilt from the system of record.
- I-10 A phase exit test never leaves the test suite.
- I-11 Response modality follows input modality; background results return through the originating channel.
- I-12 No row, cache entry, tunnel, prompt item, or outbound payload crosses a silo boundary; leakage tests gate every phase exit.

## 8. Known open questions for reviewers

- Q-01 Single-card co-residency of 27B + 4B + KV in Phase 1: is the memory arithmetic sound?
- Q-02 Is the three-channel routing doctrine complete?
- Q-03 Does the arbiter protocol have a race or deadlock the description misses?
- Q-04 Is the read-down design sufficient for the voice pipeline to discuss research intelligently without context bloat?
- Q-05 Are the escalation signals adequate, and how should thresholds be calibrated from capability cards?
- Q-06 Is anything required by R-01…R-11 unassigned to a component?
- Q-07 Is the Tier A / Tier B split of the silo layer the right cut — is anything in Tier A avoidable, and is anything in Tier B needed before a second silo?
- Q-08 Are the deterministic measurement methods in C-29 sufficient to produce trustworthy capability cards without routine human grading?
