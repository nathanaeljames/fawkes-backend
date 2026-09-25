# Fawkes Systems Inventory
All subsystems, what each does, and the phase in which it is built. Phases refer to Fawkes_Implementation_Plan.md. "Carry-over" = exists from Iteration 1 and is retained. Updated 2026-09-25 (2026-09-19 detail restored in full; 2026-09-20 and 2026-09-25 decisions applied).

## Speech and identity (carry-over, evolving)

| Subsystem | Purpose | Phase |
|---|---|---|
| Audio I/O + WebSocket layer | Streaming capture/playback, multi-client connection state; each client registers device id and IANA time zone | Carry-over |
| VAD (MarbleNet) + endpointing | Speech detection, utterance finalization, first-line noise filtering | Carry-over |
| Interim ASR (FastConformer) | Streaming partial transcripts; SSM replacement likely ~Phase 3.5 | Carry-over |
| Final ASR (Canary-Qwen) | Final utterance transcription; SSM replacement likely ~Phase 3.5 | Carry-over |
| Speaker identity (ECAPA-TDNN) | In-memory embedding matrix built from Postgres at startup and updated on enrollment; per-second matching; audio → user_id bridge; identity data lives outside silos | Carry-over |
| TTS (XTTS / Piper) + stream manager | Voice output, cloning, sequential ordering, feedback prevention; the only output channel for voice input | Carry-over |
| Identity & auth subsystem | Deferred identification tiers, recency-modulated thresholds, passphrase/auth FSM, retroactive turn attribution; user-to-silo membership with a default silo | Phase 2 |
| Operational data migration | Move iteration-1 DuckDB tables (speakers/imprints, pangrams, passages) into Postgres; retire DuckDB completely | Phase 2 |

## Cognition and routing

| Subsystem | Purpose | Phase |
|---|---|---|
| Small-model router / arbiter (Qwen3-4B-Instruct class) | Voice channel: {ignore, FSM event (+slots, tiny in-workflow replies), escalate to the voice-slot model} with confidence; primary FSM driver; Layer 2 retrieval runs only on escalate verdicts. Text channel (arbiter): {start (dispatch now), queue for next boundary, interrupt, FSM command}; never discards text. Mid-flight: {discard (voice-origin only), queue, interrupt}; ~5 s default-to-queue timeout. Experiments (harness-gated): retrieval hints and a complexity flag emitted in the same verdict, no extra round-trip | Phase 1-3 |
| Prompt builder | State-conditioned prompt assembly; cache-stable prefix; Layers 0-3 injected in fixed order (Layer 0, Layer 1, Layer 2 project slice in the cacheable prefix, then the per-turn Layer 2 topic slice as a frozen snapshot); status board refreshed per loop boundary; local-time and elapsed-time rendering; silo assertion on every context item | Phase 1 |
| FSM registry + validator | Statechart definitions; deterministic transition legality; slot schemas; rejected proposals return to the proposer with a hint, bounded, then dropped and traced | Phase 1-2 |
| Structured-output contract | JSON schema for slot updates / transitions / tool calls / response_text / confidence; constrained decoding | Phase 1 |
| Voice-slot model | Conversational cognition for escalated voice turns; may answer, write a blackboard note, or issue start/queue/interrupt imperatives; may propose FSM transitions (utterance-driven guards) for validation, in particular transitions that need conversational context or that the router missed. Qwen3.8-27B through Phase 3; Gemma 4 from Phase 4 | Phase 1 (text) / 2 (voice) |
| Research-slot model | Qwen3.8-27B with thinking on; drives the research/coding pipeline and performs its own resteering; answers text clients in text | Phase 3 |
| Tool registry + executor | Opaque, slot-bound tool surface; sandboxed (Docker) execution; every tool carries a latency class (fast: SQL lookups and joins, full-text/trigram search, document-node fetch, semantic-cache lookup, single-shot lookups; slow: LLM-guided tree descent, web, multi-document synthesis); the voice slot sees fast tools only, at most one tool round per turn under Budget B (tentative ~500 ms) | Phase 2 |
| Correction/event bus + arbiter machinery | asyncio inbox, pending counter, boundary condition wait with timeout, watcher task for out-of-band cancel, research command queue; carries background-verification results, voice-initiated research results and self-corrections back to the voice channel, and cross-pipeline imperatives | Phase 2 (hook), Phase 3 (arbiter) |
| Voice register controls | Thinking off / budgeted thinking, brevity persona, stall-word authorization | Phase 2 |

## Memory substrate

| Subsystem | Purpose | Phase |
|---|---|---|
| Silo layer — Tier A | `silos` table; non-null `silo_id` on every content and derived row; projects keyed `(silo_id, project_id)`; `silo_id` in every unique constraint and index prefix; single-silo tunnels and blocked pairs; per-silo content hash; immutable silo context on every MemoryStore call; export-and-re-ingest for cross-silo moves; permanent leakage tests. An owner change about projects that do not live in silos is pending (Specification Q-09) | Phase 0-1 |
| Silo layer — Tier B | List partitioning of vector/FTS tables by silo; silo-keyed forced row-level security under a non-owner role; per-silo roles; instance pin to an allowed silo set | Phase 5+ (first real second silo) |
| MemoryStore (Postgres + pgvector + FTS) | System of record: verbatim transcripts, documents, document trees, bi-temporal facts, model calls, traces, manifests, capability cards, feedback events, panel results; UTC `timestamptz` everywhere; backend-agnostic interface; vector and full-text indexes sized to shared_buffers and prewarmed at startup (pg_prewarm) so the voice path never hits cold NVMe pages | Phase 1 |
| Ontology + schema constraints | Versioned entity/relation/facet vocabulary; enum tables; runtime enforcement; facet vocabulary v1: fact / preference / decision / event / task-state (approved 2026-09-24; candidates insight and advice noted, not adopted) | Phase 1 |
| ingest() contract | Single write path: hashing (per silo), idempotency, provenance, silo/project stamps, per-store fan-out handlers; turns, traces, and cost rows included | Phase 1 |
| Hybrid retrieval | BM25 + vector, Reciprocal Rank Fusion, scope weighting within a silo; optional reranker; the whole utterance is the query (no keyword-selection step); deterministic sharpeners: ontology-entity dictionary match, previous-turn append | Phase 1-2 |
| Semantic cache | Query-similarity answer/retrieval cache keyed by silo with invalidation hooks; distinct from Postgres's page cache and vLLM's prefix cache | Phase 2 |
| Tiered context loader | Layers 0-3 for the voice path, each defined by content and mechanism. Layer 0: persona (directives, voice register, boundaries) + tool registry; static, prompt-cached. Layer 1: promoted durable facts, active projects and goals, research status board; SQL by user_id on identity. Layer 2 project slice: verbatim recent turns of the active project under a token cap (2-4K placeholder), older sessions as summaries plus pointers; SQL by project_id when the project resolves, refreshed at session boundaries, cache-stable. Layer 2 topic slice: utterance-relevant memories from the silo weighted to the active project; hybrid search per escalated turn, frozen snapshot, discarded next turn. Layer 3: whole-silo transcript search, SQL-join fact lookups, document-node lookup, research-transcript access; single-shot tool calls the model requests, fast-class only on voice | Phase 1 (text) / 2 (voice) |
| Memory-promotion hook | Per-turn small-model judgment: promote durable facts | Phase 2 |
| Recall ladder | Standing context → hybrid → rerank → context expansion → cited synthesis or honest gap | Phase 3 |
| Consolidation & compaction jobs | Idle-time summaries-with-manifests, archival with stubs; long-session compaction; research-job checkpoints | Phase 3 |
| Contradiction detector | Ingest-time fact-conflict check; severity-gated log or clarification FSM | Phase 3-4 |
| Tunnel manager | Cross-project scope weights within a silo: propose/attribute/reinforce/decay/destroy | Phase 3-4 |
| Wiki distillation layer (OKF-conformant) | Per-project compiled knowledge bundles (markdown files, per silo), lifecycle states, lint | Phase 4 |
| Reconciliation jobs | Cross-store agreement sampling, drift alerts, type histograms | Phase 4 |
| History importer | Back-catalog ingestion of Claude/ChatGPT exports with project assignment; after project scopes exist | Phase 5 |

## Pipelines and surfaces

| Subsystem | Purpose | Phase |
|---|---|---|
| Voice pipeline (latency-critical) | The end-to-end conversational loop; deterministic + single-shot retrieval only; voice in, voice out | Phase 1-2 |
| Ingestion source adapters | File drops and coding-session transcripts → `ingest()` (Phase 3); browser text-extraction sidecar (DevTools protocol / content script) and terminal capture (stdout/tmux) (Phase 6+, if still wanted) | Phase 3 / 6+ |
| Ingestion router | Structure scoring; route to trees / hybrid / lazy-only; Qwen-vision OCR at ingest; document-node lookup (SQL, milliseconds) is a fast voice tool, LLM-guided tree descent (multi-second) belongs to the research pipeline only | Phase 3 |
| Research pipeline | Agentic loop, CRAG grading, web fallback, multi-hop (reasoning hops), parallel fan-out (~4 medium-depth workers); LLM-guided document-tree descent; checkpoints progress for crash recovery; text responses to text clients; correction events for voice-initiated tasks | Phase 3 |
| Code tools + code knowledge graphs | ripgrep / fuzzy / tree-sitter tiers; code-review-graph (blast radius, callers/callees, PR-grade impact) and graphify (multimodal graph across code, docs, PDFs, images; community detection) used as shipped with their own storage, updated per commit | Phase 3-4 |
| Coding surface (OpenCode) | Adopted coding harness on local endpoint; transcripts into MemoryStore | Phase 3 |
| MCP server | MemoryStore (and tools) exposed to external agent hosts with scoped auth (silo, user, project) + audit | Phase 3 |
| External API bridge | Local-first routing policy (confidence, complexity, criticality, explicit request); remote tier selection from capability cards; OCR-first text/JSON/Markdown payloads; budget gate; token/cost accounting; silo-stamped audit log; compression proxy later only if spend is material | Phase 3 |
| Rubber-duck interplay | Shared blackboard + status board (notes-up, read-down, imperatives): blackboard carries notes only; imperative channel direct to the research command queue; read-down = status board (per loop boundary into voice Layer 1) + deep transcript access via Layer 3; arbiter-gated queue/interrupt; research model resteers itself. v1 on the single served Qwen | Phase 3 |
| Second-model service (Gemma 4) | Voice-slot model on its own card: conversational register + decorrelated second opinion | Phase 4 |
| Panel mode | Identical-spec parallel execution with arbiter strategies (judge-selects, vote, synthesize, test-harness-wins) and decision-point consensus; available inside heads-down mode | Phase 4 |
| Heads-down mode | Task-mode slot swap: conversational model's GPU slot given to a second coding model or Harness-1; voice falls back to a Qwen register | Phase 4+ |
| Web interface + auth | Chat, uploads, project management, sessions, row-level-security scoping; optional model-target selector; text in, text out | Phase 5 |
| Screen-context subsystem | Accessibility-API-first screen reading with event-triggered OCR fallback; HUD foundation | Phase 6+ |
| Digest/scrape crons | Scheduled ingestion → goal-cross-referenced digests | Phase 6+ |
| CV pipeline | Real-time object recognition/tracking | Phase 6+ |
| SSM ASR swap | Mamba-class replacement for interim and final ASR | Phase 6+ (likely ~3.5) |
| Cocktail Party separation | Speaker-separation research; separate repository and track | Separate track |
| KVM/HID computer control | HDMI-capture + USB-HID control of un-instrumented machines | Separate track |

## Infrastructure and quality

| Subsystem | Purpose | Phase |
|---|---|---|
| Serving (vLLM) | OpenAI-compatible endpoint; Qwen3.8-27B Int4 on the RTX 3090; serving-recipe experiments (tuned vLLM fork, tuned llama.cpp fork) measured by the harness; prefix caching as measured optimization, measured first because the verbatim Layer 2 project slice depends on it | Phase 1 |
| Utility-model serving (llama.cpp) | 4B router on the same 3090 beside vLLM; embedder on CPU until the second GPU; reranker | Phase 1-2 |
| Eval harness + capability cards | Deterministic slice scoring (labeled verdicts, seeded recall, fixed-corpus citations, failing-test repositories from Phase 3, schema validation, loop/stop/latency observation) plus behavioral checks and latency timing; nightly smoke subset, full card runs on fingerprint change and weekly; versioned capability cards read by router and bridge; remote tiers scored per release on research-tier slices only; feedback events and panel results feed the labeled set; gates phase exits | Phase 1 |
| CI pipeline (GitHub Actions) | Per-commit unit + integration + mocked-LLM FSM lifecycle functional tests (shared FakeLLM fixture) + silo leakage tests + secret scanning; phase exit tests as permanent markers | Phase 1 |
| Observability + cost accounting | One structured JSON trace per turn/call: stage timings, retrievals, FSM state, tokens in/out/cached, external cost, task id, silo; per-task rollups | Phase 1 |
| Scheduler | Idle-window arbitration for background jobs and GPU budget | Phase 3 |
| Backup/restore | System-of-record dumps from day one | Phase 1 |
| Security & sandboxing | Tool container isolation; MCP scopes; user-keyed row-level security (Phase 5); web-pass hardening | Phase 2, 5 |
| Diagram set | Architecture at complexity levels 4 (README), 7 (engineering reference), 10 (exhaustive), sources in `docs/diagrams/` | Phase 0 |
