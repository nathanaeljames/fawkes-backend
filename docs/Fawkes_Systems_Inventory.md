# Fawkes Systems Inventory
All subsystems, what each does, and the phase in which it is built. Phases refer to Fawkes_Implementation_Plan.md. "Carry-over" = exists from Iteration 1 and is retained. Updated 2026-09-19.

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
| Operational data migration | Move iteration-1 DuckDB tables into Postgres; retire DuckDB completely | Phase 2 |

## Cognition and routing

| Subsystem | Purpose | Phase |
|---|---|---|
| Small-model router / arbiter (Qwen3-4B-Instruct class) | Voice channel: {ignore, FSM event (+slots, tiny replies), escalate}. Text channel: {start, queue, interrupt, FSM command}; never discards text. Mid-flight: {discard (voice-origin only), queue, interrupt}; ~5 s default-to-queue timeout | Phase 1-3 |
| Prompt builder | State-conditioned prompt assembly; cache-stable prefix; layer injection; status board refreshed per loop boundary; local-time and elapsed-time rendering; silo assertion on every context item | Phase 1 |
| FSM registry + validator | Statechart definitions; deterministic transition legality; slot schemas; rejected proposals return to the proposer with a hint, bounded, then dropped and traced | Phase 1-2 |
| Structured-output contract | JSON schema for slot updates / transitions / tool calls / response_text / confidence; constrained decoding | Phase 1 |
| Voice-slot model | Conversational cognition for escalated voice turns; may answer, write a blackboard note, or issue start/queue/interrupt imperatives; may propose FSM transitions (utterance-driven guards) for validation. Qwen3.8-27B through Phase 3; Gemma 4 from Phase 4 | Phase 1 (text) / 2 (voice) |
| Research-slot model | Qwen3.8-27B with thinking on; drives the research/coding pipeline; resteers itself; answers text clients in text | Phase 3 |
| Tool registry + executor | Opaque, slot-bound tool surface; sandboxed (Docker) execution | Phase 2 |
| Correction/event bus + arbiter machinery | asyncio inbox, pending counter, boundary condition wait with timeout, watcher task for out-of-band cancel; carries background-verification results, voice-initiated research results back to the voice channel, and cross-pipeline imperatives | Phase 2 (hook), Phase 3 (arbiter) |
| Voice register controls | Thinking off / budgeted thinking, brevity persona, stall-word authorization | Phase 2 |

## Memory substrate

| Subsystem | Purpose | Phase |
|---|---|---|
| Silo layer | `silos` table; non-null `silo_id` on every content and derived row; projects keyed `(silo_id, project_id)`; single-silo tunnels and blocked pairs; per-silo content hash; list-partitioned derived tables; forced row-level security under a non-owner role; immutable silo context on every MemoryStore call; instance pin to an allowed silo set; export-and-re-ingest for cross-silo moves; permanent leakage tests | Phase 0-1 |
| MemoryStore (Postgres + pgvector + FTS) | System of record: verbatim transcripts, documents, document trees, bi-temporal facts, model calls, traces, manifests, capability cards; UTC `timestamptz` everywhere; backend-agnostic interface | Phase 1 |
| Ontology + schema constraints | Versioned entity/relation/facet vocabulary; enum tables; runtime enforcement | Phase 1 |
| ingest() contract | Single write path: hashing (per silo), idempotency, provenance, silo/project stamps, per-store fan-out handlers; turns, traces, and cost rows included | Phase 1 |
| Hybrid retrieval | BM25 + vector, Reciprocal Rank Fusion, scope weighting within a silo; optional reranker | Phase 1-2 |
| Semantic cache | Query-similarity cache keyed by silo with invalidation hooks | Phase 2 |
| Tiered context loader | Layers 0-3 for the voice path | Phase 2 |
| Memory-promotion hook | Per-turn small-model judgment: promote durable facts | Phase 2 |
| Recall ladder | Standing context → hybrid → rerank → context expansion → cited synthesis or honest gap | Phase 3 |
| Consolidation & compaction jobs | Idle-time summaries-with-manifests, archival with stubs; long-session compaction; research-job checkpoints | Phase 3 |
| Contradiction detector | Ingest-time fact-conflict check; severity-gated log or clarification FSM | Phase 3-4 |
| Tunnel manager | Cross-project scope weights within a silo: propose/attribute/reinforce/decay/destroy | Phase 3-4 |
| Wiki distillation layer (OKF-conformant) | Per-project compiled knowledge bundles (markdown files, per silo), lifecycle states, lint | Phase 4 |
| Reconciliation jobs | Cross-store agreement sampling, drift alerts, type histograms | Phase 4 |
| History importer | Back-catalog ingestion of Claude/ChatGPT exports with project assignment | Phase 5 |

## Pipelines and surfaces

| Subsystem | Purpose | Phase |
|---|---|---|
| Voice pipeline (latency-critical) | The end-to-end conversational loop; deterministic + single-shot retrieval only; voice in, voice out | Phase 1-2 |
| Ingestion source adapters | File drops and coding-session transcripts → `ingest()` (Phase 3); browser text-extraction sidecar and terminal capture (Phase 6+, if still wanted) | Phase 3 / 6+ |
| Ingestion router | Structure scoring; route to trees / hybrid / lazy-only; Qwen-vision OCR at ingest | Phase 3 |
| Research pipeline | Agentic loop, CRAG grading, web fallback, multi-hop, parallel fan-out; checkpoints; text responses to text clients; correction events for voice-initiated tasks | Phase 3 |
| Code tools + code knowledge graph | ripgrep / fuzzy / tree-sitter tiers; adopted code-graph MCP server used as shipped (own storage), updated per commit; callers/callees, blast radius, community structure | Phase 3-4 |
| Coding surface (OpenCode) | Adopted coding harness on local endpoint; transcripts into MemoryStore | Phase 3 |
| MCP server | MemoryStore (and tools) exposed to external agent hosts with scoped auth (silo, user, project) + audit | Phase 3 |
| External API bridge | Local-first routing policy; tier selection from capability cards; OCR-first text/JSON/Markdown payloads; budget gate; token/cost accounting; silo-stamped audit log | Phase 3 |
| Rubber-duck interplay | Blackboard (notes only); imperative channel direct to the research command queue; read-down = status board (per loop boundary into voice Layer 1) + deep transcript access via Layer 3; arbiter-gated queue/interrupt; research model resteers itself. v1 on the single served Qwen | Phase 3 |
| Second-model service (Gemma 4) | Voice-slot model on its own card: conversational register + decorrelated second opinion | Phase 4 |
| Panel mode | Identical-spec parallel execution with arbiter strategies and decision-point consensus; available inside heads-down mode | Phase 4 |
| Heads-down mode | Task-mode slot swap: conversational model's GPU slot given to a second coding model or Harness-1; voice falls back to a Qwen register | Phase 4+ |
| Web interface + auth | Chat, uploads, project management, sessions, row-level-security scoping; text in, text out | Phase 5 |
| Screen-context subsystem | Accessibility-API-first screen reading with event-triggered OCR fallback; HUD foundation | Phase 6+ |
| Digest/scrape crons | Scheduled ingestion → goal-cross-referenced digests | Phase 6+ |
| CV pipeline | Real-time object recognition/tracking | Phase 6+ |
| SSM ASR swap | Mamba-class replacement for interim and final ASR | Phase 6+ (likely ~3.5) |
| Cocktail Party separation | Speaker-separation research; separate repository and track | Separate track |
| KVM/HID computer control | HDMI-capture + USB-HID control of un-instrumented machines | Separate track |

## Infrastructure and quality

| Subsystem | Purpose | Phase |
|---|---|---|
| Serving (vLLM) | OpenAI-compatible endpoint; Qwen3.8-27B Int4 on the RTX 3090; serving-recipe experiments (tuned vLLM fork, tuned llama.cpp fork) measured by the harness; prefix caching as measured optimization | Phase 1 |
| Utility-model serving (llama.cpp) | 4B router on the same 3090 beside vLLM; embedder on CPU until the second GPU; reranker | Phase 1-2 |
| Eval harness + capability cards | Seeded question sets, behavioral checks, latency timing; per-model capability cards rewritten nightly; gates phase exits | Phase 1 |
| CI pipeline (GitHub Actions) | Per-commit unit + integration + mocked-LLM FSM lifecycle tests + silo leakage tests + secret scanning; phase exit tests as permanent markers | Phase 1 |
| Observability + cost accounting | One structured JSON trace per turn/call: stage timings, retrievals, FSM state, tokens in/out/cached, external cost, task id, silo; per-task rollups | Phase 1 |
| Scheduler | Idle-window arbitration for background jobs and GPU budget | Phase 3 |
| Backup/restore | System-of-record dumps from day one | Phase 1 |
| Security & sandboxing | Tool container isolation; MCP scopes; row-level security; web-pass hardening | Phase 0, 2, 5 |
