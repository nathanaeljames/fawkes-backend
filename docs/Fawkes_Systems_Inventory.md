# Fawkes Systems Inventory
All subsystems, what each does, and the phase in which it is built. Phases refer to Fawkes_Implementation_Plan.md. "Carry-over" = exists from Iteration 1 and is retained. Updated 2026-09-20.

## Speech and identity (carry-over, evolving)

| Subsystem | Purpose | Phase |
|---|---|---|
| Audio I/O + WebSocket layer | Streaming capture/playback, multi-client connection state; each client registers device id and IANA time zone | Carry-over |
| VAD (MarbleNet) + endpointing | Speech detection, utterance finalization, first-line noise filtering | Carry-over |
| Interim ASR (FastConformer) | Streaming partial transcripts; SSM replacement likely ~Phase 3.5 | Carry-over |
| Final ASR (Canary-Qwen) | Final utterance transcription; SSM replacement likely ~Phase 3.5 | Carry-over |
| Speaker identity (ECAPA-TDNN) | In-memory embedding matrix built from Postgres at startup; per-second matching; audio → user_id; identity data lives outside silos | Carry-over |
| TTS (XTTS / Piper) + stream manager | Voice output; the only output channel for voice input | Carry-over |
| Identity & auth subsystem | Deferred identification tiers, recency-modulated thresholds, passphrase/auth FSM, retroactive turn attribution; user-to-silo membership with a default silo | Phase 2 |
| Operational data migration | Move iteration-1 DuckDB tables into Postgres; retire DuckDB completely | Phase 2 |

## Cognition and routing

| Subsystem | Purpose | Phase |
|---|---|---|
| Small-model router / arbiter (Qwen3-4B-Instruct class) | Voice channel: {ignore, FSM event, escalate}; primary FSM driver. Text channel: {start, queue, interrupt, FSM command}; never discards. Mid-flight: {discard (voice-origin only), queue, interrupt}; ~5 s default-to-queue timeout | Phase 1-3 |
| Prompt builder | State-conditioned prompt assembly; cache-stable prefix; Layers 0-3; status board refreshed per loop boundary; local-time and elapsed-time rendering; silo assertion | Phase 1 |
| FSM registry + validator | Statechart definitions; deterministic transition legality; rejected proposals return to the proposer with a hint, bounded, then dropped and traced | Phase 1-2 |
| Structured-output contract | JSON schema for slot updates / transitions / tool calls / response_text / confidence; constrained decoding | Phase 1 |
| Voice-slot model | Conversational cognition for escalated voice turns; may answer, write a blackboard note, issue start/queue/interrupt imperatives, or propose FSM transitions that need conversational context or that the router missed. Qwen3.8-27B through Phase 3; Gemma 4 from Phase 4 | Phase 1 (text) / 2 (voice) |
| Research-slot model | Qwen3.8-27B with thinking on; drives the research/coding pipeline; resteers itself; answers text clients in text | Phase 3 |
| Tool registry + executor | Opaque, slot-bound tool surface; sandboxed execution | Phase 2 |
| Correction/event bus + arbiter machinery | asyncio inbox, pending counter, boundary condition wait with timeout, watcher cancel, research command queue; carries verification results, research results and self-corrections back to voice, and cross-pipeline imperatives | Phase 2 (hook), Phase 3 (arbiter) |
| Voice register controls | Thinking off / budgeted thinking, brevity persona, stall-word authorization | Phase 2 |

## Memory substrate

| Subsystem | Purpose | Phase |
|---|---|---|
| Silo layer — Tier A | `silos` table; non-null `silo_id` on every row; projects keyed `(silo_id, project_id)`; `silo_id` in every unique constraint and index prefix; single-silo tunnels and blocked pairs; per-silo content hash; silo context on every MemoryStore call; export-and-re-ingest for crossings; permanent leakage tests | Phase 0-1 |
| Silo layer — Tier B | List partitioning of vector/FTS tables by silo; silo-keyed forced row-level security under a non-owner role; per-silo roles; instance pin to an allowed silo set | Phase 5+ (first real second silo) |
| MemoryStore (Postgres + pgvector + FTS) | System of record: verbatim transcripts, documents, document trees, bi-temporal facts, model calls, traces, manifests, capability cards; UTC `timestamptz`; backend-agnostic interface | Phase 1 |
| Ontology + schema constraints | Versioned entity/relation/facet vocabulary; enum tables; runtime enforcement | Phase 1 |
| ingest() contract | Single write path: hashing (per silo), idempotency, provenance, silo/project stamps, per-store fan-out; turns, traces, and cost rows included | Phase 1 |
| Hybrid retrieval | BM25 + vector, Reciprocal Rank Fusion, scope weighting within a silo; optional reranker | Phase 1-2 |
| Semantic cache | Query-similarity cache keyed by silo | Phase 2 |
| Tiered context loader | Layers 0-3 for the voice path | Phase 2 |
| Memory-promotion hook | Per-turn small-model judgment: promote durable facts | Phase 2 |
| Recall ladder | Standing context → hybrid → rerank → context expansion → cited synthesis or honest gap | Phase 3 |
| Consolidation & compaction jobs | Idle-time summaries-with-manifests, archival with stubs; compaction; research-job checkpoints | Phase 3 |
| Contradiction detector | Ingest-time fact-conflict check; severity-gated | Phase 3-4 |
| Tunnel manager | Cross-project scope weights within a silo | Phase 3-4 |
| Wiki distillation layer (OKF-conformant) | Per-project compiled knowledge bundles, lifecycle states, lint | Phase 4 |
| Reconciliation jobs | Cross-store agreement sampling, drift alerts, type histograms | Phase 4 |
| History importer | Back-catalog ingestion of Claude/ChatGPT exports | Phase 5 |

## Pipelines and surfaces

| Subsystem | Purpose | Phase |
|---|---|---|
| Voice pipeline (latency-critical) | The end-to-end conversational loop; voice in, voice out | Phase 1-2 |
| Ingestion source adapters | File drops and coding-session transcripts (Phase 3); browser text-extraction sidecar and terminal capture (Phase 6+, if still wanted) | Phase 3 / 6+ |
| Ingestion router | Structure scoring; trees / hybrid / lazy-only; Qwen-vision OCR at ingest | Phase 3 |
| Research pipeline | Agentic loop, CRAG grading, web fallback, multi-hop, ~4-worker fan-out; checkpoints; text responses; correction events for voice-initiated tasks | Phase 3 |
| Code tools + code knowledge graphs | ripgrep / fuzzy / tree-sitter tiers; code-review-graph (blast radius, impact) and graphify (multimodal graph, community detection) used as shipped with their own storage; updated per commit | Phase 3-4 |
| Coding surface (OpenCode) | Adopted coding harness on local endpoint; transcripts into MemoryStore | Phase 3 |
| MCP server | MemoryStore exposed to external agent hosts with scoped auth + audit | Phase 3 |
| External API bridge | Local-first routing policy; tier selection from capability cards; OCR-first payloads; budget gate; cost accounting; audit | Phase 3 |
| Rubber-duck interplay | Shared blackboard + status board (notes-up, read-down, imperatives); imperatives direct to the research command queue; arbiter-gated; research model resteers itself | Phase 3 |
| Second-model service (Gemma 4) | Voice-slot model on its own card | Phase 4 |
| Panel mode | Identical-spec parallel execution with arbiter strategies and decision-point consensus; available inside heads-down mode | Phase 4 |
| Heads-down mode | Task-mode slot swap for a second coding model or Harness-1 | Phase 4+ |
| Web interface + auth | Chat, uploads, project management, sessions, row-level security; text in, text out | Phase 5 |
| Screen-context subsystem | Accessibility-API-first screen reading with event-triggered OCR fallback | Phase 6+ |
| Digest/scrape crons; CV pipeline | Scheduled ingestion; real-time object recognition | Phase 6+ |
| SSM ASR swap | Mamba-class replacement for interim and final ASR | Phase 6+ (likely ~3.5) |
| Cocktail Party separation; KVM/HID control | Separate repositories and tracks | Separate tracks |

## Infrastructure and quality

| Subsystem | Purpose | Phase |
|---|---|---|
| Serving (vLLM) | OpenAI-compatible endpoint; Qwen3.8-27B Int4 on the RTX 3090; serving-recipe experiments measured by the harness | Phase 1 |
| Utility-model serving (llama.cpp) | 4B router on the same 3090; embedder on CPU until the second GPU; reranker | Phase 1-2 |
| Eval harness + capability cards | Deterministic slice scoring (labeled verdicts, seeded recall, fixed-corpus citations, failing-test repositories, schema validation, loop/stop/latency observation); runs on fingerprint change plus weekly; versioned capability cards read by router and bridge; remote tiers scored per release | Phase 1 |
| CI pipeline (GitHub Actions) | Per-commit unit + integration + mocked-LLM FSM lifecycle tests + silo leakage tests + secret scanning; phase exit tests as permanent markers | Phase 1 |
| Observability + cost accounting | One structured JSON trace per turn/call; per-task rollups | Phase 1 |
| Scheduler | Idle-window arbitration for background jobs and GPU budget | Phase 3 |
| Backup/restore | System-of-record dumps from day one | Phase 1 |
| Security & sandboxing | Tool container isolation; MCP scopes; user-keyed row-level security (Phase 5); web-pass hardening | Phase 2, 5 |
| Diagram set | Architecture at complexity levels 4 (README), 7 (engineering reference), 10 (exhaustive), sources in `docs/diagrams/` | Phase 0 |
