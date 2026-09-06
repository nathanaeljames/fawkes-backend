# Fawkes Systems Inventory
All subsystems, what each does, and the phase in which it is built. Phases refer to Fawkes_Implementation_Plan.md. "Carry-over" = exists from Iteration 1 and is retained.

## Speech and identity (carry-over, evolving)

| Subsystem | Purpose | Phase |
|---|---|---|
| Audio I/O + WebSocket layer | Streaming capture/playback, multi-client connection state | Carry-over |
| VAD (MarbleNet) + endpointing | Speech detection, utterance finalization, first-line noise filtering | Carry-over |
| Interim ASR (FastConformer) | Streaming partial transcripts (lookahead tuned; SSM replacement Phase 6+) | Carry-over |
| Final ASR (Canary-Qwen) | Final utterance transcription (SSM replacement Phase 6+) | Carry-over |
| Speaker identity (ECAPA-TDNN) | In-memory embedding matrix, per-second matching, audio → user_id bridge | Carry-over |
| TTS (XTTS / Piper) + stream manager | Voice output, cloning, sequential ordering, feedback prevention | Carry-over |
| Identity & auth subsystem | Deferred identification tiers, recency-modulated thresholds, passphrase/auth FSM, retroactive turn attribution | Phase 2 |

## Cognition

| Subsystem | Purpose | Phase |
|---|---|---|
| Prompt builder | State-conditioned prompt assembly; cache-stable prefix ordering; layer injection | Phase 1 |
| FSM registry + validator | Statechart definitions (enrollment, voice clone, auth, clarification); transition legality; slot schemas | Phase 1-2 |
| Small-model router | Intent recognition, noise/backchannel gating ("ignore" class), slot capture, confidence-based escalation | Phase 1-2 |
| Structured-output contract | JSON schema for slot updates / transitions / tool calls / response_text; constrained decoding | Phase 1 |
| Tool registry + executor | Opaque tool surface for the LLM; sandboxed (Docker) execution | Phase 2 |
| Correction/event bus | Queue injecting background-verification results and cross-pipeline imperatives into conversation state at loop boundaries | Phase 2 |
| Voice register controls | Thinking off / budgeted thinking, brevity persona, stall-word authorization | Phase 2 |

## Memory substrate

| Subsystem | Purpose | Phase |
|---|---|---|
| MemoryStore (Postgres + pgvector + FTS) | System of record: verbatim transcripts, documents, bi-temporal facts; backend-agnostic interface | Phase 1 |
| Ontology + schema constraints | Versioned entity/relation/facet vocabulary; enum tables; contradiction comparability | Phase 1 |
| ingest() contract | Single write path: hashing, idempotency, provenance, per-store fan-out handlers | Phase 1 |
| Hybrid retrieval | BM25 + vector, Reciprocal Rank Fusion, scope weighting; optional reranker | Phase 1-2 |
| Semantic cache | Query-similarity answer/retrieval cache with invalidation hooks | Phase 2 |
| Tiered context loader | Layers 0-3 for the voice path | Phase 2 |
| Memory-promotion hook | Per-turn small-model judgment: promote durable facts to standing context/facts table | Phase 2 |
| History importer | Back-catalog ingestion: iteration-1 logs, exported Claude/ChatGPT history | Phase 2-3 |
| Recall ladder | Standing context → hybrid → rerank → context expansion → cited synthesis or honest gap | Phase 3 |
| Consolidation & compaction jobs | Idle-time summaries-with-manifests, archival with stubs; long-session compaction | Phase 3 |
| Contradiction detector | Ingest-time fact-conflict check; severity-gated log or clarification FSM | Phase 3-4 |
| Tunnel manager | Cross-project scope weights: propose/attribute/reinforce/decay/destroy | Phase 3-4 |
| Wiki distillation layer (OKF-conformant) | Per-project compiled knowledge bundles, lifecycle states, lint | Phase 4 |
| Reconciliation jobs | Cross-store agreement sampling, drift alerts, type histograms | Phase 4 |

## Pipelines and surfaces

| Subsystem | Purpose | Phase |
|---|---|---|
| Voice pipeline (latency-critical) | The end-to-end conversational loop; deterministic + single-shot retrieval only | Phase 1-2 |
| Ingestion router | Structure scoring; route to trees / hybrid / lazy-only; Qwen-vision OCR at ingest | Phase 3 |
| Research pipeline | Agentic loop, CRAG grading, web fallback, multi-hop, parallel fan-out | Phase 3 |
| Coding surface (OpenCode) | Adopted coding harness on local endpoint; transcripts into MemoryStore | Phase 3 |
| MCP server | MemoryStore (and tools) exposed to external agent hosts with scoped auth + audit | Phase 3 |
| Claude escalation | Budget-gated Anthropic API tool for low-confidence/complex tasks | Phase 3 |
| Rubber-duck subagent + blackboard | Parallel speech session; notes-up, read-down, imperative channel, loop-boundary steering | Phase 4 |
| Second-model service (Gemma 4) | Conversational register / second opinion (with second GPU) | Phase 4 |
| Web interface + auth | Chat, uploads, project management, sessions, row-level-security scoping | Phase 5 |
| Digest/scrape crons | Scheduled ingestion → goal-cross-referenced digests | Phase 6+ |
| CV pipeline | Real-time object recognition/tracking | Phase 6+ |

## Infrastructure and quality

| Subsystem | Purpose | Phase |
|---|---|---|
| Serving (vLLM) | OpenAI-compatible endpoint; Qwen3.8-27B Int4; prefix caching as measured optimization | Phase 1 |
| Small-model serving | Router/utility models (4B-class) + embedding model + reranker | Phase 1-2 |
| Eval harness | Seeded question sets, behavioral checks, latency timing; reads trace records | Phase 1 |
| Observability | One structured JSON trace per turn: stage timings, retrievals, FSM state, tokens | Phase 1 |
| Scheduler | Idle-window arbitration for background jobs and GPU budget | Phase 3 |
| Backup/restore | System-of-record dumps from day one | Phase 1 |
| Security & sandboxing | Tool container isolation; MCP scopes; web-pass hardening | Phase 2, 5 |
