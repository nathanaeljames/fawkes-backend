# Fawkes Systems Inventory
All subsystems, what each does, and the phase in which it is built. Phases refer to Fawkes_Implementation_Plan.md. "Carry-over" = exists from Iteration 1 and is retained. Updated 2026-09-06.

## Speech and identity (carry-over, evolving)

| Subsystem | Purpose | Phase |
|---|---|---|
| Audio I/O + WebSocket layer | Streaming capture/playback, multi-client connection state | Carry-over |
| VAD (MarbleNet) + endpointing | Speech detection, utterance finalization, first-line noise filtering | Carry-over |
| Interim ASR (FastConformer) | Streaming partial transcripts; SSM replacement likely ~Phase 3.5 | Carry-over |
| Final ASR (Canary-Qwen) | Final utterance transcription; SSM replacement likely ~Phase 3.5 | Carry-over |
| Speaker identity (ECAPA-TDNN) | In-memory embedding matrix, per-second matching, audio → user_id bridge | Carry-over |
| TTS (XTTS / Piper) + stream manager | Voice output, cloning, sequential ordering, feedback prevention | Carry-over |
| Identity & auth subsystem | Deferred identification tiers, recency-modulated thresholds, passphrase/auth FSM, retroactive turn attribution | Phase 2 |
| Operational data migration | Move iteration-1 DuckDB tables (speakers/imprints, pangrams, passages) into Postgres; retire DuckDB | Phase 2 |

## Cognition

| Subsystem | Purpose | Phase |
|---|---|---|
| Prompt builder | State-conditioned prompt assembly; cache-stable prefix ordering; layer injection; current-time and elapsed-time injection | Phase 1 |
| FSM registry + validator | Statechart definitions (enrollment, voice clone, auth, clarification); deterministic transition legality; slot schemas | Phase 1-2 |
| Small-model router (4B) | Intent recognition, noise/backchannel gating ("ignore" class), slot capture, confidence-based escalation; also the mid-flight arbiter | Phase 1-2 |
| Structured-output contract | JSON schema for slot updates / transitions / tool calls / response_text; constrained decoding | Phase 1 |
| Tool registry + executor | Opaque, slot-bound tool surface for whatever model occupies a pipeline; sandboxed (Docker) execution | Phase 2 |
| Correction/event bus + arbiter | Queue injecting background-verification results and cross-pipeline imperatives at loop boundaries; arbiter triage {discard, queue, interrupt}; out-of-band cancel for interrupts | Phase 2 (hook), Phase 3 (arbiter) |
| Voice register controls | Thinking off / budgeted thinking, brevity persona, stall-word authorization | Phase 2 |

## Memory substrate

| Subsystem | Purpose | Phase |
|---|---|---|
| MemoryStore (Postgres + pgvector + FTS) | System of record: verbatim transcripts, documents, bi-temporal facts; backend-agnostic interface | Phase 1 |
| Ontology + schema constraints | Versioned entity/relation/facet vocabulary; enum tables; runtime enforcement; contradiction comparability | Phase 1 |
| ingest() contract | Single write path: hashing, idempotency, provenance, per-store fan-out handlers | Phase 1 |
| Hybrid retrieval | BM25 + vector, Reciprocal Rank Fusion, scope weighting; optional reranker | Phase 1-2 |
| Semantic cache | Query-similarity answer/retrieval cache with invalidation hooks | Phase 2 |
| Tiered context loader | Layers 0-3 for the voice path | Phase 2 |
| Memory-promotion hook | Per-turn small-model judgment: promote durable facts to standing context/facts table | Phase 2 |
| Recall ladder | Standing context → hybrid → rerank → context expansion → cited synthesis or honest gap | Phase 3 |
| Consolidation & compaction jobs | Idle-time summaries-with-manifests, archival with stubs; long-session compaction; research-job checkpoints | Phase 3 |
| Contradiction detector | Ingest-time fact-conflict check; severity-gated log or clarification FSM | Phase 3-4 |
| Tunnel manager | Cross-project scope weights: propose/attribute/reinforce/decay/destroy | Phase 3-4 |
| Wiki distillation layer (OKF-conformant) | Per-project compiled knowledge bundles, lifecycle states, lint | Phase 4 |
| Reconciliation jobs | Cross-store agreement sampling, drift alerts, type histograms | Phase 4 |
| History importer | Back-catalog ingestion (iteration-1 logs, Claude/ChatGPT exports) with project assignment; after project scopes exist | Phase 5 |

## Pipelines and surfaces

| Subsystem | Purpose | Phase |
|---|---|---|
| Voice pipeline (latency-critical) | The end-to-end conversational loop; deterministic + single-shot retrieval only | Phase 1-2 |
| Ingestion router | Structure scoring; route to trees / hybrid / lazy-only; Qwen-vision OCR at ingest | Phase 3 |
| Research pipeline | Agentic loop, CRAG grading, web fallback, multi-hop, parallel fan-out; checkpoints progress for crash recovery | Phase 3 |
| Coding surface (OpenCode) | Adopted coding harness on local endpoint; transcripts into MemoryStore | Phase 3 |
| MCP server | MemoryStore (and tools) exposed to external agent hosts with scoped auth + audit | Phase 3 |
| Claude escalation | Budget-gated Anthropic API tool for low-confidence/complex tasks | Phase 3 |
| Rubber-duck interplay | Shared blackboard; notes-up; read-down = status board (research summary refreshed per loop boundary into voice Layer 1) + deep access to the full research transcript via voice Layer 3 tools; imperative channel; arbiter-gated queue/interrupt; research model resteers itself. v1 on the single served Qwen | Phase 3 |
| Second-model service (Gemma 4) | Conversational register + decorrelated second opinion (with second GPU) | Phase 4 |
| Heads-down mode | Task-mode slot swap: conversational model's GPU slot given to a second coding model or Harness-1; voice falls back to a Qwen register | Phase 4+ |
| Web interface + auth | Chat, uploads, project management, sessions, row-level-security scoping | Phase 5 |
| Digest/scrape crons | Scheduled ingestion → goal-cross-referenced digests | Phase 6+ |
| CV pipeline | Real-time object recognition/tracking | Phase 6+ |
| SSM ASR swap | Mamba-class replacement for interim and final ASR | Phase 6+ (likely ~3.5) |
| Cocktail Party separation | Speaker-separation research; separate repository and track | Separate track |

## Infrastructure and quality

| Subsystem | Purpose | Phase |
|---|---|---|
| Serving (vLLM) | OpenAI-compatible endpoint; Qwen3.8-27B Int4 on the RTX 3090; prefix caching as measured optimization | Phase 1 |
| Utility-model serving (llama.cpp on GTX 1660 Super) | 4B router + embedding model until the second 3090; reranker | Phase 1-2 |
| Eval harness | Seeded question sets, behavioral checks, latency timing; reads trace records; nightly/on-demand; gates phase exits | Phase 1 |
| CI pipeline (GitHub Actions) | Per-commit unit + integration + mocked-LLM FSM lifecycle functional tests; phase exit tests as permanent markers; shared FakeLLM fixture | Phase 1 |
| Observability | One structured JSON trace per turn: stage timings, retrievals, FSM state, tokens | Phase 1 |
| Scheduler | Idle-window arbitration for background jobs and GPU budget | Phase 3 |
| Backup/restore | System-of-record dumps from day one | Phase 1 |
| Security & sandboxing | Tool container isolation; MCP scopes; web-pass hardening | Phase 2, 5 |
