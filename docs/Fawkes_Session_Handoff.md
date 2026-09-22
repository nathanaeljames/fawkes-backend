# Fawkes Session Handoff
A living document for resuming work in any new conversation window. Update at the end of any session that changes state. Companion documents: Wishlist_v2, Guiding_Principles, Systems_Inventory, Implementation_Plan, Toolbox, Architecture_Specification (v1.1), README_draft.

## Current state (as of 2026-09-19)

- **Iteration 1** (Rasa-based) complete; no conversation logging exists in iteration 1. Keeps running in parallel through Phase 1; retired in Phase 2.
- **Iteration 2** (LLM + statechart) in final planning; **no code written yet**. Architecture Specification v1.1 is the document for independent multi-model review before Phase 0.
- Hardware: one RTX 3090; second RTX 3090 planned by end of year (one model per card by default).
- Next action: independent review of the Architecture Specification, then **Phase 0 / Milestone Zero** — now including the silo layer in migration 001. Say "go."

## Decisions log (major, most recent first)

- 2026-09-19: **Silo layer** adopted into migration 001 (silos table, non-null silo_id everywhere, composite project keys, single-silo tunnels, per-silo content hash, list partitioning with one default partition, forced RLS under a non-owner role, immutable silo context, identity outside silos with membership + default silo, export-and-re-ingest crossings, permanent leakage tests). Three mechanisms kept distinct: silos (hard), scopes/tunnels (soft), RLS (authorization). **Schema-first, analytics-later** principle recorded. **Response modality follows input modality** (voice in/voice out; text in/text out; background results return through the originating channel). Verdict vocabulary unified: voice {ignore | FSM event | escalate}; voice-slot powers {answer | note | start | queue | interrupt}; text {start | queue | interrupt | FSM command}. Illegal proposals return to their proposer with a hint, bounded, then dropped and traced; text-channel rejections never reach the voice slot. Imperatives go directly to the research command queue; only notes go to the blackboard. Browser/terminal ingestion adapters deferred to Phase 6+ (the motivating goal belongs to the KVM/HID track); file drops and coding transcripts stay in Phase 3. Capability-card process specified (nightly versioned rows per model/quant/config; router and bridge read them). Serving-recipe experiment scheduled in Phase 1 (stock vLLM vs tuned vLLM fork vs tuned llama.cpp fork); two-card default is one model per card, tensor parallelism gated on verified NVLink/x16/PSU and measured gain; drafters are per-target-model (no public DFlash2 for Gemma). Compression tooling deferred indefinitely. Architecture Specification promoted to a permanent project document. **Public-documents hygiene** rule adopted: no secrets, hostnames, employer names, or third-party names; deployment defaults moved out of docs; secret scanning on push; the silo design note stays out of the public repo. Master diagram v3: subgraphs per pipeline, unified labels, text-response endpoint, voice-slot → apply → TTS path, re-prompt node, single Postgres cylinder plus file-based derived cylinder, status-board edge solid, escalate label, README embeds an exported image with source in docs/diagrams.
- 2026-09-15: GTX 1660 Super removed; Phase 1 on one 3090 with split-mode fallback; input routing doctrine fixed; external API bridge (Phase 3) with OCR-first payloads; token/cost accounting in traces; panel mode and code-graph MCP added; back-catalog = Claude/ChatGPT exports only (Phase 5); DuckDB retires after Phase 2; UTC timestamptz with per-session time zones; screen-capture conversation harvested; diagram color semantics assigned.
- 2026-09-06: Clean-slate ruling; operational DuckDB migration (Phase 2); Canary-Qwen-as-router scratched; rubber-duck v1 on single served Qwen (Phase 3), Gemma Phase 4, heads-down mode; tools bound to slots; time injection; CI/testing strategy; SSM ASR swap separated from Cocktail Party; arbiter protocol.
- 2026-08-31: Qwen3.8-27B primary; vLLM now; small-model router fronts the FSM; Postgres SoR from v1; OpenCode adopted (Phase 3); wiki OKF-conformant; Claude integration = API-key escalation + MemoryStore-MCP (Phase 3); SimonScrapes patterns harvested.
- Earlier: two-pipeline latency split is structural; verbatim-first; bi-temporal facts; ontology v1 before data; eval harness in Phase 1; tunnels lifecycle; deferred risk-based voice auth; adopt-at-edges/build-the-core; monorepo; Adi Insights discredited; Harness-1 supersedes Context-1.

## Open questions (parked, mostly measurables)

- Single-card Phase 1 co-residency in practice; which serving recipe wins per role.
- Identity-outside-silos placement and project resolution from voice (settle before the enrollment port).
- Gemma 4 vs tuned Qwen for voice register; decorrelated-judgment value in panel mode.
- Routing thresholds for the external API bridge (calibration of the confidence field against capability cards).
- Wiki vs Cog-RAG; hand-rolled loop vs deepagents; Harness-1; code-graph MCP vs grep-only; tensor-parallel vs one-model-per-card.
- OpenCode vs DeepSeek Harness maturity at Phase-3 start; GLM-class coding model for heads-down mode.
- When bi-temporal SQL → Graphiti; when the SSM ASR swap (~3.5).

## Standing rules (recorded in assistant memory as well)

- Never proceed if a referenced file/URL/document is missing — stop and ask.
- Expand uncommon acronyms on first use.
- Multi-user support at every stage; never regresses. Silo isolation never regresses.
- Any input meriting a change to project files returns the full revised file(s) in the same turn, with an explanation of exactly what was edited and why.
- Project documents are publishable as written: no secrets, hostnames, employer names, or third-party personal names.
- Patch-style code edits with surrounding-context anchors; no unsolicited refactors; no emoji; Mermaid house colorway with the semantics recorded in the toolbox.

## How to resume a session

Paste or reference this document, name the phase you're in, and state the immediate goal. If code exists, attach the current file(s) being modified. If the session changes any decision, update the Decisions log here and regenerate affected companion documents.
