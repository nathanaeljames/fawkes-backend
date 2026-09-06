# Fawkes Implementation Plan
Iteration 2 (the LLM + statechart rebuild), structured as phases. Each phase has a goal, contents, and an exit test. Phases are intentionally re-orderable at their edges; the plan is a living document — revise it, don't obey it.

**Model/runtime baseline (current):** Qwen3.8-27B (Int4/AWQ-class) on vLLM (>= 0.19-class), OpenAI-compatible endpoint, thinking budget configured per pipeline (low/off for voice, high for research). Small utility model (Qwen 4B-class) + embedding model (bge-m3-class) + optional reranker served alongside. GGUF/llama.cpp retained only as a fallback path and for tiny models — not the production runtime.

## Phase 0 — Milestone Zero (one afternoon)
Goal: first running code of Iteration 2; break the no-code streak.
- Add `postgres` (with pgvector) service to docker-compose.
- Migration 001: `transcripts`, `facts` (bi-temporal), `documents` tables + FTS + vector indexes + enum seed tables.
- `MemoryStore` skeleton: `remember_utterance()`, `recall()` (hybrid BM25+vector, Reciprocal Rank Fusion).
- One pytest: insert three utterances, retrieve by meaning and by keyword, assert timestamps and provenance fields.
Exit: `pytest` green; a transcript survives a container restart.

## Phase 1 — Cognitive core + substrate foundation
Goal: the new brain exists; the old Rasa functionality is reachable through it in text.
- Prompt builder + validator skeleton; structured-output JSON contract; FSM registry data structures.
- Small-model router v1: {advance-FSM event | freeform | ignore/backchannel | escalate} with confidence; wired before the 27B.
- vLLM serving stood up (both models); prompt prefix ordered for cache stability.
- Ontology v1 (`ontology.md` + enum tables + constraints).
- `ingest()` v1 with content hashing and provenance.
- Eval harness v0 (~30 questions + behavioral checks + timing) reading trace records; observability schema (one JSON trace per turn).
- Backup cron for Postgres.
Exit: text-mode conversation through the statechart passes the eval set's FSM and recall slices; every turn produces a trace record.

## Phase 2 — Voice memory + identity (voice is primary)
Goal: full voice loop on the new core, multi-user, with real memory.
- Layers 0-2 tiered loading; semantic cache; Layer-3 tool calls (remember/recall/single-shot lookups).
- Enrollment FSM ported; voice-clone FSM ported; auth FSM (deferred identification tiers, recency-modulated thresholds, passphrase).
- Voice register: thinking off/budgeted, brevity persona, stall words; correction-event hook (queue + prompt-builder input).
- Memory-promotion hook (small model) after each turn.
- Sandboxed tool container.
- History importer v1 (iteration-1 logs at minimum).
Exit: two different speakers hold personalized conversations on separate devices; enrollment completes end-to-end; a fact stored yesterday is recalled with citation today; voice-path pre-LLM overhead measured under ~150 ms.

## Phase 3 — Ingestion, research pipeline, external hands
Goal: Fawkes reads, researches, codes, and talks to the outside world.
- Ingestion router (structure scoring; Qwen-vision OCR at ingest); document-summary selection layer; PageIndex-style trees (checksum-cached).
- Research pipeline v1: agentic loop with grep/FTS/tree/web tools, CRAG grading, web fallback; recall ladder with context expansion and citations.
- Consolidation/compaction jobs + scheduler (idle-window arbitration).
- OpenCode integrated as coding surface on the local endpoint; its transcripts ingested.
- MCP server over MemoryStore (scoped auth, audit log) — consumed by OpenCode; usable by Claude.
- Claude escalation tool (Anthropic API key, budget-gated, logged).
- Salience weights v1; contradiction detector v1 (log-level).
- History importer extended to Claude/ChatGPT exports.
Exit: a multi-document research question answered with correct citations; a coding task completed in OpenCode with Fawkes memory available via MCP; one Claude escalation round-trip.

## Phase 4 — Synthesis layer + measured experiments (second GPU era)
Goal: compounding knowledge; the dual-model architecture; let the eval harness adjudicate deferred debates.
- Wiki distillation layer, OKF-conformant bundles, lifecycle states, lint cron.
- Tunnel manager full lifecycle; contradiction detector escalation to clarification FSM.
- Gemma 4 stood up on the second RTX 3090; rubber-duck subagent with blackboard (notes-up, read-down, imperative channel, loop-boundary steering of the coding/research agent).
- Reconciliation crons; type histograms.
- Benchmarks: wiki vs Cog-RAG (official repo, local endpoint); hand-rolled research loop vs deepagents; Harness-1 as retrieval subagent (hardware permitting); Qwen-vs-Gemma voice register listening tests.
Exit: wiki answers a corpus-theme question from compiled pages with source descent; one benchmark decision recorded in the toolbox with data.

## Phase 5 — Web interface + multi-surface
Goal: Fawkes beyond the microphone.
- Web app: chat, uploads, project creation/notes, authentication/sessions; security hardening pass (this is an attack surface: treat it like one).
- Row-level security scoping for additional users; per-project visibility.
- Voice-driven project management commands.
- Mobile-facing API; remaining MCP surface; Claude handoff packaging (context bundles; claude.ai Projects have no public API).
Exit: a project created by voice is visible and editable on the web by an authenticated user; a second user sees only their scope.

## Phase 6+ — Frontier (planned, not scheduled)
- SSM/Mamba ASR swap (interim + final); Cocktail Party integration.
- Digest/scrape crons; Microsoft To Do; Maps/Waze; home automation; CV pipeline.
- Graphiti if bi-temporal SQL hits its ceiling; serving scale-out; power management tuning; fine-tuning experiments only on demonstrated need.

## Standing constraints (all phases)
- Multi-user support never regresses. 
- No reasoning-driven retrieval on the voice hot path.
- Any memory transformation writes a manifest; any retrieval writes a trace.
- Every adopted component sits behind a seam (OpenAI-compatible endpoint, MemoryStore interface, MCP).
- Document set (wishlist, principles, inventory, plan, toolbox, handoff) updated in full whenever a decision changes it.
