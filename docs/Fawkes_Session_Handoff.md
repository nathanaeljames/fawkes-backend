# Fawkes Session Handoff
A living document for resuming work in any new conversation window. Update at the end of any session that changes state. Companion documents: Wishlist_v2, Guiding_Principles, Systems_Inventory, Implementation_Plan, Toolbox.

## Current state (as of 2026-09-06)

- **Iteration 1** (Rasa-based) complete: enrollment, speaker recognition, voice cloning, refactored `server03f.py` class architecture. Keeps running in parallel through Phase 1; retired in Phase 2. Repo history is the resume artifact.
- **Iteration 2** (LLM + statechart) in final planning; **no code written yet**. Research round complete; architecture settled; six-document set current as of this date.
- Hardware: one RTX 3090 (27B in Phase 1; speech stack in Phase 2), GTX 1660 Super (4B router + embedder via llama.cpp), second RTX 3090 planned by end of year.
- Next action: **Phase 0 / Milestone Zero** — Postgres+pgvector Compose service, migration 001, MemoryStore skeleton, one green pytest. Say "go."

## Decisions log (major, most recent first)

- 2026-09-06: Clean-slate ruling refined — back-catalog import (iteration-1 logs, Claude/ChatGPT exports) lands in **Phase 5** after project scopes exist; operational DuckDB migration (speakers/imprints, pangrams, passages) lands in **Phase 2**, after which DuckDB retires. Canary-Qwen-as-router scratched; 4B is the router/arbiter. Rubber-duck interplay v1 built on the single served Qwen in Phase 3; Gemma Phase 4; **heads-down mode** added (Phase 4+). Voice pipeline reads down (status board + deep transcript access); research receives only notes-up/imperatives. Tools bound to slots, not models (Principle 13). Time and elapsed-time injection added to Principle 3. CI/testing strategy adopted: per-commit unit/integration/functional walkthroughs with a shared FakeLLM fixture; nightly GPU eval harness gating exits; test-first for the deterministic core. Phase-1 hardware layout: 27B on the 3090, 4B + embedder on the 1660 Super. SSM ASR swap separated from Cocktail Party (own track); ASR swap likely ~Phase 3.5. Q5_K_S reference disavowed (LLM-generated; no decision weight). Arbiter protocol: asyncio inbox + pending counter + boundary condition wait + watcher cancel; research model resteers itself. Assistant failure recorded: previous turn described document edits without delivering files — corrected this turn with full regeneration.
- 2026-08-31: Primary model **Qwen3.8-27B**; runtime **vLLM now**; small-model router fronts the FSM; Postgres SoR from v1 (multi-user mandate); OpenCode adopted for coding (Phase 3); DeepSeek Harness watchlisted; wiki layer **OKF-conformant**; Claude integration = API-key escalation tool + MemoryStore-MCP (Phase 3); standing rule: full revised docs at end of any changing turn, with an explanation of what changed and why.
- 2026-08-31: SimonScrapes review → adopted per-turn memory-promotion hook and recall-ladder context expansion; validated RLS scoping plan; back-catalog import pattern harvested.
- Earlier this round: two-pipeline latency split is structural; verbatim-first storage; bi-temporal facts table; ontology v1 before data; eval harness in Phase 1; tunnels with full lifecycle; deferred risk-based voice auth; no reasoning-retrieval on voice path; adopt-at-edges/build-the-core; monorepo (frontend + cocktail-party split later); Adi Insights article discredited; Harness-1 supersedes Context-1 on watchlist.
- Iteration-2 handoff doc (project files): router-first tiered execution over the statechart (amends "one Qwen call per turn"); person-centric memory (user_id via ECAPA, no session partitions); FSM registry limited to genuinely multi-step flows.

## Open questions (parked, mostly Phase-4 measurables)

- Qwen3.8 independent benchmark confirmation; Int4 checkpoint availability/quality for vLLM.
- Gemma 4 vs tuned Qwen for voice register (listening test); decorrelated-judgment value in second opinions.
- Wiki vs Cog-RAG on the research corpus; hand-rolled loop vs deepagents; Harness-1 subagent value.
- Reranker on the voice path: worth 30-100 ms?
- OpenCode vs DeepSeek Harness maturity at Phase-3 start; GLM-class coding model for heads-down mode.
- When bi-temporal SQL → Graphiti (trigger: routine temporal multi-hop or entity-resolution failures).
- Exact timing of the SSM ASR swap (~3.5) and which Mamba-class checkpoints are production-ready.

## Standing rules (recorded in assistant memory as well)

- Never proceed if a referenced file/URL/document is missing — stop and ask.
- Expand uncommon acronyms on first use.
- Multi-user support at every stage; never regresses.
- Any input meriting a change to project files returns the full revised file(s) in the same turn, with an explanation of exactly what was edited and why.
- Patch-style code edits with surrounding-context anchors; no unsolicited refactors; no emoji; Mermaid house colorway.

## How to resume a session

Paste or reference this document, name the phase you're in, and state the immediate goal. If code exists, attach the current file(s) being modified. If the session changes any decision, update the Decisions log here and regenerate affected companion documents.
