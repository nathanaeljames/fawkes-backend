# Fawkes Session Handoff
A living document for resuming work in any new conversation window. Update at the end of any session that changes state. Companion documents: Wishlist_v2, Guiding_Principles, Systems_Inventory, Implementation_Plan, Toolbox.

## Current state (as of 2026-08-31)

- **Iteration 1** (Rasa-based) complete: enrollment, speaker recognition, voice cloning, refactored `server03f.py` class architecture. Preserved as-is; repo fork point marks it as a resume artifact.
- **Iteration 2** (LLM + statechart) in final planning; **no code written yet**. Research round complete; architecture settled; document set created.
- Hardware: one RTX 3090 (speech stack), GTX 1660 Super, second RTX 3090 planned by end of year.
- Next action: **Phase 0 / Milestone Zero** (see Implementation_Plan) — Postgres+pgvector Compose service, migration 001, MemoryStore skeleton, one green pytest.

## Decisions log (major, most recent first)

- 2026-08-31: Primary model **Qwen3.8-27B** (superseding 3.6 target; same VRAM class; vendor benchmarks unverified; thinking budget must be configured — default "xhigh" overthinks). Runtime **vLLM now** (llama.cpp demoted to fallback; GGUF irrelevant on this path). Small-model router fronts the FSM. Postgres SoR from v1 (multi-user mandate). OpenCode adopted for coding (Phase 3); DeepSeek Harness watchlisted. Wiki layer will be **OKF-conformant**. Claude integration = API-key escalation tool + MemoryStore-MCP (Phase 3). Documents set created; standing rule: full revised docs at end of any changing turn.
- 2026-08-31: SimonScrapes review → adopted back-catalog import, per-turn memory-promotion hook, recall-ladder context expansion; validated RLS scoping plan.
- Earlier this round: two-pipeline latency split is structural; verbatim-first storage; bi-temporal facts table; ontology v1 before data; eval harness in Phase 1; tunnels with full lifecycle; deferred risk-based voice auth; no reasoning-retrieval on voice path; adopt-at-edges/build-the-core doctrine; monorepo (frontend + cocktail-party split later); Adi Insights article discredited; Harness-1 supersedes Context-1 on watchlist.
- Iteration-2 handoff doc (project files): Qwen-drives/FSM-constrains statechart; person-centric memory (user_id via ECAPA, no session partitions); FSM registry limited to genuinely multi-step flows.

## Open questions (parked, mostly Phase-4 measurables)

- Qwen3.8 independent benchmark confirmation; Int4 checkpoint availability/quality for vLLM.
- Gemma 4 vs tuned Qwen for voice register (listening test).
- Wiki vs Cog-RAG on the research corpus; hand-rolled loop vs deepagents; Harness-1 subagent value.
- Reranker on the voice path: worth 30-100 ms?
- OpenCode vs DeepSeek Harness maturity at Phase-3 start.
- When bi-temporal SQL → Graphiti (trigger: routine temporal multi-hop or entity-resolution failures).

## Standing rules (recorded in assistant memory as well)

- Never proceed if a referenced file/URL/document is missing — stop and ask.
- Expand uncommon acronyms on first use.
- Multi-user support at every stage; never regresses.
- Full revised project documents delivered at the end of any turn that changes them.
- Patch-style code edits with surrounding-context anchors; no unsolicited refactors; no emoji; Mermaid house colorway.

## How to resume a session

Paste or reference this document, name the phase you're in, and state the immediate goal. If code exists, attach the current file(s) being modified. If the session changes any decision, update the Decisions log here and regenerate affected companion documents.
