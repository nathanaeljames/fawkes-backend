# Fawkes Session Handoff
A living document for resuming work in any new conversation window. Update at the end of any session that changes state. Companion documents: Wishlist_v2, Guiding_Principles, Systems_Inventory, Implementation_Plan, Toolbox, Architecture_Specification (v1.3), README_draft, diagram sources (`architecture_L4.mmd`, `_L7.mmd`, `_L10.mmd`).

## Current state (as of 2026-09-21)

- **Iteration 1** (Rasa-based) complete; no conversation logging exists. Runs in parallel through Phase 1; retired in Phase 2.
- **Iteration 2** in final planning; **no code written yet**. Architecture Specification v1.2 and the three-level diagram set are the review artifacts for independent multi-model review (in progress with a second model).
- Hardware: one RTX 3090; second RTX 3090 planned by end of year (one model per card by default).
- Next action: answer Q01-Q06 of the annotated review worksheet (Phase 0 blockers), fold the answers into the specification, then **Phase 0 / Milestone Zero** with silo Tier A in migration 001. Say "go."

## Decisions log (major, most recent first)

- 2026-09-21: Silo Tier A confirmed by owner ("cheap now, expensive to refactor later"). **Evaluation cadence differentiated**: per-commit CI; nightly ~10-case smoke; full card run on fingerprint change and weekly; coding-patch slice weekly from Phase 3 (hand-authored micro-repositories); remote tiers per release plus monthly capped spot-check, research-tier slices only — never routing or recall. Slice defined as a task family with its own seed cases and scoring rule; models carded only on the slices their slot serves. **Feedback capture** adopted schema-first (`feedback_events` from Phase 1; voice capture Phase 2; web thumbs Phase 5). **Panel results** recorded to `panel_results` and summarized onto cards as uncontrolled field observations. Conversation-trajectory ("spiral-in") analytics parked as far-future research; raw material collected now. Specification v1.3. **Independent review worksheet** (twelve open questions, Q01-Q12) received from the second reviewer and annotated with Fable opinions in Fawkes_Decisions_To_Resolve_annotated.md; owner answers pending; nothing from the worksheet is amended into the specification until answered. Document-regeneration rule tightened: only files that change are regenerated.
- 2026-09-20: **Silo layer tiered.** Tier A in migration 001 (silo key on every row and in every unique constraint and index prefix, composite project references, single-silo tunnels, per-silo hash, silo context, leakage tests — about an hour, no runtime cost). Tier B deferred to the first real second silo (partitioning, silo-keyed forced RLS, per-silo roles, instance pin — real friction, no value at one silo); user-keyed RLS still lands in Phase 5. Nothing retracted. **Capability-card measurement** specified: deterministic slice scoring (labeled verdicts, seeded recall, fixed-corpus citations, failing-test repositories for patches, schema validation, loop/stop/latency), fixed seed set, runs keyed by a config fingerprint and skipped when unchanged, several sampling seeds, usage corrections become labeled cases, remote paid tiers carded per release with a monthly capped spot-check; panel mode is a runtime feature, not the eval. **Code graphs:** adopt both — code-review-graph for blast radius (priority if only one), graphify for the multimodal bird's-eye view. **Voice-slot FSM proposals** clarified as a context-richer proposer plus safety net; router remains primary. Labels restored: blackboard "notes-up, read-down, imperatives" and correction edge "research results / self-correction, voice-initiated tasks." **Diagram complexity levels** defined (1-10); v3 rated level 7; delivered L10 (exhaustive), L7 (engineering), L4 (README).
- 2026-09-19: Silo layer adopted; schema-first principle; modality rule; verdict vocabulary unified; illegal proposals return to proposer bounded; adapters deferred to 6+; capability cards introduced; serving-recipe experiment; one model per card default; compression deferred indefinitely; specification promoted to permanent; public-documents hygiene rule; diagram v3.
- 2026-09-15: GTX 1660 Super removed; routing doctrine fixed; external API bridge; token/cost accounting; panel mode and code-graph MCP added; back-catalog = Claude/ChatGPT only; DuckDB retires after Phase 2; UTC timestamptz with per-session time zones; screen-capture conversation harvested; diagram color semantics.
- 2026-09-06: Clean-slate ruling; DuckDB migration (Phase 2); Canary-as-router scratched; rubber-duck v1 on single Qwen (Phase 3), Gemma Phase 4, heads-down mode; tools bound to slots; time injection; CI/testing strategy; SSM ASR separated from Cocktail Party; arbiter protocol.
- 2026-08-31: Qwen3.8-27B primary; vLLM now; small-model router fronts the FSM; Postgres SoR; OpenCode adopted; wiki OKF-conformant; Claude integration = API-key escalation + MemoryStore-MCP.
- Earlier: two-pipeline latency split is structural; verbatim-first; bi-temporal facts; ontology v1 before data; eval harness in Phase 1; tunnels lifecycle; deferred risk-based voice auth; adopt-at-edges/build-the-core; monorepo; Adi Insights discredited; Harness-1 supersedes Context-1.

## Open questions (parked, mostly measurables)

- Single-card Phase 1 co-residency; which serving recipe wins per role.
- Identity-outside-silos and project resolution from voice (settle before the enrollment port).
- Exact feature boundary of code-review-graph vs graphify at Phase 3 (both move fast).
- Gemma 4 vs tuned Qwen; decorrelated-judgment value in panel mode.
- Escalation thresholds calibrated from capability cards.
- Wiki vs Cog-RAG; hand-rolled loop vs deepagents; Harness-1; code-graph vs grep-only; tensor-parallel vs one-model-per-card.
- OpenCode vs DeepSeek Harness maturity at Phase-3 start; GLM-class coding model.
- When bi-temporal SQL → Graphiti; when the SSM ASR swap (~3.5).

## Standing rules (recorded in assistant memory as well)

- Never proceed if a referenced file/URL/document is missing — stop and ask.
- Expand uncommon acronyms on first use.
- Multi-user support at every stage; never regresses. Silo isolation never regresses.
- Any input meriting a change to project files returns the full revised file(s) in the same turn, with an explanation of exactly what was edited and why.
- Project documents are publishable as written.
- Patch-style code edits with surrounding-context anchors; no unsolicited refactors; no emoji; Mermaid house colorway with the semantics and complexity levels recorded in the toolbox.

## How to resume a session

Paste or reference this document, name the phase you're in, and state the immediate goal. If code exists, attach the current file(s) being modified. If the session changes any decision, update the Decisions log here and regenerate affected companion documents.
