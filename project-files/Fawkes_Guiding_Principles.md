# Fawkes Guiding Principles
The manifesto. Every design decision should be defensible against this document; if a decision contradicts it, either the decision or the document must change — explicitly.

## Memory and truth

1. **One system of record.** A single ACID store (Postgres) is authoritative. Every other structure — indexes, vectors, trees, wiki pages, caches — is derived, rebuildable, and permitted to be temporarily wrong. The source of record is not.
2. **Verbatim first.** Exact text, timestamped, attributed to a resolved speaker, preserved forever. Synthesis (summaries, wiki pages, themes) is an overlay that always links back to sources and never replaces them.
3. **Time is not optional.** Every turn, fact, resource, and modification carries timestamps. Facts are bi-temporal (when true, when learned). Temporal reasoning is a first-class capability, not an afterthought.
4. **One write path.** All memory enters through `ingest()`: hashed, idempotent, provenance-stamped. Derived stores fan out from it; nothing writes around it.
5. **Transformations leave receipts.** Any compression, summarization, or archival writes a manifest (what was removed, what was preserved, why, safe uses). Every retrieval records its trace. Nothing is silently rewritten. Memory is versioned, auditable, and restorable.
6. **Epistemic hygiene.** Answers about the past cite their sources — the conversation, the line, the date. When memory holds no answer, Fawkes says so instead of confabulating. Provenance beats fluency.

## Architecture

7. **Latency class determines retrieval class.** The voice loop gets deterministic lookups, cached answers, and single-shot hybrid search — never an extra LLM round-trip. Reasoning-driven retrieval (traversal, multi-hop, agentic loops) belongs to the research pipeline. This split is structural, not an optimization.
8. **Structure beats similarity wherever structure exists.** SQL for facts, trees for documents, syntax trees for code, scoped hierarchy for projects. Embedding similarity handles the unstructured residue — it is a tool, never the foundation.
9. **Lazy answers now, structure in idle time.** Every query is answerable immediately via the lazy path (search over raw stored text). Ingestion queues background indexing (trees, wiki, facts) for idle windows. Volatile sources are flagged lazy-only and never carry stale indexes.
10. **The LLM proposes; deterministic code ratifies.** Interpretive judgment routes and drafts; validators, schemas, and constraints enforce legality. Never a closed-set classifier gating conversation; never an unchecked LLM mutating state.
11. **Deterministic below, interpretive above.** Context loads by tier: identity-keyed and topic-keyed layers load without an LLM; deep search is LLM-invoked. Small models handle routing, extraction, and background jobs; large models handle cognition. Escalation valves everywhere; no misroute is irreversible.
12. **Person-centric, multi-user, always.** Memory is organized by resolved speaker identity across all sessions and devices — never by session. Identity resolution precedes personalization; authentication precedes sensitive action; scopes isolate by default, with deliberate, weighted, attributed tunnels between them.

## Engineering discipline

13. **Boundaries are the durable decisions.** Components swap; seams persist. The OpenAI-compatible endpoint decouples product from runtime. The MemoryStore interface decouples logic from storage. MCP decouples Fawkes services from external agent hosts. Invest in seams, not in marriages to components.
14. **Measure, don't debate.** Model choice, quant level, framework adoption, retrieval strategy — these are benchmarks on our own hardware and data, not opinions. The eval harness is a component of the system. A question that can be measured in an afternoon does not get argued for a week.
15. **The harness decides whether the model feels smart.** Chat template correctness, tool-call parsing, thinking-token handling, compaction strategy, sandbox permissions, retry behavior, stop conditions — quality lives in these details. Harness failures are never misdiagnosed as model failures.
16. **Adopt at the edges, build the core.** The voice statechart, memory substrate, and identity system are Fawkes's differentiation — built and owned. Coding harnesses, temporal graph engines, serving runtimes — adopted, behind seams. Patterns are harvested from open source; codebases are adopted only whole and only at the edges.
17. **No silent rework.** Plan enough to avoid tear-downs; when work must be wasted, waste it at the simple stages. Successive passes that build on each other beat big-bang rewrites. Monolith-first: extract modules when boundaries have proven themselves.
18. **Stop and ask.** Missing files, absent context, ambiguous references — halt and request rather than guess. Applies to Fawkes's agents as much as to its developer's assistants.
19. **Local by default.** Everything runs on owned hardware. Cloud calls (Claude escalation) are explicit, budget-gated, and logged. Privacy is a property of the architecture, not a setting.
20. **24/7 with grace.** Fawkes is always available; heavy work schedules itself into idle windows; the GPU works when asked and rests when not.
