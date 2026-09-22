# Fawkes Guiding Principles
The manifesto. Every design decision should be defensible against this document; if a decision contradicts it, either the decision or the document must change — explicitly. Updated 2026-09-19 (unchanged 2026-09-20).

## Memory and truth

1. **One system of record.** A single ACID store (Postgres) is authoritative. Every other structure — indexes, vectors, trees, wiki pages, caches — is derived, rebuildable, and permitted to be temporarily wrong. The source of record is not.
2. **Verbatim first.** Exact text, timestamped, attributed to a resolved speaker, preserved forever. Synthesis (summaries, wiki pages, themes) is an overlay that always links back to sources and never replaces them.
3. **Time is not optional.** Every turn, fact, resource, and modification carries a full date-time, stored as UTC with time-zone-aware types; every device and session registers its time zone, and the prompt builder renders local time and the elapsed time since the previous turn. Facts are bi-temporal (when true, when learned). Temporal reasoning is a first-class capability.
4. **One write path.** All memory enters through `ingest()`: hashed, idempotent, provenance-stamped. Derived stores fan out from it; nothing writes around it.
5. **Transformations leave receipts.** Any compression, summarization, or archival writes a manifest. Every retrieval records its trace; every model call records its tokens and cost. Nothing is silently rewritten.
6. **Epistemic hygiene.** Answers about the past cite their sources. When memory holds no answer, Fawkes says so instead of confabulating.

## Architecture

7. **Latency class determines retrieval class.** The voice loop gets deterministic lookups, cached answers, and single-shot hybrid search — never an extra LLM round-trip. Reasoning-driven retrieval belongs to the research pipeline.
8. **Structure beats similarity wherever structure exists.** SQL for facts, trees for documents, syntax trees and code graphs for code, scoped hierarchy for projects. Embedding similarity handles the unstructured residue.
9. **Lazy answers now, structure in idle time.** Every query is answerable immediately via the lazy path. Ingestion queues background indexing for idle windows. Volatile sources never carry stale indexes.
10. **The LLM proposes; deterministic code ratifies.** A rejected proposal goes back to its proposer with the reason, a bounded number of times. Never a closed-set classifier gating conversation; never an unchecked LLM mutating state.
11. **Deterministic below, interpretive above.** Context loads by tier without an LLM; deep search is LLM-invoked. Small models handle routing, triage, extraction, and background jobs; large models handle cognition and their own resteering. Escalation valves everywhere; no misroute is irreversible.
12. **Person-centric, multi-user, always — and partitioned when it matters.** Memory is organized by resolved speaker identity, never by session. Identity resolution precedes personalization; authentication precedes sensitive action. Three distinct mechanisms, never conflated: silos are hard partitions that nothing crosses; project scopes and tunnels are soft ranking weights inside a silo; row-level security is per-user authorization inside a silo. Modality follows the channel: voice in, voice out; text in, text out.

## Engineering discipline

13. **Boundaries are the durable decisions.** The OpenAI-compatible endpoint decouples product from runtime. The MemoryStore interface decouples logic from storage. MCP decouples Fawkes from external agent hosts. Tools are bound to pipeline slots, never to a model's identity.
14. **Measure, don't debate.** Model choice, quant level, serving recipe, framework adoption, retrieval strategy, routing thresholds — benchmarks on our own hardware and data. Capability cards are written from eval results, never from vendor tables.
15. **The harness decides whether the model feels smart.** Chat template correctness, tool-call parsing, thinking-token handling, compaction, sandbox permissions, retry behavior, stop conditions — quality lives in these details.
16. **Adopt at the edges, build the core.** The voice statechart, memory substrate, and identity system are built and owned. Coding harnesses, code graphs, temporal graph engines, serving runtimes — adopted, behind seams, with their own storage where they bring it.
17. **No silent rework, and no silent regression.** Successive passes that build on each other beat big-bang rewrites. Monolith-first. The deterministic core is developed test-first, and every phase's exit test remains in the suite forever.
18. **Schema first, analytics later.** Data that a later phase will need is collected from the first phase that can produce it: silo and project keys, token counts and cost, bi-temporal fields, salience components, trace records. Not every wishlist item ships in each pass, but the infrastructure never forecloses one.
19. **Stop and ask.** Missing files, absent context, ambiguous references — halt and request rather than guess.
20. **Local by default, text outward.** Everything runs on owned hardware. Cloud calls are explicit, budget-gated, logged, and carry extracted text, not documents or images, unless visual judgment is the task.
21. **Public by construction.** Project documents are written to be published: no secrets, hostnames, credentials, employer names, or third-party personal names; deployment specifics live in configuration. Secret scanning runs before every push.
22. **24/7 with grace.** Fawkes is always available; heavy work schedules itself into idle windows.
