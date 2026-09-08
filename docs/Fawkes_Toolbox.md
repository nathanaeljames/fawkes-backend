# Fawkes Toolbox
Every tool, model, and framework vetted across this research round, sorted into: **Adopted** (committed), **Watchlist** (possible future / experiment first), **Pass** (not a fit). A final section records how we appropriate tools — the personal decisions and patterns that are ours. Last updated 2026-09-06.

## Adopted — critical path

| Tool | Role | Note |
|---|---|---|
| Qwen3.8-27B (Apache 2.0) | Primary cognition, vision/OCR (interactive and ingest-time) | Dense 27B, 262K ctx, vision+video; Int4 fits a 3090; default reasoning "xhigh" — set thinking budgets per pipeline; vendor benchmarks pending independent confirmation |
| vLLM | Serving runtime for the 27B | OpenAI-compatible endpoint, concurrency-class serving, official Qwen recipes; prefix caching over linear-attention layers experimental — measure |
| llama.cpp (server) | Utility-model host on the GTX 1660 Super (4B router, embedder) until the second 3090; fallback runtime | GGUF Q4 runs well on Turing; not the 27B's runtime |
| Qwen 4B-class small model | Router, mid-flight arbiter, extraction, memory-promotion hook, background jobs | Confidence-gated escalation to 27B; upgrade to a larger utility model only if it fails the routing eval slice |
| bge-m3-class embedding model | Embeddings for pgvector | Local, ~1.2 GB |
| bge-reranker-v2-m3-class | Optional rerank stage (research path; voice only if measured cheap) | Sub-1B cross-encoder |
| Postgres + pgvector + native FTS | System of record + hybrid retrieval | Multi-process concurrency for the multi-user mandate; row-level security for Phase 5 scoping; one Compose service |
| OpenCode | Coding surface (Phase 3) | Adopted, not built; local-endpoint-first; transcripts ingested |
| OKF (Open Knowledge Format) | Wiki bundle format (Phase 4) | Google's June 2026 spec formalizing the LLM-wiki pattern: markdown + YAML frontmatter + index.md; adopt the format, keep our maintenance loops |
| MCP (Model Context Protocol) | Boundary to external agent hosts (OpenCode, Claude) | Server over MemoryStore with scopes + audit; NOT used between internal components |
| OpenAI-compatible `/v1/chat/completions` | Model-serving seam | The boundary that makes runtimes swappable; distinct from MCP |
| Anthropic API (API-key) | Claude escalation tool | Third-party API-key use is fully supported; the 2026 restriction was subscription-OAuth in harnesses, not the API |
| Docker sandbox | Tool execution isolation | No agent access to project root |
| FastAPI / plain Python / asyncio | Internal service boundaries; arbiter/queue machinery | MCP is for the outside; Python for the inside |
| pytest + GitHub Actions + seeded eval sets | CI and the eval harness | Per-commit functional tests with a shared FakeLLM fixture; nightly GPU evals; behavioral checks from the harness-quality checklist |
| Mermaid.js | Diagrams | House colorway per preferences |

## Adopted — patterns (harvested, not installed)

| Source | Pattern taken |
|---|---|
| MNEMOS | Manifest-on-transform, versioning/audit contracts, ACID-SoR doctrine |
| MemPalace | Layered injection (0-3), wings/rooms→scoped weights, tunnels (improved: lifecycle + reinforcement), raw-text-beats-summaries evidence |
| Letta/MemGPT | Memory-paging vocabulary; LLM-invoked deep search as Layer 3 |
| Hermes / SimonScrapes stack | Frozen-snapshot injection, per-turn memory-promotion hook, back-catalog import (Phase 5), recall ladder with context expansion, cite-or-admit |
| GBrain | Citation discipline; honest "not found" |
| Claude Code / deepagents | Context offloading, tool lazy-loading idea, compaction triggers, subagent isolation, loop-boundary injection |
| Cline | Three-tier code retrieval (ripgrep / fuzzy / tree-sitter) for the research pipeline's code tools |
| PageIndex / Alpha Iterations | Document-tree building (own implementation, pymupdf4llm-style parsing control, checksum-cached) |
| Sudip P. | One-write-fan-out ingestion contract; reconciliation sampling; "graph DB only when a real multi-hop query exists" |
| Agent Native | Harness-quality checklist; quant-behavior evaluation; API-boundary reference architecture |

## Watchlist — possible future, experiment before commitment

| Tool | Trigger to revisit |
|---|---|
| Gemma 4 (31B dense / 26B-A4B) | Conversational register + decorrelated second opinion; arrives with second RTX 3090 (Phase 4); listening tests decide |
| GLM-class coding model | Heads-down mode: second coding model in the conversational slot, size permitting (Phase 4+) |
| Harness-1 (UIUC/Berkeley/Chroma, Apache 2.0) | Retrieval subagent for heads-down mode / third GPU; benchmark on our corpus (Phase 4) |
| Cog-RAG (official repo) | Benchmark vs wiki on research corpus (Phase 4); OpenAI-compatible config → local Qwen |
| DeepSeek Harness (dsh) | Coding-surface alternative to OpenCode; MIT, everything-is-a-plugin, append-only session log; developer preview with breaking changes — let it mature |
| deepagents | Research-loop orchestration; evaluate vs hand-rolled loop (Phase 4); local-capable via OpenAI-compatible endpoints |
| Graphiti (OSS, Neo4j/FalkorDB) | Temporal knowledge graph when bi-temporal SQL strains: routine temporal multi-hop or failing entity resolution; costs a service, a derived data copy, and ingest-LLM calls; clean backfill from facts table |
| LightRAG | If a lightweight entity/theme graph is wanted before/instead of hypergraphs |
| LangGraph | Only if research workflows need checkpoint/resume machinery we'd otherwise hand-build |
| SGLang | Lateral alternative to vLLM if it misbehaves; strong structured output |
| Redis / Valkey | Multi-process shared cache/locks (Phase 5 era) |
| Qwen3.6-35B-A3B MoE | Throughput-biased alternative if measured latency demands; more VRAM, faster tokens |
| Mamba-class ASR checkpoints | The ~Phase 3.5 ASR swap, when production-quality streaming checkpoints exist |
| DSPy | Prompt optimization once eval sets are rich (late) |
| Search-R1 / Tongyi DeepResearch | Retrieval-policy research references |
| Mistral OCR / Marker / MinerU | Bulk-ingestion OCR only if Qwen-vision throughput becomes the bottleneck |

## Pass — evaluated, not a fit

| Tool | Reason |
|---|---|
| Rasa | Replaced; closed-set intent classification is the canonical failure mode |
| DuckDB (after Phase 2) | Served iteration 1 well; operational tables migrate to Postgres in Phase 2 and it retires |
| Canary-Qwen LLM mode as router | Scratched: 4B is the router; a transcription-tuned 1.7B backbone is the wrong tool for structured routing |
| LangChain (as core dependency) | Framework overhead in a hand-owned statechart core; patterns already absorbed |
| Ollama (production) | Convenience layer lagging model support; llama.cpp direct or vLLM instead |
| GGUF quants for the 27B | vLLM's GGUF path is experimental/out-of-tree; use Int4 AWQ/GPTQ-class checkpoints |
| Mem0 / Zep (as products) | Summary-first storage contradicts verbatim-first principle (Graphiti engine separately on watchlist) |
| MemPalace (as codebase) | Patterns harvested; SQLite triple store and consumer defaults not a fit |
| MNEMOS (as deployment) | Contracts adopted; full machinery overweight for this scale |
| Hermes / OpenClaw / SimonScrapes AgenticOS (as platforms) | Fawkes occupies this slot natively; patterns harvested |
| GBrain (as system) | Resolver pattern already present in the router+FSM; cron-and-skills machinery redundant |
| nano-graphrag / GraphRAG (full) | Heavy LLM indexing + staleness; theme layer served by wiki; community detection not needed at this scale |
| Hypergraph stores (as build target) | One synthesis layer at a time; Cog-RAG stays an experiment |
| ChromaDB | pgvector inside the ACID boundary covers it |
| Chroma Context-1 | Harness unreleased; superseded by Harness-1 |
| Triton Inference Server | Only for heterogeneous multi-model fleet serving — unlikely |
| Adi Insights MoE-Mamba claims | Verified garbled-to-fabricated; hybrids (already inside Qwen3.x) are the real story |
| Regex thought-stripping (Elmali-style) | Structured output separates response_text by design; sanitize at the contract, not with regex |
| Hand-built coding harness | Adopt OpenCode/dsh-class instead; build voice+memory, not repo-map machinery |
| Cloud-first memory services | Local-by-default principle |

## Appropriations and personal decisions (how we use the tools)

**Context layers (adopted from MemPalace, rebuilt).** Layer 0 — persona and tool registry: static, prompt-cached, identical on every turn. Layer 1 — per-user standing context: loaded when ECAPA resolves identity (mid-session if identification arrives late); holds the promoted durable facts, active projects/goals, and — during research sessions — the research status board. Layer 2 — per-turn topic pre-retrieval: hybrid BM25+vector over the finalized utterance, scope-weighted, tens of milliseconds. Layer 3 — deep search: LLM-invoked tools (recall, grep over transcripts, document trees) on demand only.

**Organization (replacing MemPalace's wings/rooms/halls).** Wings → **projects** (scope keys on every row; project management UI in Phase 5). Rooms → **topics** within a project, emergent, not pre-created. Halls → replaced by the **facet taxonomy**: fact / preference / decision / event / task-state (a starting set, extended via ontology versions; MemPalace's Travel/Work/Health/Relationships/General were consumer defaults we discarded). Scopes act as **soft weights** on retrieval scores, never hard filters, except where row-level security demands isolation between users.

**Tunnels (MemPalace's idea, our lifecycle).** Propose (autonomously when cross-project relevance is high or local context sparse, always attributed and flagged on first use) → cross on explicit command → reinforce on repeated use → decay on veto ("don't factor that in") → destroy on request ("never connect these"). Layer-1 standing-context matches form weak provisional tunnels.

**Recall ladder (Hermes/SimonScrapes/GBrain synthesis).** Standing context → hybrid BM25+vector with Reciprocal Rank Fusion → rerank → context expansion to neighboring turns → cited synthesis, or an honest "not found." Verbatim text is what gets cited; summaries are never the source of truth.

**Ingestion contract (Sudip P.'s event bus, without the bus).** One `ingest()` entry point; content hash + timestamps + provenance; idempotent per-store handlers (transcripts, facts, FTS, vectors, trees, wiki). Swap the internals for a real bus (Redis Streams/NATS) when multi-process, without changing callers.

**Receipts (MNEMOS contracts).** Manifest rows on every summarization/compression/archival; trace records on every retrieval; versioned ontology migrations; nothing silently rewritten.

**Routing doctrine (the anti-Rasa).** A small instruction-following LLM classifies every utterance into {FSM-advance event | freeform | ignore/backchannel | escalate} with a confidence; the deterministic validator ratifies every proposed transition; misroutes are always recoverable. Coughs and fragments that leak past VAD land in the ignore class.

**Dual-pipeline interplay (our design).** Voice reads down (status board in Layer 1 + deep transcript access in Layer 3); research receives only notes-up and imperatives. Mid-flight input goes to the arbiter (the 4B router): discard / queue for the next loop boundary / interrupt via out-of-band cancel. Machinery is an asyncio inbox queue, a pending counter, a condition the research loop waits on at each boundary (with timeout), and a watcher task that cancels generation on interrupt. The research model — never the arbiter — decides how to resteer.

**Thinking policy.** Same weights, per-request settings: voice runs thinking off or with a small budget; research runs thinking high with thinking preservation. Gemma, when present, adds register and decorrelated judgment, not a different thinking policy.

**Ontology v1 sketch (Phase 1 deliverable as `ontology.md`).** Entities: Person (user), Device, Project, Conversation, Turn, Document, DocumentNode, Fact, Task, Tool. Relations (domain → range): spoke (Person→Turn), belongs_to (Turn/Document/Fact→Project), cites (Fact→Turn/DocumentNode), supersedes (Fact→Fact), prefers / decided / scheduled (Person→Fact by facet), assigned_to (Task→Person). Every Fact carries facet, valid_from, valid_to, recorded_at, source. Rule: a type or relation earns its place only with a concrete query behind it; start strict.

## VRAM ledger (keep current)

| Card | Resident | Approx. |
|---|---|---|
| RTX 3090 #1 (24 GB) | Phase 1: Qwen3.8-27B Int4 ~16-17 GB + KV cache (~2-3 GB on the hybrid architecture). Phase 2 interim: speech stack (Canary-Qwen ~5 GB, XTTS ~1.8 GB, ECAPA/MarbleNet/FastConformer ~1-2 GB) with the 27B swapped in when resident | ~20 GB |
| GTX 1660 Super (6 GB) | 4B router Q4 ~2.7 GB + bge-m3 ~1.2 GB (+ reranker ~0.6 GB) via llama.cpp | ~4.5-5 GB |
| RTX 3090 #2 (planned) | Speech stack + Gemma 4 or heads-down model; frees #1 for the 27B full-time | ~20-22 GB |
