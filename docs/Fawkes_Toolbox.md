# Fawkes Toolbox
Every tool, model, and framework vetted across this research round, sorted into: **Adopted** (committed), **Watchlist** (possible future / experiment first), **Pass** (not a fit). A final section records how we appropriate tools — the personal decisions and patterns that are ours. Last updated 2026-09-15.

## Adopted — critical path

| Tool | Role | Note |
|---|---|---|
| Qwen3.8-27B (Apache 2.0) | Research-slot model; voice-slot model through Phase 3; vision/OCR (interactive and ingest-time) | Dense 27B, 262K ctx, vision+video; Int4 fits a 3090; default reasoning "xhigh" — set thinking budgets per pipeline; vendor benchmarks pending independent confirmation |
| Qwen3-4B-Instruct class | Router (voice channel), arbiter (text channel and mid-flight injections), extraction, memory-promotion hook, background jobs | Confidence-gated escalation; upgrade to a 14B-class utility model only if it fails the routing eval slice |
| vLLM | Serving runtime for the 27B | OpenAI-compatible endpoint, concurrency-class serving, official Qwen recipes; prefix caching over linear-attention layers experimental — measure |
| llama.cpp (server) | Utility-model host for the 4B beside vLLM on the 3090; fallback runtime | GGUF Q4; not the 27B's runtime |
| bge-m3-class embedding model | Embeddings for pgvector | CPU until the second GPU (~50-150 ms/query, acceptable for text-mode and ingestion) |
| bge-reranker-v2-m3-class | Optional rerank stage (research path; voice only if measured cheap) | Sub-1B cross-encoder |
| Postgres + pgvector + native FTS | System of record + hybrid retrieval | Multi-process concurrency for the multi-user mandate; row-level security for Phase 5 scoping; `timestamptz` UTC everywhere; one Compose service |
| OpenCode | Coding surface (Phase 3) | Adopted, not built; local-endpoint-first; transcripts ingested |
| Code-graph MCP server (code-review-graph or graphify class) | Codebase knowledge graph for OpenCode and the research pipeline (Phase 3-4) | tree-sitter AST graph of files/symbols/calls/imports; SQLite-backed; sub-second incremental updates per commit; MCP tools for callers/callees/blast radius; adopt at the edge, do not build |
| OKF (Open Knowledge Format) | Wiki bundle format (Phase 4) | Google's June 2026 spec formalizing the LLM-wiki pattern: markdown + YAML frontmatter + index.md; adopt the format, keep our maintenance loops |
| MCP (Model Context Protocol) | Boundary to external agent hosts (OpenCode, Claude) | Server over MemoryStore with scopes + audit; NOT used between internal components |
| OpenAI-compatible `/v1/chat/completions` | Model-serving seam | The boundary that makes runtimes swappable; distinct from MCP |
| Anthropic API (API-key) | Claude escalation via the external API bridge | Third-party API-key use is fully supported; OCR-first text payloads; images are token-billed at the API (no per-chat quota) but text is cheaper and more precise |
| codeburn | Token/cost accounting for adopted coding-agent sessions (OpenCode, Claude Code) | Reads session files on disk, prices calls including cache read/write, deterministic task categories; complements Fawkes's own trace records |
| Docker sandbox | Tool execution isolation | No agent access to project root |
| FastAPI / plain Python / asyncio | Internal service boundaries; arbiter/queue machinery | MCP is for the outside; Python for the inside |
| pytest + GitHub Actions + seeded eval sets | CI and the eval harness | Per-commit functional tests with a shared FakeLLM fixture; nightly GPU evals; behavioral checks from the harness-quality checklist; capability cards |
| Mermaid.js | Diagrams | House colorway per preferences (semantics in the appropriations section) |

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
| Sudip P. | One-write-fan-out ingestion contract; reconciliation sampling; "graph DB only when a real multi-hop query exists"; segmented evals before adding a store |
| Agent Native | Harness-quality checklist; quant-behavior evaluation; API-boundary reference architecture |
| Plano / Arch-Router | Complexity-and-intent routing as a small trained model in front of the model fleet; OpenTelemetry-style span attributes (model, provider, tokens, duration, time-to-first-token) for our trace schema |
| OrcaRouter Fusion / OpenRouter Fusion | Arbiter strategies for panel mode: judge-selects (best-of-n, served verbatim), vote, synthesize (Mixture-of-Agents), test-harness-wins, race; declarative routing rules with a required default |
| Headroom | Cache-aligned "live-zone" compression (never rewrite the frozen prefix); reversible compression with on-demand retrieval of originals |
| Screen-capture conversation (2026-09) | Text-extraction over screenshots (DOM/DevTools, terminal capture); accessibility-API-first screen reading with event-triggered OCR; output-rate control over capture-rate |

## Watchlist — possible future, experiment before commitment

| Tool | Trigger to revisit |
|---|---|
| Gemma 4 (31B dense / 26B-A4B) | Voice-slot model: conversational register + decorrelated second opinion; arrives with second RTX 3090 (Phase 4); listening tests decide |
| GLM-class coding model | Heads-down mode: second coding model in the conversational slot, size permitting (Phase 4+) |
| Harness-1 (UIUC/Berkeley/Chroma, Apache 2.0) | Retrieval subagent for heads-down mode / third GPU; benchmark on our corpus (Phase 4) |
| Plano (gateway) | If the external API bridge outgrows a Python policy module: Envoy-based OpenAI-compatible proxy with Arch-Router complexity routing, guardrails, and tracing; stateless container |
| Headroom (proxy) | Context-compression proxy for external traffic only if spend proves material after structured-extraction payloads; independent measurements show single-digit-percent real savings for coding workloads |
| graphify (full multimodal graph) | If the code graph should also span docs/PDFs/images with community detection; heavier than code-review-graph |
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
| rtk / Ponytail | Shell-output compaction and leaner code output inside adopted coding harnesses; cheap to try, marginal measured gains |

## Pass — evaluated, not a fit

| Tool | Reason |
|---|---|
| Rasa | Replaced; closed-set intent classification is the canonical failure mode |
| DuckDB (after Phase 2) | Served iteration 1; operational tables migrate to Postgres in Phase 2 and it retires completely — the ECAPA matrix is a RAM structure built at startup from any store |
| GTX 1660 Super | No longer in the machine; removed from all layouts |
| Canary-Qwen LLM mode as router | Scratched: 4B is the router; a transcription-tuned 1.7B backbone is the wrong tool for structured routing |
| Caveman (as a token saver) | Shortens model replies only; raised total tokens 7% in a coding benchmark; our voice register already enforces brevity by prompt |
| pxpipe | Text-to-image context packing loses character-level precision; incompatible with epistemic hygiene |
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
| OrcaRouter / OpenRouter (as hosted gateways) | Cloud gateways contradict local-by-default; their arbiter patterns are adopted, their services are not |
| Triton Inference Server | Only for heterogeneous multi-model fleet serving — unlikely |
| Adi Insights MoE-Mamba claims | Verified garbled-to-fabricated; hybrids (already inside Qwen3.x) are the real story |
| Regex thought-stripping (Elmali-style) | Structured output separates response_text by design; sanitize at the contract, not with regex |
| Hand-built coding harness | Adopt OpenCode/dsh-class instead; build voice+memory, not repo-map machinery |
| Cloud-first memory services | Local-by-default principle |

## Appropriations and personal decisions (how we use the tools)

**Context layers (adopted from MemPalace, rebuilt).** Layer 0 — persona and tool registry: static, prompt-cached, identical on every turn. Layer 1 — per-user standing context: loaded when ECAPA resolves identity (mid-session if identification arrives late); holds promoted durable facts, active projects/goals, and — during research sessions — the research status board. Layer 2 — per-turn topic pre-retrieval: hybrid BM25+vector over the finalized utterance, scope-weighted, tens of milliseconds. Layer 3 — deep search: LLM-invoked tools (recall, grep over transcripts, document trees, research-transcript access) on demand only.

**Organization (replacing MemPalace's wings/rooms/halls).** Wings → **projects** (scope keys on every row; project management UI in Phase 5). Rooms → **topics** within a project, emergent, not pre-created. Halls → replaced by the **facet taxonomy**: fact / preference / decision / event / task-state (a starting set, extended via ontology versions; MemPalace's Travel/Work/Health/Relationships/General were consumer defaults we discarded). Scopes act as **soft weights** on retrieval scores, never hard filters, except where row-level security demands isolation between users.

**Tunnels (MemPalace's idea, our lifecycle).** Propose (autonomously when cross-project relevance is high or local context sparse, always attributed and flagged on first use) → cross on explicit command → reinforce on repeated use → decay on veto ("don't factor that in") → destroy on request ("never connect these"). Layer-1 standing-context matches form weak provisional tunnels.

**Recall ladder (Hermes/SimonScrapes/GBrain synthesis).** Standing context → hybrid BM25+vector with Reciprocal Rank Fusion → rerank → context expansion to neighboring turns → cited synthesis, or an honest "not found." Verbatim text is what gets cited; summaries are never the source of truth.

**Ingestion contract (Sudip P.'s event bus, without the bus).** One `ingest()` entry point; content hash + timestamps + provenance; idempotent per-store handlers (transcripts, facts, FTS, vectors, trees, wiki). Swap the internals for a real bus (Redis Streams/NATS) when multi-process, without changing callers. What we took from Sudip: the relational store is the only system of record; one write fans out to every derived store; reconciliation sampling checks that stores agree; a graph database is added only when a real multi-hop query demands it; a segmented eval harness precedes any new store.

**Receipts (MNEMOS contracts, extended).** Manifest rows on every summarization/compression/archival; trace records on every retrieval; tokens and cost on every model call (Plano-style span attributes); versioned ontology migrations; nothing silently rewritten.

**Timestamp policy.** Every timestamp is a full date-time stored as UTC `timestamptz`. Every device and web session registers an IANA time zone on connect (default `America/Detroit`); the prompt builder renders local time and elapsed-since-previous-turn per session. No naive timestamps, no local-time storage.

**Input routing doctrine (three channels; the anti-Rasa).** (1) Voice: all ASR text → 4B router → {ignore | FSM event (+slots, tiny in-workflow replies rendered by the router) | escalate to the voice-slot model}; almost everything not dropped or FSM-bound is escalated. ASR text never reaches the research pipeline except through the voice pipeline. (2) Text (web/app): all text → 4B as arbiter → {dispatch to research now | queue for next loop boundary | interrupt running task | FSM command}; never discard; ~5 s default-to-queue timeout. (3) Rubber-duck powers: the voice-slot model may answer, start a background research task, write a blackboard note, queue an injection, or interrupt research. The deterministic validator ratifies every proposed transition; coughs and fragments that leak past VAD land in the ignore class.

**Dual-pipeline interplay (our design).** *Notes-up*: distilled context flowing from the voice pipeline up into the research pipeline's blackboard (voice → research, filtered). *Read-down*: research state flowing down to the voice pipeline — a status board refreshed at every loop boundary into voice Layer 1, plus deep access to the full research transcript via voice Layer 3 tools (research → voice; the voice pipeline can always speak intelligibly about what the researcher is doing without carrying a copy of its context). *Imperative channel*: explicit commands from voice or text that execute actions in the research pipeline (start, queue, interrupt) rather than merely adding context. Machinery: asyncio inbox queue, pending counter, a condition the research loop waits on at each boundary (with timeout), and a watcher task that cancels generation on interrupt. The research model — never the arbiter — decides how to resteer.

**Routing to remote models (external API bridge).** Order of decision: explicit user request → task criticality flag → local confidence (the structured-output confidence field, treated as a weak signal) cross-checked by cheap verification (sample agreement, verifier/judge pass, test-harness result where applicable) → complexity estimate (length, tool needs, multi-step markers, domain) → budget. Capability cards for every local and remote model are written from segmented eval results, never from vendor tables, and revised nightly. Payloads outward are extracted text, JSON, or Markdown; documents and images stay local unless visual judgment is the task.

**Panel mode (decorrelated judgment; from the Greek-tutor porting pipeline and Fusion patterns).** Two grades: identical-spec parallel execution with an arbiter (judge-selects verbatim, vote, synthesize, test-harness-wins), and decision-point consensus where only extracted key decisions are cross-checked. Every fan-out records which models ran, each candidate, and the verdict. Reserved for complex, critical, or sensitive tasks; Qwen self-fusion (multiple samples) available before Gemma exists.

**Thinking policy.** Same weights, per-request settings: voice runs thinking off or with a small budget; research runs thinking high with thinking preservation. Gemma, when present, adds register and decorrelated judgment, not a different thinking policy.

**Diagram conventions (house colorway, semantics assigned).** Colors mark logic roles, not component types: `action` (green) = LLM inference nodes; `handler` (brown) = deterministic processing code; `decision` (orange diamond) = deterministic dispatch/branch points; `server` (grey) = data stores and services; `form` (blue) = pipelines and subsystems shown as a unit; `startEnd` (thick green stadium) = entry/exit. Shapes: cylinder = store, diamond = branch, stadium = entry/exit, rectangle = process; dotted edges = load-once/startup relationships.

**Ontology v1 sketch (Phase 1 deliverable as `ontology.md`).** Entities: Person (user), Device, Project, Conversation, Turn, Document, DocumentNode, Fact, Task, Tool, ModelCall. Relations (domain → range): spoke (Person→Turn), belongs_to (Turn/Document/Fact/Task→Project), cites (Fact→Turn/DocumentNode), supersedes (Fact→Fact), prefers / decided / scheduled (Person→Fact by facet), assigned_to (Task→Person), produced_by (Turn→ModelCall). Every Fact carries facet, valid_from, valid_to, recorded_at, source. Rule: a type or relation earns its place only with a concrete query behind it; start strict.

## VRAM ledger (keep current)

| Card | Resident | Approx. |
|---|---|---|
| RTX 3090 (24 GB), Phase 1 text-mode | Qwen3.8-27B Int4 ~16-17 GB + KV ~2 GB (vLLM at ~0.80 utilization); 4B router Q4 ~2.7 GB via llama.cpp; embedder on CPU | ~22 GB |
| RTX 3090 (24 GB), Phase 2 interim | Speech stack (Canary-Qwen ~5 GB, XTTS ~1.8 GB, ECAPA/MarbleNet/FastConformer ~1-2 GB) + 4B router ~2.7 GB; 27B swapped in only when no speech session needs the stack | ~12 GB (+17 when 27B loaded) |
| RTX 3090 #2 (planned) | Speech stack + Gemma 4 (or heads-down model); frees card one for the 27B full-time; embedder and reranker move to GPU | ~20-22 GB |
