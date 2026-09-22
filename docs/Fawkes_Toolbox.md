# Fawkes Toolbox
Every tool, model, and framework vetted across this research round, sorted into: **Adopted** (committed), **Watchlist** (possible future / experiment first), **Pass** (not a fit). A final section records how we appropriate tools — the personal decisions and patterns that are ours. Last updated 2026-09-20.

## Adopted — critical path

| Tool | Role | Note |
|---|---|---|
| Qwen3.8-27B (Apache 2.0) | Research-slot model; voice-slot model through Phase 3; vision/OCR | Dense 27B, hybrid architecture (48 of 64 layers Gated DeltaNet), 262K ctx; Int4 fits a 3090; set thinking budgets per pipeline |
| Qwen3-4B-Instruct class | Router (voice), arbiter (text and injections), extraction, memory-promotion hook, background jobs | Confidence-gated escalation; upgrade only if it fails the routing eval slice |
| vLLM | Serving runtime for the 27B | OpenAI-compatible endpoint; official Qwen recipes; prefix caching over linear-attention layers experimental — measure |
| llama.cpp (server) | Utility-model host for the 4B beside vLLM; fallback runtime; candidate long-context engine via tuned fork | GGUF Q4 |
| bge-m3-class embedding model; bge-reranker-v2-m3-class | Embeddings for pgvector; optional rerank | CPU until the second GPU |
| Postgres + pgvector + native FTS | System of record + hybrid retrieval + document trees + capability cards | Silo keys from migration 001 (Tier A); partitioning and silo-keyed RLS deferred to a second silo (Tier B); `timestamptz` UTC |
| OpenCode | Coding surface (Phase 3) | Adopted, not built |
| code-review-graph | Code knowledge graph: blast radius, callers/callees, PR-grade impact | Ready-to-use MCP server; SQLite-backed tree-sitter AST graph; sub-second incremental updates; code-only; used as shipped with its own storage |
| graphify | Multimodal knowledge graph: code plus docs, PDFs, images; community detection; bird's-eye view | Ready-to-use MCP server (embedded, plus community wrappers); heavier; complements code-review-graph rather than replacing it; used as shipped |
| OKF (Open Knowledge Format) | Wiki bundle format (Phase 4) | Adopt the format, keep our maintenance loops |
| MCP (Model Context Protocol) | Boundary to external agent hosts | Server over MemoryStore with silo/user/project scopes + audit; not used between internal components |
| OpenAI-compatible `/v1/chat/completions` | Model-serving seam | Distinct from MCP |
| Anthropic API (API-key) | Claude escalation via the external API bridge | OCR-first text payloads; remote tiers carded per release, spot-checked monthly |
| codeburn | Token/cost accounting for adopted coding-agent sessions | Complements Fawkes trace records |
| Docker sandbox; FastAPI / plain Python / asyncio | Tool isolation; internal boundaries and arbiter machinery | |
| pytest + GitHub Actions + seeded eval sets + secret scanning | CI, the eval harness, publish hygiene | Deterministic slice scoring; FakeLLM fixture; silo leakage tests; gitleaks-class scanning; capability cards on fingerprint change |
| Mermaid.js | Diagrams at three complexity levels | Sources in `docs/diagrams/`; README embeds an exported image |

## Adopted — patterns (harvested, not installed)

| Source | Pattern taken |
|---|---|
| MNEMOS | Manifest-on-transform, versioning/audit contracts, ACID-SoR doctrine |
| MemPalace | Layered injection (0-3), wings/rooms→scoped weights, tunnels with lifecycle, raw-text-beats-summaries evidence |
| Letta/MemGPT | Memory-paging vocabulary; LLM-invoked deep search as Layer 3 |
| Hermes / SimonScrapes stack | Frozen-snapshot injection, per-turn memory-promotion hook, back-catalog import, recall ladder with context expansion, cite-or-admit |
| GBrain | Citation discipline; honest "not found" |
| Claude Code / deepagents | Context offloading, tool lazy-loading, compaction triggers, subagent isolation, loop-boundary injection |
| Cline | Three-tier code retrieval (ripgrep / fuzzy / tree-sitter) |
| PageIndex / Alpha Iterations | Document-tree building (own implementation, checksum-cached, stored in Postgres) |
| Sudip P. | One-write-fan-out ingestion; reconciliation sampling; graph DB only on a real multi-hop query; segmented evals before adding a store |
| Agent Native | Harness-quality checklist; quant-behavior evaluation; API-boundary reference architecture |
| Plano / Arch-Router | Complexity-and-intent routing as a small trained model; OpenTelemetry-style span attributes |
| OrcaRouter Fusion / OpenRouter Fusion | Arbiter strategies for panel mode: judge-selects, vote, synthesize, test-harness-wins, race |
| Headroom | Cache-aligned live-zone compression; reversible compression with on-demand originals |
| syv-ai qwen38-27b-rtx3090 / llamAmpere (Zhu write-ups) | Draft-model speculative decoding with a model-specific draft vocabulary; requantized embeddings; pinned KV pool with boot-time fit check; "a tok/s figure is meaningless without the prompt"; ~4 medium-depth workers as the fan-out design point |
| SWE-bench pattern | Coding-patch pass rate measured by applying the model's patch to a failing-test repository and running the suite |
| Silo design note (2026-09-18) | Silo → project → records; single-silo-column tunnels; per-silo content hash; immutable silo context; identity outside silos; export-and-re-ingest crossings; tiered into A (now) and B (later) |
| Screen-capture conversation (2026-09) | Accessibility-API-first screen reading; output-rate control over capture-rate; KVM/HID as its own track |

## Watchlist — possible future, experiment before commitment

| Tool | Trigger to revisit |
|---|---|
| Tuned vLLM fork (syv-ai/qwen38-27b-rtx3090) | Phase 1 serving experiment; fast at short context with parallel slots; degrades past ~40K; pinned fork — keep behind the API boundary |
| Tuned llama.cpp fork (JakeATX/llamAmpere + ATX quant) | Phase 1 serving experiment for deep-context research; steady at 150K+; value compression is not Q8 fidelity |
| DFlash2 / MTP speculative decoding | Drafters are per-target-model; no public DFlash2 for Gemma 4 — n-gram/lookup drafting or a small Gemma draft model instead |
| SGLang | Lateral runtime alternative (RadixAttention, strong structured output); heavier install |
| Tensor parallelism across two 3090s | Only with verified NVLink, x16 lanes, PSU headroom; default is one model per card; measure |
| Gemma 4; GLM-class coding model; Harness-1 | Phase 4 slots and experiments |
| Plano (gateway) | If the bridge outgrows a Python policy module |
| Headroom (proxy) | Deferred indefinitely |
| Cog-RAG; DeepSeek Harness; deepagents; Graphiti; LightRAG; LangGraph; Redis/Valkey; DSPy; Search-R1; Mistral OCR/Marker/MinerU; rtk/Ponytail | Unchanged triggers from prior rounds |

## Pass — evaluated, not a fit

| Tool | Reason |
|---|---|
| Rasa | Closed-set intent classification is the canonical failure mode |
| DuckDB (after Phase 2) | Operational tables migrate to Postgres; the ECAPA matrix is a RAM structure built at startup |
| GTX 1660 Super | No longer in the machine |
| Canary-Qwen LLM mode as router | 4B is the router |
| Caveman / pxpipe / compression proxies | Measured savings marginal or fidelity-destroying; deferred indefinitely |
| LangChain (core dependency); Ollama (production); GGUF under vLLM | Framework overhead; lagging support; experimental path |
| Mem0 / Zep (products); MemPalace / MNEMOS / Hermes / OpenClaw / GBrain (codebases or platforms) | Summary-first or patterns-only |
| nano-graphrag / GraphRAG; hypergraph stores; ChromaDB; Chroma Context-1; OrcaRouter / OpenRouter (hosted); Triton | Covered elsewhere or contradict local-by-default |
| Adi Insights MoE-Mamba claims; regex thought-stripping; hand-built coding harness; cloud-first memory | Discredited or superseded |

## Appropriations and personal decisions (how we use the tools)

**Context layers (adopted from MemPalace, rebuilt).** Layer 0 persona and tool registry, prompt-cached. Layer 1 per-user standing context, loaded on identity, holding promoted facts, active projects, and the research status board refreshed at every loop boundary. Layer 2 per-turn hybrid pre-retrieval, scope-weighted within the silo. Layer 3 LLM-invoked deep search.

**Partitioning, three mechanisms kept distinct.** *Silos* are hard partitions of data. Tier A (migration 001, about an hour, no runtime cost): the silo key on every row, in every unique constraint and index prefix, composite project references, single-silo tunnels, per-silo content hash, silo context on every call, leakage tests. Tier B (first real second silo): partitioning by silo, silo-keyed forced row-level security, per-silo roles, instance pin. *Project scopes and tunnels* are soft ranking weights inside a silo. *Row-level security* is per-user authorization inside a silo (user-keyed in Phase 5 regardless). Identity lives outside silos. Fawkes runs one silo; the key is the glue that lets one codebase serve many-silos-per-database and one-silo-per-database deployments alike.

**Tunnels (within a silo).** Propose → cross on explicit command → reinforce on use → decay on veto → destroy on request.

**Recall ladder.** Standing context → hybrid BM25+vector with Reciprocal Rank Fusion → rerank → context expansion → cited synthesis, or an honest "not found."

**Ingestion contract (Sudip P.'s event bus, without the bus).** One `ingest()` entry point; hash per silo; timestamps, provenance, silo and project stamps; idempotent handlers; turns, traces, and cost rows enter the same way.

**Receipts.** Manifests on transformation; traces on retrieval; tokens and cost on every model call; versioned ontology migrations.

**Timestamp policy.** UTC `timestamptz` everywhere; per-session IANA time zone (default is a deployment setting, not a documented value); local time and elapsed time rendered per session.

**Input routing doctrine (three channels).** Voice: ASR → 4B router → {ignore | FSM event | escalate}; the router is the primary FSM driver; the voice-slot model may also propose transitions that need conversational context or that the router missed. Text: → 4B arbiter → {start | queue | interrupt | FSM command}; never discard; illegal commands return to the arbiter. Rubber-duck powers: answer, note, or start/queue/interrupt directly to the research command queue. Response modality follows input modality. The validator returns rejections to the proposer with a hint, bounded.

**Dual-pipeline interplay.** Shared blackboard + status board — *notes-up* (voice → research), *read-down* (status board into Layer 1, deep transcript access via Layer 3), *imperatives* (start/queue/interrupt to the research command queue). Machinery: asyncio inbox, pending counter, boundary condition with timeout, watcher cancel. The research model resteers itself.

**Serving policy.** The endpoint is the seam; the engine behind it is chosen by the harness per role. Two cards default to one model per card. Tensor parallelism is an experiment. A throughput number is recorded with the prompt and context depth that produced it.

**Capability cards, how they are measured.** Deterministic scoring wherever possible: labeled routing verdicts (exact match), seeded recall facts (answer contains fact and cites expected turn), fixed-corpus multi-hop (expected citation ids), coding patches (apply to a failing-test repository and run the suite), schema validation, loop/stop/latency observed by the harness; a reference-anchored LLM judge only for free-text slices. The seed set is fixed; the run is keyed by a fingerprint of model, quantization, serving configuration, prompt version, and index contents, skipped when unchanged, repeated over several sampling seeds with a confidence interval, scheduled into idle windows plus a weekly sanity run. Real-usage corrections and thumbs-downs become new labeled cases. Remote paid tiers are carded once per release and spot-checked monthly on a capped sample. Cards are absolute per model; comparison falls out of shared tasks. Panel mode is a runtime feature, not the measurement.

**Routing to remote models.** Explicit request → criticality → verified local confidence → complexity → budget; tiers from capability cards; text/JSON/Markdown outward.

**Panel mode.** Identical-spec parallel execution with an arbiter, or decision-point consensus; every fan-out recorded; reserved for complex, critical, or sensitive tasks; available inside heads-down mode.

**Thinking policy.** Same weights, per-request settings: voice off or small budget; research high with preservation.

**Diagram conventions.** Colors mark logic roles: `action` (green) = LLM inference; `handler` (brown) = deterministic code; `decision` (orange diamond) = dispatch point; `server` (grey cylinder) = store; `form` (blue) = subsystem shown as a unit; `startEnd` (thick green stadium) = entry/exit. Dotted = load-once or retry-return. Subgraphs mark pipeline boundaries. **Complexity levels:** 1 = five boxes; 4 = README (a visitor grasps it in ten seconds; ~15-20 nodes, one store, no retry loops, only sanctioned crossings); 7 = engineering reference (~35 nodes, subgraphs, every crossing labeled); 10 = exhaustive (every component and pathway in the specification, one node per moving part). Sources: `docs/diagrams/architecture_L4.mmd`, `_L7.mmd`, `_L10.mmd`; the README embeds an exported image of L4.

**Public-documents hygiene.** No secrets, hostnames, credentials, employer names, or third-party personal names in project documents; deployment defaults live in configuration; design notes referencing an employer or compliance context stay out of the public repository; secret scanning on every push.

**Ontology v1 sketch.** Entities: Silo, Person, Device, Project, Conversation, Turn, Document, DocumentNode, Fact, Task, Tool, ModelCall, CapabilityCard. Relations: spoke, belongs_to (→Project→Silo), cites, supersedes, prefers/decided/scheduled, assigned_to, produced_by, member_of. Every Fact carries facet, valid_from, valid_to, recorded_at, source, silo_id. Start strict.

## VRAM ledger (keep current)

| Card | Resident | Approx. |
|---|---|---|
| RTX 3090 (24 GB), Phase 1 text-mode | Qwen3.8-27B Int4 ~15-17 GB + KV ~2 GB; 4B router Q4 ~2.7 GB; embedder on CPU | ~22 GB |
| RTX 3090 (24 GB), Phase 2 interim | Speech stack ~8-9 GB + 4B router ~2.7 GB; 27B swapped in only when no speech session needs the stack | ~12 GB (+17 with 27B) |
| RTX 3090 #2 (planned) | Speech stack + Gemma 4 (or heads-down model); card one holds the 27B; one model per card | ~20-22 GB |
