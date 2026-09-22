# Fawkes Toolbox
Every tool, model, and framework vetted across this research round, sorted into: **Adopted** (committed), **Watchlist** (possible future / experiment first), **Pass** (not a fit). A final section records how we appropriate tools — the personal decisions and patterns that are ours. Last updated 2026-09-19.

## Adopted — critical path

| Tool | Role | Note |
|---|---|---|
| Qwen3.8-27B (Apache 2.0) | Research-slot model; voice-slot model through Phase 3; vision/OCR | Dense 27B, 262K ctx, vision+video; hybrid architecture (48 of 64 layers Gated DeltaNet, 16 attention) so KV cache and long context are cheap; Int4 fits a 3090; set thinking budgets per pipeline; vendor benchmarks pending independent confirmation |
| Qwen3-4B-Instruct class | Router (voice channel), arbiter (text channel and mid-flight injections), extraction, memory-promotion hook, background jobs | Confidence-gated escalation; upgrade to a 14B-class utility model only if it fails the routing eval slice |
| vLLM | Serving runtime for the 27B | OpenAI-compatible endpoint, concurrency-class serving, official Qwen recipes; prefix caching over linear-attention layers experimental — measure |
| llama.cpp (server) | Utility-model host for the 4B beside vLLM on the 3090; fallback runtime; candidate long-context engine via tuned fork | GGUF Q4; not the 27B's production runtime unless the serving experiment says otherwise |
| bge-m3-class embedding model | Embeddings for pgvector | CPU until the second GPU |
| bge-reranker-v2-m3-class | Optional rerank stage (research path; voice only if measured cheap) | Sub-1B cross-encoder |
| Postgres + pgvector + native FTS | System of record + hybrid retrieval + document trees + capability cards | Multi-process concurrency; row-level security forced from Phase 0; list partitioning by silo; `timestamptz` UTC; one Compose service |
| OpenCode | Coding surface (Phase 3) | Adopted, not built; local-endpoint-first; transcripts ingested |
| Code-graph MCP server (code-review-graph or graphify class) | Codebase knowledge graph for OpenCode and the research pipeline (Phase 3-4) | Shipped as ready-to-use MCP servers; tree-sitter AST graph of files/symbols/calls/imports in their own SQLite; sub-second incremental updates per commit; tools for callers/callees/blast radius; used as shipped — their storage is a derived store rebuildable from the repository, not migrated into Postgres |
| OKF (Open Knowledge Format) | Wiki bundle format (Phase 4) | Markdown + YAML frontmatter + index.md; adopt the format, keep our maintenance loops |
| MCP (Model Context Protocol) | Boundary to external agent hosts (OpenCode, Claude) | Server over MemoryStore with silo/user/project scopes + audit; NOT used between internal components |
| OpenAI-compatible `/v1/chat/completions` | Model-serving seam | The boundary that makes runtimes swappable; distinct from MCP |
| Anthropic API (API-key) | Claude escalation via the external API bridge | OCR-first text payloads; images are token-billed at the API (no per-chat quota) but text is cheaper and more precise |
| codeburn | Token/cost accounting for adopted coding-agent sessions | Reads session files on disk, prices calls including cache read/write; complements Fawkes's own trace records |
| Docker sandbox | Tool execution isolation | No agent access to project root |
| FastAPI / plain Python / asyncio | Internal service boundaries; arbiter/queue machinery | MCP is for the outside; Python for the inside |
| pytest + GitHub Actions + seeded eval sets + secret scanning | CI, the eval harness, and publish hygiene | Per-commit functional tests with a shared FakeLLM fixture; silo leakage tests; gitleaks-class scanning on push; nightly GPU evals writing capability cards |
| Mermaid.js | Diagrams | House colorway; source kept in `docs/diagrams/*.mmd`; README embeds an exported SVG/PNG because GitHub's renderer differs from mermaid.live |

## Adopted — patterns (harvested, not installed)

| Source | Pattern taken |
|---|---|
| MNEMOS | Manifest-on-transform, versioning/audit contracts, ACID-SoR doctrine |
| MemPalace | Layered injection (0-3), wings/rooms→scoped weights, tunnels (improved: lifecycle + reinforcement), raw-text-beats-summaries evidence |
| Letta/MemGPT | Memory-paging vocabulary; LLM-invoked deep search as Layer 3 |
| Hermes / SimonScrapes stack | Frozen-snapshot injection, per-turn memory-promotion hook, back-catalog import (Phase 5), recall ladder with context expansion, cite-or-admit |
| GBrain | Citation discipline; honest "not found" |
| Claude Code / deepagents | Context offloading, tool lazy-loading idea, compaction triggers, subagent isolation, loop-boundary injection |
| Cline | Three-tier code retrieval (ripgrep / fuzzy / tree-sitter) |
| PageIndex / Alpha Iterations | Document-tree building (own implementation, checksum-cached, stored in Postgres) |
| Sudip P. | One-write-fan-out ingestion contract; reconciliation sampling; graph DB only on a real multi-hop query; segmented evals before adding a store |
| Agent Native | Harness-quality checklist; quant-behavior evaluation; API-boundary reference architecture |
| Plano / Arch-Router | Complexity-and-intent routing as a small trained model in front of the model fleet; OpenTelemetry-style span attributes for our trace schema |
| OrcaRouter Fusion / OpenRouter Fusion | Arbiter strategies for panel mode: judge-selects, vote, synthesize, test-harness-wins, race; declarative routing rules with a required default |
| Headroom | Cache-aligned "live-zone" compression (never rewrite the frozen prefix); reversible compression with on-demand retrieval of originals |
| syv-ai qwen38-27b-rtx3090 / llamAmpere (Zhu write-ups) | Draft-model speculative decoding with a model-specific draft vocabulary; requantized embedding matrices; pinned KV pool with boot-time fit check; fp16 recurrent state; "a tok/s figure is meaningless without the prompt that produced it"; design agent fan-out for ~4 medium-depth workers |
| Silo design note (2026-09-18) | Silo → project → records hierarchy; single-silo-column tunnels; per-silo content hash; immutable silo context; forced RLS under a non-owner role; identity outside silos; export-and-re-ingest for crossings |
| Screen-capture conversation (2026-09) | Accessibility-API-first screen reading; output-rate control over capture-rate; KVM/HID as its own track |

## Watchlist — possible future, experiment before commitment

| Tool | Trigger to revisit |
|---|---|
| Tuned vLLM fork (syv-ai/qwen38-27b-rtx3090: pinned vLLM + patch stack + DFlash2 drafter) | Phase 1 serving experiment; ~100-177 tok/s at short context, ~4-8 parallel slots on one 3090, degrades past ~40K context; pinned versions and patch stacks are fragile — keep behind the API boundary |
| Tuned llama.cpp fork (JakeATX/llamAmpere + ATX-4-XS quant) | Phase 1 serving experiment for long-context research work; steady 65-80 tok/s at 150K+ context on one 3090; Turbo3 value compression is not Q8 fidelity |
| DFlash2 / MTP speculative decoding | Any drafter must be trained per target model (reads target hidden states); Qwen has both; Gemma 4 has no public DFlash2 drafter — use n-gram/lookup drafting or a small Gemma draft model instead of training one |
| SGLang | Lateral runtime alternative to vLLM (same class: OpenAI-compatible, RadixAttention prefix caching, strong structured output); heavier install; the 220 tok/s tensor-parallel recipe assumed x16 slots and peer-to-peer access |
| Tensor parallelism across two 3090s | Only with a verified NVLink bridge, both cards on x16 CPU lanes, and PSU transient headroom; default remains one model per card; measure single-stream gain vs voice-latency isolation |
| Gemma 4 (31B dense / 26B-A4B) | Voice-slot model with second RTX 3090 (Phase 4); listening tests decide |
| GLM-class coding model | Heads-down mode second coding model (Phase 4+) |
| Harness-1 (Apache 2.0) | Retrieval subagent for heads-down mode; benchmark on our corpus (Phase 4) |
| graphify (full multimodal graph) | If the code graph should also span docs/PDFs/images with community detection for a systems-level bird's-eye view; heavier than code-review-graph |
| Plano (gateway) | If the external API bridge outgrows a Python policy module |
| Headroom (proxy) | Deferred indefinitely; revisit only if external spend becomes material |
| Cog-RAG (official repo) | Benchmark vs wiki on research corpus (Phase 4) |
| DeepSeek Harness (dsh) | Coding-surface alternative to OpenCode; explicit parallel-slot subagent model; developer preview — let it mature |
| deepagents | Research-loop orchestration; evaluate vs hand-rolled loop (Phase 4) |
| Graphiti (OSS, Neo4j/FalkorDB) | Temporal knowledge graph when bi-temporal SQL strains; clean backfill from facts table |
| LightRAG / LangGraph / Redis-Valkey / DSPy / Search-R1 / Mistral OCR-Marker-MinerU / rtk-Ponytail | Unchanged triggers from prior rounds |

## Pass — evaluated, not a fit

| Tool | Reason |
|---|---|
| Rasa | Replaced; closed-set intent classification is the canonical failure mode |
| DuckDB (after Phase 2) | Operational tables migrate to Postgres; the ECAPA matrix is a RAM structure built at startup from any store; nothing to retain |
| GTX 1660 Super | No longer in the machine |
| Canary-Qwen LLM mode as router | 4B is the router |
| Caveman / pxpipe / compression proxies (as token savers) | Measured savings marginal or fidelity-destroying; structured-extraction payloads are the real lever; deferred indefinitely |
| LangChain (as core dependency) | Framework overhead in a hand-owned statechart core |
| Ollama (production) | Convenience layer lagging model support |
| GGUF quants for the 27B under vLLM | vLLM's GGUF path is experimental/out-of-tree; GGUF belongs to llama.cpp |
| Mem0 / Zep (as products) | Summary-first storage contradicts verbatim-first |
| MemPalace / MNEMOS / Hermes / OpenClaw / GBrain (as codebases or platforms) | Patterns harvested; codebases not a fit |
| nano-graphrag / GraphRAG (full); hypergraph stores (as build target) | Theme layer served by wiki; one synthesis layer at a time |
| ChromaDB | pgvector inside the ACID boundary covers it |
| Chroma Context-1 | Harness unreleased; superseded by Harness-1 |
| OrcaRouter / OpenRouter (as hosted gateways) | Cloud gateways contradict local-by-default; patterns adopted, services not |
| Triton Inference Server | Only for heterogeneous multi-model fleet serving |
| Adi Insights MoE-Mamba claims | Verified garbled-to-fabricated |
| Regex thought-stripping | Structured output separates response_text by design |
| Hand-built coding harness | Adopt OpenCode/dsh-class instead |
| Cloud-first memory services | Local-by-default principle |

## Appropriations and personal decisions (how we use the tools)

**Context layers (adopted from MemPalace, rebuilt).** Layer 0 — persona and tool registry: static, prompt-cached. Layer 1 — per-user standing context: loaded when ECAPA resolves identity; holds promoted durable facts, active projects/goals, and — during research sessions — the research status board, refreshed at every loop boundary. Layer 2 — per-turn topic pre-retrieval: hybrid BM25+vector, scope-weighted within the silo, tens of milliseconds. Layer 3 — deep search: LLM-invoked tools on demand only.

**Partitioning, three mechanisms kept distinct.** *Silos* are hard partitions of data: every row carries one silo, tunnels cannot cross, caches and prompts are keyed and asserted, and the schema works with many silos per database or one silo per database. *Project scopes and tunnels* are soft ranking weights inside a silo (wings → projects, rooms → topics, halls → the facet taxonomy). *Row-level security* is per-user authorization inside a silo. Identity (speakers, imprints, devices) lives outside silos with a membership table and a default silo per user. Fawkes runs one silo; the layer exists so that a hardened deployment needs infrastructure, not code changes.

**Tunnels (MemPalace's idea, our lifecycle, within a silo).** Propose → cross on explicit command → reinforce on repeated use → decay on veto → destroy on request. Layer-1 standing-context matches form weak provisional tunnels.

**Recall ladder.** Standing context → hybrid BM25+vector with Reciprocal Rank Fusion → rerank → context expansion → cited synthesis, or an honest "not found."

**Ingestion contract (Sudip P.'s event bus, without the bus).** One `ingest()` entry point; content hash unique per silo; timestamps, provenance, silo and project stamps; idempotent per-store handlers; turns, traces, and model-call cost rows enter the same way. What we took from Sudip: the relational store is the only system of record; one write fans out to every derived store; reconciliation sampling checks that stores agree; a graph database is added only when a real multi-hop query demands it; a segmented eval harness precedes any new store.

**Receipts (MNEMOS contracts, extended).** Manifest rows on every transformation; trace records on every retrieval; tokens and cost on every model call; versioned ontology migrations.

**Timestamp policy.** Every timestamp is a full date-time stored as UTC `timestamptz`. Every device and web session registers an IANA time zone on connect (the default is a deployment setting, not a documented value); the prompt builder renders local time and elapsed-since-previous-turn per session.

**Input routing doctrine (three channels; the anti-Rasa).** (1) Voice: all ASR text → 4B router → {ignore | FSM event | escalate to the voice-slot model}. ASR text never reaches the research pipeline except through the voice pipeline. (2) Text: all text → 4B as arbiter → {start | queue | interrupt | FSM command}; never discard; ~5 s default-to-queue timeout; illegal FSM commands return to the arbiter. (3) Rubber-duck powers: the voice-slot model may answer, write a blackboard note, or issue start/queue/interrupt imperatives directly to the research command queue. Response modality follows input modality. The validator ratifies every proposed transition and returns rejections to the proposer with a hint, bounded.

**Dual-pipeline interplay.** *Notes-up*: distilled context from the voice pipeline into the research blackboard. *Read-down*: research state into the voice pipeline — status board into Layer 1 at every loop boundary, plus deep transcript access via Layer 3. *Imperative channel*: start/queue/interrupt commands from voice or text executed by the research pipeline's command queue. Machinery: asyncio inbox, pending counter, boundary condition with timeout, watcher task that cancels generation on interrupt. The research model resteers itself.

**Serving policy.** The OpenAI-compatible endpoint is the seam; the engine behind it is chosen by the eval harness per role: short-context, high-concurrency work (voice turns, agent fan-out) favors a speculative-decoding vLLM configuration; deep-context research favors a steady engine that does not depend on drafter acceptance. Two cards default to one model per card. Tensor parallelism is an experiment, not a plan. A throughput number is recorded with the prompt and context depth that produced it.

**Capability cards.** One versioned row per (model, quantization, serving config) in `capability_cards`, written by the nightly eval run: per-slice accuracy (FSM routing, voice recall, research multi-hop, coding patch, structured-output validity), latency percentiles at stated context depths, tool-call validity rate, loop rate, stop-compliance rate, cost per task. The router reads cards for escalation thresholds; the external bridge reads them for tier selection; a card revision that regresses a slice opens a review.

**Routing to remote models (external API bridge).** Explicit request → task criticality → local confidence cross-checked by cheap verification → complexity estimate → budget; tiers chosen from capability cards; payloads outward are extracted text, JSON, or Markdown.

**Panel mode.** Identical-spec parallel execution with an arbiter (judge-selects, vote, synthesize, test-harness-wins), or decision-point consensus; every fan-out recorded; reserved for complex, critical, or sensitive tasks; available inside heads-down mode; Qwen self-fusion before Gemma exists.

**Thinking policy.** Same weights, per-request settings: voice thinking off or small budget; research thinking high with preservation.

**Diagram conventions (house colorway, semantics assigned).** Colors mark logic roles: `action` (green) = LLM inference nodes; `handler` (brown) = deterministic code; `decision` (orange diamond) = deterministic dispatch/branch points; `server` (grey cylinder) = data stores; `form` (blue) = pipelines and subsystems shown as a unit; `startEnd` (thick green stadium) = entry/exit. Dotted edges = load-once/startup relationships only. Subgraphs mark pipeline boundaries; a labeled edge crossing a subgraph is a deliberate crossing. Source lives in `docs/diagrams/`; the README embeds an exported image.

**Public-documents hygiene.** Project documents contain no secrets, hostnames, credentials, employer names, or third-party personal names; deployment defaults (time zone, ports, paths) live in configuration; design notes that reference an employer or a compliance context stay out of the public repository; secret scanning runs on every push; `.env`, `api_key.txt`, and model directories are gitignored.

**Ontology v1 sketch.** Entities: Silo, Person, Device, Project, Conversation, Turn, Document, DocumentNode, Fact, Task, Tool, ModelCall. Relations: spoke (Person→Turn), belongs_to (Turn/Document/Fact/Task→Project→Silo), cites, supersedes, prefers/decided/scheduled (Person→Fact by facet), assigned_to, produced_by (Turn→ModelCall), member_of (Person→Silo). Every Fact carries facet, valid_from, valid_to, recorded_at, source, silo_id. Start strict.

## VRAM ledger (keep current)

| Card | Resident | Approx. |
|---|---|---|
| RTX 3090 (24 GB), Phase 1 text-mode | Qwen3.8-27B Int4 ~15-17 GB + KV ~2 GB; 4B router Q4 ~2.7 GB via llama.cpp; embedder on CPU | ~22 GB |
| RTX 3090 (24 GB), Phase 2 interim | Speech stack ~8-9 GB + 4B router ~2.7 GB; 27B swapped in only when no speech session needs the stack | ~12 GB (+17 when 27B loaded) |
| RTX 3090 #2 (planned) | Speech stack + Gemma 4 (or heads-down model); card one holds the 27B full-time; embedder and reranker move to GPU; one model per card by default | ~20-22 GB |
