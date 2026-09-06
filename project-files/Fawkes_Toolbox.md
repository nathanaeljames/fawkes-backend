# Fawkes Toolbox
Every tool, model, and framework vetted across this research round, sorted into: **Adopted** (committed), **Watchlist** (possible future / experiment first), **Pass** (not a fit). Reasons are one line; provenance is the conversation record. Last updated 2026-08-31.

## Adopted — critical path

| Tool | Role | Note |
|---|---|---|
| Qwen3.8-27B (Apache 2.0) | Primary cognition, vision/OCR | Dense 27B, 262K ctx, vision+video; Int4 fits a 3090; default reasoning "xhigh" — set thinking budgets per pipeline; vendor benchmarks pending independent confirmation |
| vLLM | Serving runtime | OpenAI-compatible endpoint, concurrency-class serving, official Qwen recipes; prefix caching on hybrid-attention layers still experimental — measure |
| Qwen 4B-class small model | Router, extraction, memory-promotion hook, background jobs | Confidence-gated escalation to 27B; the anti-Rasa: interpretive, open-set, with an "ignore" class |
| bge-m3-class embedding model | Embeddings for pgvector | Local, ~1-2 GB |
| bge-reranker-v2-m3-class | Optional rerank stage (research path; voice only if measured cheap) | Sub-1B cross-encoder |
| Postgres + pgvector + native FTS | System of record + hybrid retrieval | Multi-process concurrency for the multi-user mandate; row-level security for Phase 5 scoping; one Compose service |
| DuckDB | Speech-side operational data (speakers, pangrams), analytics scratch | Retained in current role; not the memory SoR |
| OpenCode | Coding surface (Phase 3) | Adopted, not built; local-endpoint-first; transcripts ingested |
| OKF (Open Knowledge Format) | Wiki bundle format (Phase 4) | Google's June 2026 spec formalizing the LLM-wiki pattern: markdown + YAML frontmatter + index.md; adopt the format, keep our maintenance loops |
| MCP (Model Context Protocol) | Boundary to external agent hosts (OpenCode, Claude) | Server over MemoryStore with scopes + audit; NOT used between internal components |
| OpenAI-compatible `/v1/chat/completions` | Model-serving seam | The boundary that makes runtimes swappable; distinct from MCP |
| Anthropic API (API-key) | Claude escalation tool | Third-party API-key use is fully supported; the 2026 restriction was subscription-OAuth in harnesses, not the API |
| Docker sandbox | Tool execution isolation | No agent access to project root |
| FastAPI / plain Python | Internal service boundaries | MCP is for the outside; Python for the inside |
| pytest + seeded eval sets | Eval harness | The decision instrument; behavioral checks from the harness-quality checklist |
| Mermaid.js | Diagrams | House colorway per preferences |

## Adopted — patterns (harvested, not installed)

| Source | Pattern taken |
|---|---|
| MNEMOS | Manifest-on-transform, versioning/audit contracts, ACID-SoR doctrine |
| MemPalace | Layered injection (0-3), wings/rooms→scoped weights, tunnels (improved: lifecycle + reinforcement), raw-text-beats-summaries evidence |
| Letta/MemGPT | Memory-paging vocabulary; LLM-invoked deep search as Layer 3 |
| Hermes / SimonScrapes stack | Frozen-snapshot injection, per-turn memory-promotion hook, back-catalog import, recall ladder with context expansion, cite-or-admit |
| GBrain | Citation discipline; honest "not found" |
| Claude Code / deepagents | Context offloading, tool lazy-loading idea, compaction triggers, subagent isolation |
| Cline | Three-tier code retrieval (ripgrep / fuzzy / tree-sitter) for the research pipeline's code tools |
| PageIndex / Alpha Iterations | Document-tree building (own implementation, pymupdf4llm-style parsing control, checksum-cached) |
| Sudip P. | One-write-fan-out ingestion contract; reconciliation sampling; "graph DB only when a real multi-hop query exists" |
| Agent Native | Harness-quality checklist; quant-behavior evaluation; API-boundary reference architecture |

## Watchlist — possible future, experiment before commitment

| Tool | Trigger to revisit |
|---|---|
| Harness-1 (UIUC/Berkeley/Chroma, Apache 2.0) | Retrieval subagent when third GPU / spare 16 GB exists; environment is open — benchmark on our corpus (Phase 4) |
| Cog-RAG (official repo) | Benchmark vs wiki on research corpus (Phase 4); OpenAI-compatible config → local Qwen |
| DeepSeek Harness (dsh) | Coding-surface alternative to OpenCode; MIT, everything-is-a-plugin, append-only session log; developer preview with breaking changes — let it mature |
| Gemma 4 (31B dense / 26B-A4B) | Conversational register + second opinion; arrives with second RTX 3090 (Phase 4); adopt only if listening tests beat tuned Qwen |
| deepagents | Research-loop orchestration; evaluate vs hand-rolled loop (Phase 4); local-capable via OpenAI-compatible endpoints |
| Graphiti (OSS, Neo4j/FalkorDB) | Temporal knowledge graph when bi-temporal SQL strains: routine temporal multi-hop or failing entity resolution; clean backfill from facts table |
| LightRAG | If a lightweight entity/theme graph is wanted before/instead of hypergraphs |
| LangGraph | Only if research workflows need checkpoint/resume machinery we'd otherwise hand-build |
| vLLM structured-output / SGLang | If constrained decoding needs outgrow vLLM's guided decoding |
| Redis / Valkey | Multi-process shared cache/locks (Phase 5 era) |
| Qwen3.6-35B-A3B MoE | Throughput-biased alternative if measured latency demands; more VRAM, faster tokens |
| Triton Inference Server | Only for heterogeneous multi-model fleet serving — unlikely |
| DSPy | Prompt optimization once eval sets are rich (late) |
| Search-R1 / Tongyi DeepResearch | Retrieval-policy research references |
| Mistral OCR / Marker / MinerU | Bulk-ingestion OCR only if Qwen-vision throughput becomes the bottleneck |

## Pass — evaluated, not a fit

| Tool | Reason |
|---|---|
| Rasa | Replaced; closed-set intent classification is the canonical failure mode |
| LangChain (as core dependency) | Framework overhead in a hand-owned statechart core; patterns already absorbed |
| Ollama (production) | Convenience layer lagging model support (Qwen vision mmproj); llama.cpp direct or vLLM instead |
| llama.cpp (as primary runtime) | Superseded by the vLLM decision; retained only as fallback / tiny-model path |
| GGUF quants for the 27B | vLLM's GGUF path is experimental/out-of-tree; use Int4 AWQ/GPTQ-class checkpoints |
| Mem0 / Zep (as products) | Summary-first storage contradicts verbatim-first principle (Graphiti engine separately on watchlist) |
| MemPalace (as codebase) | Patterns harvested; SQLite triple store and consumer defaults not a fit |
| MNEMOS (as deployment) | Contracts adopted; full machinery overweight for this scale |
| Hermes / OpenClaw / SimonScrapes AgenticOS (as platforms) | Fawkes occupies this slot natively; patterns harvested |
| GBrain (as system) | Resolver pattern already present in Qwen+FSM; cron-and-skills machinery redundant |
| nano-graphrag / GraphRAG (full) | Heavy LLM indexing + staleness; theme layer served by wiki; community detection not needed at this scale |
| Hypergraph stores (as build target) | One synthesis layer at a time; Cog-RAG stays an experiment, not a second layer |
| ChromaDB | pgvector inside the ACID boundary covers it |
| Chroma Context-1 | Harness unreleased; superseded by Harness-1 on the watchlist |
| Adi Insights MoE-Mamba claims | Verified garbled-to-fabricated; hybrids (already inside Qwen3.x) are the real story |
| Regex thought-stripping (Elmali-style) | Structured output separates response_text by design; sanitize at the contract, not with regex |
| Hand-built coding harness | Adopt OpenCode/dsh-class instead; build voice+memory, not repo-map machinery |
| Cloud-first memory services | Local-by-default principle |

## VRAM ledger (keep current)

| Card | Resident | Approx. |
|---|---|---|
| RTX 3090 #1 (24 GB) | Speech stack: Canary-Qwen ~5 GB, XTTS ~1.8 GB, ECAPA/MarbleNet/FastConformer ~1-2 GB; small router 4B Int4 ~2.5-3 GB; embedder ~1-2 GB; reranker ~0.5 GB | ~12-15 GB, headroom for batches |
| RTX 3090 #2 (planned) | Qwen3.8-27B Int4 ~16-17 GB + KV cache | ~20-22 GB |
| GTX 1660 Super (6 GB) | Spillover experiments only | — |
| Future third slot / upgrade | Gemma 4 (Phase 4) or Harness-1 | — |
