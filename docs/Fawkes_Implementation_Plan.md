# Fawkes Implementation Plan
Iteration 2 (the LLM + statechart rebuild), structured as phases. Each phase has a goal, contents, and an exit test. Phases are intentionally re-orderable at their edges; the plan is a living document — revise it, don't obey it. Not every wishlist item ships in its nominal phase; the rule is that no phase forecloses a later item (schema first, analytics later). Updated 2026-09-20.

**Model/runtime baseline (current):** Qwen3.8-27B (Int4/AWQ-class) on vLLM (>= 0.19-class), OpenAI-compatible endpoint, thinking budget configured per pipeline (off/low for voice, high for research). Qwen3-4B-Instruct-class router/arbiter via llama.cpp; bge-m3-class embedding model on CPU until the second GPU; optional reranker. All tools are bound to pipeline slots, not to models. The voice slot is Qwen through Phase 3 and Gemma 4 from Phase 4; the research slot is Qwen.

**Hardware sequencing (one RTX 3090 until card two):** Phase 1 is text-mode with the speech stack down: vLLM at ~0.80 memory utilization holds the 27B Int4 (~16-17 GB) plus KV cache at 16K context (~2 GB), the 4B router runs beside it under llama.cpp (~2.7 GB), and the embedder runs on CPU — roughly 22 GB in use. If this thrashes, develop in split mode. Phase 2 needs the speech stack and the 27B together, which one card cannot hold: FSM voice flows run on the 4B alone and the 27B is swapped in for open conversation when no speech session needs the stack. Full concurrency lands with card two, where the default layout pins one model per card; tensor parallelism is an experiment gated on verified NVLink, x16 slots, and PSU headroom, adopted only on measured gain.

## Phase 0 — Milestone Zero (one afternoon)
Goal: first running code of Iteration 2; break the no-code streak; make the schema silo-compatible before any data exists, at Tier A cost only.
- Add `postgres` (with pgvector) service to docker-compose.
- Migration 001: `silos` (seed one default), `projects` keyed `(silo_id, project_id)`, `user_silos`, `transcripts`, `facts` (bi-temporal), `documents`, `document_nodes`, `model_calls`, `traces`, `tunnels` and `blocked_pairs` (single shared `silo_id` column) — all with non-null `silo_id`, composite foreign keys to projects, `silo_id` in every unique constraint and index prefix, content hash unique per silo, `timestamptz` UTC everywhere, enum seed tables. No partitioning and no silo-keyed row-level security yet (Tier B, deferred to the first real second silo).
- `MemoryStore` skeleton taking an immutable silo context on every method: `remember_utterance()`, `recall()` (hybrid BM25+vector, Reciprocal Rank Fusion).
- Tests: insert three utterances, retrieve by meaning and by keyword, assert timestamps and provenance; ingest in silo A and query as silo B through facts, full-text, and vector paths, assert empty; attempt a cross-silo tunnel insert, assert the constraint fails.
Exit: `pytest` green; a transcript survives a container restart; the silo leakage tests are permanent markers.

## Phase 1 — Cognitive core + substrate foundation
Goal: the new brain exists; the old Rasa functionality is reachable through it in text, while iteration 1 keeps running in parallel.
- Prompt builder (local-time and elapsed-time rendering; silo assertion on every context item) + validator skeleton with bounded re-prompt of the proposer; structured-output JSON contract with a confidence field; FSM registry data structures.
- Small-model router v1 for the voice channel: {FSM event | ignore/backchannel | escalate} with confidence; wired before the voice-slot model.
- vLLM serving for the 27B; llama.cpp serving the 4B on the same card; prompt prefix ordered for cache stability. Serving-recipe experiment: stock vLLM vs the tuned vLLM fork vs the tuned llama.cpp fork, scored by the eval harness; winner recorded in the toolbox with data.
- Ontology v1 (`ontology.md` + enum tables + constraints), evolved only via versioned migrations.
- `ingest()` v1 with content hashing, provenance, silo/project stamps, and fan-out handler registry; turns, traces, and model-call cost rows all enter through it.
- Eval harness v0: ~30 seed cases sliced by task family with deterministic scoring (labeled routing verdicts, seeded recall facts with expected citations, fixed-corpus multi-hop citation ids, schema validation, loop/stop/latency observation); the coding-patch slice arrives in Phase 3 with the coding surface; nightly smoke subset plus full runs on fingerprint change and weekly; first capability cards written to `capability_cards`; `feedback_events` table created (schema first) so corrections, re-asks, and thumbs can be captured as soon as a channel produces them.
- CI pipeline (GitHub Actions): unit tests for the deterministic core (test-first), integration tests against Postgres, mocked-LLM FSM lifecycle walkthroughs via a shared `FakeLLM` fixture, silo leakage tests, secret scanning on every push; Phase 0 and Phase 1 exit tests as permanent pytest markers.
- Observability schema: one JSON trace per turn/call including tokens in/out/cached, model, task id, silo, and (for external calls) cost; backup cron for Postgres.
Exit: text-mode conversation through the statechart passes the eval set's FSM and recall slices; every turn produces a trace record; a first capability card exists for each served model; CI green.

## Phase 2 — Voice memory + identity (voice is primary)
Goal: full voice loop on the new core, multi-user, with real memory; Rasa retired.
- Layers 0-2 tiered loading; semantic cache keyed by silo; Layer-3 tool calls.
- Enrollment FSM ported; voice-clone FSM ported; auth FSM (deferred identification tiers, recency-modulated thresholds, passphrase); Rasa switched off once lifecycle tests and eval slices pass.
- Identity outside silos: user_silos membership with a default silo per user; project resolution from voice (active project in Layer 1 or explicit command) — settled before the enrollment port.
- Operational data migration: iteration-1 DuckDB tables into Postgres; ECAPA matrix built from Postgres at startup; DuckDB retired.
- Voice register: thinking off/budgeted, brevity persona, stall words; correction-event hook that also carries research results and self-corrections back to the voice channel.
- Memory-promotion hook (small model) after each turn.
- Sandboxed tool container.
- Interim single-GPU fallback: FSM voice flows on the 4B alone whenever the 27B cannot be co-resident.
Exit: two different speakers hold personalized conversations on separate devices; enrollment completes end-to-end; a fact stored yesterday is recalled with citation and correct local date today; voice-path pre-LLM overhead measured under ~150 ms; lifecycle walkthroughs for all ported FSMs green in CI.

## Phase 3 — Ingestion, research pipeline, external hands, dual-pipeline interplay
Goal: Fawkes reads, researches, codes, talks to the outside world, and the voice pipeline rides alongside the research pipeline.
- Ingestion source adapters: file drops and coding-session transcripts → `ingest()` (browser and terminal adapters deferred to Phase 6+); ingestion router (structure scoring; Qwen-vision OCR at ingest); document-summary selection layer; PageIndex-style trees in `document_nodes` (checksum-cached).
- Research pipeline v1: agentic loop with grep/FTS/tree/web tools, CRAG grading, web fallback; recall ladder with context expansion and citations; job checkpoints; text responses to text clients; correction events for voice-initiated tasks; ~4 medium-depth workers as the fan-out design point.
- Code tools: ripgrep + fuzzy + tree-sitter tiers; adopt code-graph MCP servers as shipped (code-review-graph for blast radius and impact; graphify for the multimodal bird's-eye view), per-commit incremental updates, consumed by OpenCode and the research pipeline.
- Router extended to the text channel as arbiter: {start | queue | interrupt | FSM command}, never discard; ~5 s default-to-queue timeout; illegal FSM commands return to the arbiter.
- Rubber-duck interplay v1 on the single served Qwen: blackboard (notes-up), imperative channel direct to the research command queue (start/queue/interrupt), read-down status board + deep transcript access, arbiter machinery with out-of-band cancel; research model resteers itself.
- Consolidation/compaction jobs + scheduler (idle-window arbitration).
- OpenCode integrated as coding surface; its transcripts ingested.
- Coding-patch eval slice: 10-15 hand-authored micro-repositories (5-30 files each, a task statement, failing tests) drawn from our own past bugs, stored under `tests/eval/repos/`, optionally supplemented by a small public subset (HumanEval+ or Aider polyglot); the harness copies each to a temp directory, runs the coding loop, applies the patch, runs the suite; weekly cadence, and included in full card runs.
- MCP server over MemoryStore (scoped auth by silo, user, project; audit log).
- External API bridge: Claude escalation tool with local-first routing policy, tier selection from capability cards, OCR-first text/JSON/Markdown payloads, budget gate, per-call cost accounting, silo-stamped audit log; remote tiers scored once per model release and spot-checked monthly on a capped sample.
- Salience weights v1; contradiction detector v1 (log-level); per-task token/cost rollups and expensive-task flags.
- (~Phase 3.5, as VRAM pressure dictates) SSM/Mamba ASR swap for interim and final slots.
Exit: a multi-document research question answered with correct citations; a coding task completed in OpenCode with Fawkes memory via MCP and code-graph context; one Claude escalation round-trip carrying extracted text; a voice question about the live research session answered from the status board; a mid-flight correction delivered through the arbiter; a text-channel message started, queued, and interrupting in three separate tests; a voice-initiated research result delivered by voice.

## Phase 4 — Synthesis layer + second GPU + measured experiments
Goal: compounding knowledge; the dual-model architecture; let the eval harness adjudicate deferred debates.
- Wiki distillation layer, OKF-conformant bundles, lifecycle states, lint cron.
- Tunnel manager full lifecycle (within silo only); contradiction detector escalation to clarification FSM.
- Gemma 4 stood up on the second RTX 3090 as the voice-slot model (one model per card by default); listening tests vs tuned Qwen; tensor-parallel experiment only if NVLink is verified and the harness shows a gain.
- Panel mode: identical-spec parallel execution with arbiter strategies and decision-point consensus; reserved for complex, critical, or sensitive tasks.
- Heads-down mode: swap the conversational slot for a second coding model (GLM-class) or Harness-1 by task mode; panel mode remains available inside heads-down mode.
- Reconciliation crons; type histograms.
- Benchmarks: wiki vs Cog-RAG; hand-rolled research loop vs deepagents; Harness-1 as retrieval subagent; code-graph MCP vs grep-only code tools.
Exit: wiki answers a corpus-theme question from compiled pages with source descent; a panel-mode decision recorded with both candidates and the arbiter's verdict; one benchmark decision recorded in the toolbox with data.

## Phase 5 — Web interface, multi-surface, back-catalog
Goal: Fawkes beyond the microphone; Fawkes as the daily research and coding driver.
- Web app: chat, uploads, project creation/notes, authentication/sessions, optional model-target selector; security hardening pass.
- Row-level security keyed to user for additional users; per-project visibility. If a second silo is wanted, Tier B lands here: list partitioning by silo, silo-keyed row-level security under a non-owner role, instance pin to an allowed silo set.
- Voice-driven project management commands.
- Back-catalog import: Claude/ChatGPT exports via an `ingest()` adapter with per-day summaries, embeddings, and project assignment.
- Mobile-facing API; remaining MCP surface; Claude handoff packaging.
Exit: a project created by voice is visible and editable on the web by an authenticated user; a second user sees only their scope; if a second silo exists, it passes the leakage tests; an imported historical conversation is recalled with its original date and citation.

## Phase 6+ — Frontier (planned, not scheduled)
- SSM/Mamba ASR swap if not already pulled into ~3.5.
- Browser text-extraction sidecar and terminal capture adapters (if still wanted once the KVM/HID track exists).
- Screen-context subsystem; HUD foundation.
- Context-compression proxy for external traffic — deferred indefinitely on measured evidence.
- Digest/scrape crons; Microsoft To Do; Maps/Waze; home automation; CV pipeline.
- Graphiti if bi-temporal SQL hits its ceiling; serving scale-out; power management tuning; fine-tuning experiments only on demonstrated need.

## Separate tracks (not scheduled here)
- Cocktail Party speaker separation (own repository); KVM/HID computer control (own repository, own milestone zero, begins no earlier than Fawkes Phase 3).

## Testing and evaluation cadence (all phases)

| Cadence | Where | What runs | Purpose |
|---|---|---|---|
| Per commit | GitHub Actions, no GPU, seconds | Unit tests for the deterministic core; integration tests against Postgres; FSM lifecycle walkthroughs against the shared `FakeLLM` fixture; schema contract tests; silo leakage tests; arbiter protocol tests; secret scanning | Regression armor; a phase exit test never leaves this suite |
| Nightly | Workstation, GPU, minutes | Smoke eval: a fixed ~10-case subset (routing verdicts, seeded recall, structured-output validity) against the currently served local models, plus latency timings; results appended to the eval log, no new card unless the fingerprint changed | Catch drift from prompt, harness, or config edits within a day |
| On fingerprint change (as needed) | Workstation, GPU, tens of minutes | Full local capability-card run: every applicable slice, several sampling seeds → new card version | Re-measure whenever model, quant, serving config, prompt version, or index contents change |
| Weekly | Workstation, GPU | Full card run even without a fingerprint change; the coding-patch suite (once it exists, Phase 3) runs here, not nightly | Catch environmental drift (drivers, libraries, index growth) |
| Per remote model release, plus a monthly capped spot-check | Anthropic API, budgeted | Remote-tier cards on the research-tier slices only (multi-hop, complex reasoning, coding patch if remote coding escalation is enabled) — never routing or voice-recall slices, which no remote model ever serves | Keep tier selection honest at bounded cost |
| Opportunistic | Whenever panel mode runs | Per-model tokens, loops, compute time, and judge rubric grades on the same real task, written to `panel_results` and summarized onto cards as field observations, labeled as uncontrolled | Head-to-head evidence on real tasks that the seed set cannot produce |

- The general eval harness (all local slices, all cadences above) and the card-stamping routine (the subset that produces a versioned card row) are the same code; a card is simply the harness output frozen at a fingerprint.
- Feedback capture is schema-first: a `feedback_events` table exists from Phase 1 (turn_id, task_id, kind: thumbs_up / thumbs_down / explicit_correction / re_ask / confirm / abandon, source channel, payload, silo). Phase 2 fills it from voice via the router (a correction class: "no, I meant…", re-asks of the same intent within a window, verbal thumbs) and the promotion hook; Phase 5 adds thumbs up/down in the web interface. Every feedback event links a turn to a task, so exchanges-before-answer is derivable later from transcripts alone.
- Test-first for the deterministic core; prompts iterate against the eval harness instead.

## Standing constraints (all phases)
- Multi-user support never regresses; silo isolation never regresses.
- No reasoning-driven retrieval on the voice hot path.
- ASR text never reaches the research pipeline except through the voice pipeline; text-channel messages are never discarded; voice in means voice out, text in means text out.
- Any memory transformation writes a manifest; any retrieval writes a trace; any model call records tokens and cost.
- Every adopted component sits behind a seam; tools are bound to slots, not models.
- Outbound cloud payloads are extracted text and structured data unless visual judgment is the task.
- Project documents contain nothing that cannot be published.
- Document set (wishlist, principles, inventory, plan, toolbox, handoff, architecture specification, diagram sources) updated in full whenever a decision changes it, with an explanation of what changed and why.
