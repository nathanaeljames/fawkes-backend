# Fawkes Diagram Standards
Version 0.3 — 2026-09-28. Adopted by the owner after Fable's review of the 2026-09-21 draft (v0.2). Supersedes v0.2. The v0.2 header's statement that its legend had been accepted is struck: nothing in a reviewer draft is adopted until it is adopted in the project record. This document governs how Fawkes diagrams are drawn; it does not change Fawkes behavior. Companion documents (base file names): Fawkes_Architecture_Specification.md, Fawkes_Behavior_Contract_Fable.md, Fawkes_Toolbox.md.

## 1. Purpose and authority

A diagram is a view of the architecture for a stated reader and question. It is not the architecture and need not show every detail. The Architecture Specification and the decision record in the Session Handoff are authoritative; the Implementation Plan supplies sequencing; this document supplies drawing rules. If they conflict, report the conflict — never invent a connection or silently pick a winner. Agreement between reviewers cannot override an explicit invariant.

Every diagram declares, in a caption or in source comments:

- Audience and the question it answers.
- View type: overview, component relationships, scenario flow, statechart, data model, or deployment.
- Scope and phase: planned target, current implementation, or a named migration step.
- **The exact specification version it was drawn from** (for example "Specification v1.10") and any decision-record dates relied on.
- What has been grouped or omitted, and any unresolved assumption affecting a drawn path.

Component identifiers (C-nn, D-nn, P-nn from the specification) appear in source comments or in a mapping table; README-level images may use human-readable names. A repeated role of one component is not a second deployment or a second model instance.

## 2. Complexity levels and the three views

Diagrams are drawn at one of three declared complexity levels on the project's 1-10 scale:

| Level | Use | Working size | Reader |
|---|---|---|---|
| 4 | README overview: how Fawkes fits together, named models, the sanctioned crossings, a few decision gates | Roughly 12-18 visible nodes; readability governs, not the count | A visitor with ten seconds |
| 7 | Engineering reference: component ownership, interfaces, stores, every boundary crossing labeled | Roughly 30-40 nodes with subgraphs | An engineer navigating the system |
| 10 | Exhaustive: every component and pathway in the specification, one node per moving part | Unbounded; rendered only with mermaid.live or mermaid-cli | The reviewer checking coverage |

Sources live in `docs/diagrams/` as `architecture_L4.mmd`, `architecture_L7.mmd`, and `architecture_L10.mmd`, plus one file per detail view. A detail view (scenario, statechart, or data model) contains only the participants, states, or entities needed for its named question. The README may combine subsystem boxes with a few routing diamonds; that is a declared overview convention, not incompleteness. Do not expand boxes to make a graph look exhaustive, and do not create a chart because a component exists or a target count is unmet; node and edge counts are warning signals, not split rules.

The C4 idea of separate zoom levels and of dynamic views for selected interactions is the organizing principle; these flowcharts are not formal C4 diagrams.

## 3. Legend and node classification

The six house classDefs and their hex values are canonical and are used exactly as written in every Fawkes diagram; the semantics below are assigned to them. Text labels must remain understandable without color, and shape carries meaning where color is unavailable. Phase, trust boundary, local-versus-cloud placement, and implementation status are separate dimensions expressed by labels or boundaries, never by color.

| Class | Shape | Meaning |
|---|---|---|
| `action` | Rectangle | A learned-model inference call: language, speech, embedding, vision, speaker-matching, and reranking models. "Model versus code" is the test, not stochastic versus deterministic. |
| `handler` | Rectangle | Application logic: assembly, validation, persistence coordination, queue manipulation, tool execution, `ingest()`. |
| `decision` | Diamond | A branch over an available condition or verdict; outgoing edges name the alternatives. |
| `server` | Cylinder | Logical stored state; label it authoritative, derived, or ephemeral when relevant. |
| `form` | Rectangle | A deliberately collapsed subsystem containing several responsibilities. |
| `startEnd` | Stadium | A workflow entry or exit: the user, client, source, or outcome. |

Canonical class definitions (append inside every flowchart):

```text
classDef startEnd fill:#2d5016,stroke:#4a7c1f,color:#e8f5e9,stroke-width:3px
classDef form fill:#1a3d5c,stroke:#2e5c8b,color:#e3f2fd,stroke-width:2px
classDef action fill:#4a7c59,stroke:#6b9b7a,color:#e8f5e9,stroke-width:2px
classDef handler fill:#5c4a1a,stroke:#8b7520,color:#fff9c4,stroke-width:2px
classDef decision fill:#7c5a1a,stroke:#a8762b,color:#fff9c4,stroke-width:2px
classDef server fill:#696969,stroke:#808080,color:#f5f5f5,stroke-width:2px
```

Classification rules:

1. A learned model is `action` even if its inference is reproducible; a separately drawn ASR, embedder, or speaker matcher is green.
2. A speech front end containing models, buffers, and streaming logic is a blue `form` when collapsed; expand it into green model nodes and brown handlers only in a detail view.
3. An LLM choosing a routing verdict is green; the code dispatching on that verdict is an orange diamond. A complicated operation is not automatically a decision.
4. A loop boundary that drains a queue, checkpoints, and refreshes a status board is a handler; its "pending items?" test may be a separate diamond in a detail view.
5. Wiki lint, relevance grading, and consolidation are colored by their actual implementation, or drawn as a blue subsystem while mixed or undecided.
6. A cylinder is a store, not a generic server; split an adopted service from its database only when that distinction answers the diagram's question.
7. At level 4, both pipelines may be blue subsystems naming their model bindings inside them; this does not change the green-inference meaning at levels 7 and 10.
8. A whole statechart controller is a handler or a collapsed subsystem; one test such as "transition legal?" is a diamond. The controller and a decision it performs are different subjects.
9. `ingest()` is an application write boundary and is drawn as a handler; Postgres is the authoritative store and is drawn as a cylinder; the `derived_jobs` table is a store, drawn only when the diagram's question concerns durability or fan-out.

## 4. Arrows and boundaries

- Solid directed arrows represent runtime relationships. Label non-obvious connections with a verb and, where helpful, the payload: `persist turn`, `read status`, `queue correction`, `approved transition`.
- Each diagram states whether its arrows are relationships or ordered control flow; never infer a global execution order from a component map.
- **Dotted arrows are reserved for initialization-only dependencies**, labeled `startup` or `load once` (for example, the speaker matrix loaded from Postgres at boot). They never mean retry, rejection, optional, asynchronous, runtime refresh, or future work.
- Retries and asynchronous messages are solid arrows with explicit labels and a stated bound (for example, `constraint hint, at most 2 retries`).
- If a relationship occurs both at startup and at runtime, draw two labeled arrows in a detail view or one solid relationship in an overview.
- Bidirectional arrows are acceptable in an overview for an explicitly summarized exchange; separate the directions in a scenario view when payloads or authorization differ.
- A decision in a detailed scenario shows every outcome that affects the scenario, including rejection and failure; an overview may omit low-level alternatives only through declared abstraction.
- A subgraph is a grouping boundary on one diagram; it is not automatically a process, service, database, or deployment boundary. State which kind it represents.
- **Listen-along and other non-transcribed audio paths** are solid arrows labeled `listen-along, no transcription` (or `binding events only`), never dotted.
- **Sanctioned boundary crossings** are the only labeled edges permitted between pipeline subgraphs, and they use the specification's names: `notes-up` (voice → blackboard), `read-down` / `status board` (research → voice Layer 1), `start / queue / interrupt` (voice or text → research command queue), `research results / self-correction` (research → voice for voice-initiated tasks), and `escalate` (research → external bridge). An edge between pipelines without one of these labels is a defect.

## 5. When a separate detail view earns its place

Create a detail view when it answers a concrete question the overview obscures: who owns a state transition, authorization decision, or stored record; what happens when an interrupt races a timeout or a completion; which step must commit before another may safely occur; what survives a restart and how derived state is rebuilt; which states, guards, and exits define a workflow.

A useful detail view has a short purpose, a coherent boundary, stable external interfaces, and an identifiable reader or test. Expand internal machinery while preserving the parent's observable behavior. At normal reading size the reader follows the main path and identifies its outputs without visiting another diagram.

Grouping guidance: group by responsibility, interface ownership, and state ownership; keep one level of abstraction within a group; separate structural questions from temporal and concurrency questions; use a sequence or scenario view for timing, a statechart for lifecycle, a data model for identity and ownership; add deployment views only when hardware or process placement is the question.

**Workflows are statecharts.** Enrollment, voice cloning, authentication, clarification, and any future FSM are drawn with Mermaid `stateDiagram-v2` (states, guards, entry and exit actions, and exits are first-class), not as flowcharts. Colors do not apply to state diagrams; guards are written on transitions.

## 6. Fawkes profile

The README target is one level-4 overview answering "how does Fawkes fit together?" It keeps understandable: the major model names, channel-correct delivery, shared validation, the voice/research interaction, shared ingestion and Postgres, and governed external escalation. Model bindings and phases are described as planned, not as installed software. The profile prescribes neither coordinates nor a graph topology nor a subgraph count.

Five areas for optional deeper views: voice, text, research and coordination, memory and ingestion, and statechart workflows. Serving, identity, external integration, and observability receive views when a real question requires one.

Every simplification preserves these semantics (from the specification's invariants):

- Voice ASR reaches research only through the voice pipeline.
- Text input has no discard verdict.
- Model output proposes; deterministic validation authorizes FSM slot and transition mutations; rejected proposals return to the original proposer with a bound.
- Responses and background completions retain their originating channel.
- Blackboard notes and research commands are different channels.
- Voice retrieval is deterministic or single-shot; research may iterate.
- Content reads use MemoryStore; content writes use `ingest()`; no component invents a bypass. The voice runtime, the research runtime, and the statechart controller are independent callers of the shared write boundary; transcript logging is not conditional on an accepted transition or a tool call; ignored turns and rejected attempts are still recorded.
- Ingestion and Postgres appear as connected participants whenever storage is in scope; prefer one occurrence of each component, and label and justify any repeat.
- Content carries its scope (workspace, user, or silo); ranking weights never expand authorization.
- Postgres holds authoritative Fawkes memory; derived artifacts name their rebuild source; source repositories are the authority for repository-derived code graphs.
- Cloud inference uses the governed bridge with payload, budget, and accounting controls.

An overview may state these as a short caption instead of drawing every audit arrow; a drawn edge may not contradict them. Missing detail is not a defect; a contradictory path is.

## 7. Independent review of diagrams

Independent blind reconstruction was used once, for the first behavioral review, and is not a standing requirement now that the behavior contracts have converged. It is used again only when a new area is opened for review: give reviewers the same frozen sources and a neutral scenario worksheet, withhold prior diagrams and this document until first-pass answers are saved, and ask for cited clauses and marked gaps rather than silent decisions.

When diagrams are compared, normalize aliases, grouped internals, and repeated role views, then compare allowed outcomes, prohibited paths, authority, and persistence. Coordinates, source order, edge routing, and subgraph count are never correctness metrics. Classify each difference as presentation, permitted abstraction, diagram error against a clear requirement, unresolved specification, or proposed design change; only the last two drive architecture changes. Two reviewers can share an error; pairwise agreement and conformance to the specification are different measures.

## 8. Rendering and portability

- The editable source is the versioned artifact: `.mmd` for Mermaid (the project standard; `.drawio` or `.d2` only for views Mermaid cannot express). Exports are regenerated from source by mermaid-cli in CI so they never drift; a matching PNG is stored for the README, with an SVG optionally alongside for zoom.
- Record the renderer version and configuration used for exports; syntax success does not prove readability or semantics.
- `<br/>` line breaks inside node labels are permitted and expected; they are supported by GitHub, mermaid.live, and mermaid-cli and are the only practical way to keep labels legible at README width. Use quoted labels for any text containing parentheses or colons.
- GitHub's built-in renderer lays diagrams out differently from mermaid.live and cannot be made to match; the README embeds the exported image and links to the source.
- Keep overviews top-down and short-labeled; avoid more than five nodes across.
- Inspect the exported image for clipped text, crossing labels, and illegible scaling; check that shape and label carry the meaning when colors are unavailable (a color-blind-safe reading must survive).
- Reuse the palette consistently; never use one dotted style for several meanings.
- After a material revision, deliver exports under a visibly new revision filename and keep the version history.

Historical note: the draw.io exports produced before this version used the same house colorway with speech nodes drawn as handlers; they predate rule 1's green-for-all-models clarification and are not examples of v0.3 compliance.

## 9. Review cadence and a stopping rule

At each phase start, review only the contracts and views needed for that phase. At phase exit, compare the relevant diagram with implemented behavior and permanent tests. Keep the README stable unless the architectural story changes. Generate a new scenario or detail view when a concurrency, state, persistence, authorization, or deployment question arises; do not demand a new exhaustive graph at every milestone.

Stop the planning loop when the critical contracts for the next slice are settled, known uncertainties are assigned tests or later owners, and further drawing changes only presentation. Move evidence gathering into code and tests at that point.
