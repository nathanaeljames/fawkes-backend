# <ROUND> — <title>
Base commit: <sha>. Mode: dual | solo. Opened: <date>. Dependencies: <rounds>.

## Scope
What this round implements, naming the specification clauses (C-, D-, P-, I-, X-) and plan bullets.

## Out of scope
What is explicitly not done here, and which round owns it.

## Behavior-contract cases
BC-nn …

## Tests (pipeline-owned; see manifest.json)
- tests/<path>::<test> — what it proves; which BC case
Run: `pytest --runxfail -m <ROUND>`; regression: `pytest`.

## Interfaces and fixtures
Signatures, fixtures, and any test doubles the implementer may use for components that arrive in later rounds.

## Own (viva functions)
- `function_name()` — why it matters

## Acceptance checklist
- [ ] every test above passes for real, with a prove-it-bites line in RESULTS
- [ ] regression suite green
- [ ] invariants: one write path; MemoryStore reads; X-01 for language models only; scope on every content row
- [ ] BUILD written by round.py with transcript path

## Questions answered before opening
Q: … A (owner, date): …
