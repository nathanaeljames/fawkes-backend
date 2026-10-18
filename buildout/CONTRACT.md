# Implementer contract

One file, both implementers, every round. The grader holds both to this text and nothing else. If a SPEC restates a rule from here, this file wins and the SPEC has a bug.

## What you receive
- `buildout/<ROUND>/SPEC.md`: scope, out-of-scope, the specification clauses implemented, dependencies, the contract cases, the test files, the viva functions.
- The round's tests, already committed, marked `xfail(strict=True, reason="<ROUND> open")`.
- A git worktree on branch `round/<ROUND>/<seat>` at the round's base commit. Your seat letter is A, B, or N.
- The current project documents in `docs/`. The Architecture Specification defines behavior; the Implementation Plan sequences it; the Behavior Contract supplies acceptance scenarios. Where the SPEC and the specification disagree, stop and say so in RESULTS; do not pick a winner.

## What you must do
1. Run `python tools/round.py open <ROUND> --seat <seat>` before touching code, and `python tools/round.py watch &` in the background so activity is recorded.
2. Implement the SPEC's scope and nothing else. Off-task recommendations go in RESULTS under "Recommendations", never in the diff.
3. Make the round's tests pass for real: `pytest --runxfail -m <ROUND>` must show passes, not expected failures. Run the full suite normally as well: `pytest`.
4. Prove every new assertion bites: remove the fix or inject the fault, watch the check fail, restore, and paste the failing line into RESULTS. An assertion nobody has watched fail is untested.
5. Diagnose first. If the SPEC's model of the code is wrong, inspect and report before reshaping.
6. Never silently absorb a surprise. Flag it in RESULTS even if you worked around it.
7. Match the existing style. No mass refactoring. Patch surgically. Keep functions small enough that the owner can explain them at the viva; cleverness the owner cannot explain is a defect.
8. When a component's shape changes, search the test harness for the old shape before calling the change done.
9. Run `python tools/round.py close <ROUND> --seat <seat> --transcript <path to your harness log> --inline-diff` as the last action. It writes `buildout/<ROUND>/BUILD-<seat>.md`.
10. Write `buildout/<ROUND>/RESULTS-<seat>.md` from the template.

## What you must not do
- Edit, weaken, skip, or re-mark any test file listed in the SPEC manifest. CI refuses the diff.
- Edit pipeline-owned files: `buildout/**` other than your own RESULTS and BUILD, `docs/**`, `tools/round.py`, `.github/**`.
- Write to the backend other than through `ingest()`; read memory other than through `MemoryStore`; call a model other than through the X-01 seam (language models) or the typed speech and embedding interfaces. These are specification invariants; CI tests some of them and the grader checks the rest.
- Commit or push. Your branch is yours to work on; the owner commits it. A local commit is tolerated and will be re-authored.
- Look at the other implementer's worktree, branch, or documents.
- Invent facts. If a required fact cannot be located in the SPEC or the documents, stop and ask in RESULTS.

## Reporting
RESULTS has fixed sections: what changed per module; deviations from the SPEC with reasons; acceptance results per test with the prove-it-bites line; surprises; questions; recommendations. BUILD is generated. Polished prose earns nothing; a disclosed gap outranks a claimed completion.

## Style
No emoji. No restating the request. Errors are owned plainly.
