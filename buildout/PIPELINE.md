# The Fawkes development pipeline

v2, 2026-10-20 (v1 2026-10-13). How Iteration 2 code gets written, judged, understood, and integrated. Derived from the Greek Tutor pipeline (22 graded rounds) with the changes that project's record argued for. This file owns the process; `CONTRACT.md` owns the implementer rules; `GRADER.md` owns grading; `ROUNDS.md` owns the round list; `docs/` owns what Fawkes is.

## Authority
| Question | Owner |
|---|---|
| What Fawkes must do | `docs/Fawkes_Architecture_Specification.md`, then `docs/Fawkes_Behavior_Contract.md` |
| In what order | `docs/Fawkes_Implementation_Plan.md`, cut into rounds by `buildout/ROUNDS.md` |
| How a round is run | this file |
| What an implementer may do | `buildout/CONTRACT.md` |
| How a round is graded | `buildout/GRADER.md` |
| What was decided and when | `docs/Fawkes_Session_Handoff.md` decisions log; `buildout/LEDGER.md` for round outcomes |
| Deliberate departures from the specification | `buildout/DIVERGENCE-LOG.md` (numbered DV-n, never renumbered) |
| Deferred minor items | `buildout/NIT-LOG.md` (numbered NT-n) |
| Security-relevant deferrals | a private note outside the repository (see Publication); the public log entry names only the resolving round |
When two documents disagree, the one named here wins and the other has a bug; say so rather than reconciling silently.

## Seats
| Seat | Held by | Does | Never |
|---|---|---|---|
| Owner | Nathanael | Product decisions, rulings, all git writes, integration, the viva, the owner-port step | Delegates version control |
| Planning (Fable, chat) | Claude | SPEC, tests, LEARN, living documents, review of RESULTS | Writes app code in a round it specified |
| Implementer A | Claude Code | Code against the SPEC in worktree A; RESULTS, BUILD | Git writes beyond its branch; pipeline-owned files; seeing B |
| Implementer B (Astra) | Codex | Same, worktree B | Same |
| Implementer N (optional) | Owner, blind | Same, worktree N; used on rounds the owner chooses | — |
| Grader | Fresh Fable chat with `GRADER.md`; Astra on close or contested rounds; a third family (GLM) on every fifth round as calibration | Claims-vs-diff audit, grades, XPATCH, viva questions, ledger | Grades a round it implemented |

## Day 0 (once, before Phase 0)
Day 0 is pipeline scaffolding and contains no Fawkes code. It is separate from Phase 0, which is the first three Fawkes rounds.
1. Commit `buildout/` (this file, CONTRACT, GRADER, ROUNDS, templates, empty logs and ledger), `tools/round.py`, `.github/workflows/ci.yml`, and the `pyproject.toml` additions (round markers, `xfail_strict`).
2. Add `.round/` to `.gitignore`.
3. Confirm CI is green on an empty suite.
4. Create the first worktrees when P0-1 opens, not before.

## A round, step by step
1. **Plan.** Planning seat writes `buildout/<ROUND>/SPEC.md` from the template. Questions whose answers reshape the round are asked before the SPEC lands; owner answers are written into the SPEC.
2. **Tests first.** Planning seat writes the round's tests under `tests/`, each marked `@pytest.mark.round("<ROUND>")` and `@pytest.mark.xfail(strict=True, reason="<ROUND> open")`. Owner commits them to `main`, then records them: `python tools/round.py manifest <ROUND> --tests tests/...`. CI's `spec-guard` job now refuses any diff that changes those files.
3. **Open.** Owner cuts the branches and worktrees from the same base commit:
   ```
   git worktree add ../fawkes-backend-A -b round/<ROUND>/A main
   git worktree add ../fawkes-backend-B -b round/<ROUND>/B main
   ```
   Decide dual or solo per `ROUNDS.md`; the split rule: a round is split when its tests have different oracles (deterministic CPU test, GPU measurement, human judgment) and merged when they share one.
4. **Implement.** Each implementer, in its worktree: `round.py open`, `round.py watch &`, work, `pytest --runxfail -m <ROUND>` and `pytest`, prove-it-bites, `round.py close --transcript <harness log>`, RESULTS. The transcript itself stays outside the repository (under `~/.claude/` or `../claude-scratch/fawkes-backend/`); BUILD records its path, size, and SHA-256 so the grader can verify the file the implementer names.
5. **Seal.** Owner commits each worktree's state on its branch and tags it: `git -C ../fawkes-backend-A add -A && git -C ../fawkes-backend-A commit -m "<ROUND> A"`; `git tag <ROUND>/A round/<ROUND>/A`; same for B. Both implementations are now permanent history.
6. **Blind.** Owner copies RESULTS and BUILD into `buildout/<ROUND>/` under neutral seat letters if they are not already neutral, keeps the mapping privately, and hands the grader the branch names or the inline diffs.
7. **Grade.** Grader runs Steps 0-8 of `GRADER.md` and writes `GRADE.md`. Owner challenges anything that smells wrong. Within the tie band or on dispute, Astra grades the same bundles and the owner reconciles.
8. **Port, if argued.** Winner applies `XPATCH.md` in its worktree; owner commits it on the winner's branch; `round.py close` again with the patch time added.
9. **Owner port.** Owner reads both branches with `LEARN.md` in hand and makes any changes of his own on the winner's branch; those changes run the same tests and face the same viva.
10. **Viva.** Owner answers the grader's questions on each Own function; gaps recorded in `GRADE.md`; the round cannot close with an open gap on an Own function.
11. **Learn.** Planning seat writes `LEARN.md` from the shipped diff: what was written, why, why that way, what was ported and not, and what each seat saw that the other did not.
12. **Verify and integrate.** Owner runs the judgment items in `VERIFY.md`, then:
    ```
    git checkout main
    git merge --no-ff round/<ROUND>/A -m "<ROUND>: ship A (+n ports from B)"
    # in the same commit or the next: remove the xfail markers on the round's tests; set manifest status to "closed"
    git worktree remove ../fawkes-backend-A && git worktree remove ../fawkes-backend-B
    ```
    The loser's branch and tag remain. CI must be green with the promoted tests passing for real. Retroactive grade adjustment if integration contradicts a claim; absorb every owner edit into the next SPEC's model of the code. Each deviation in the shipped RESULTS gets a DV entry with the owner's ruling; minor findings deferred by the grader or the owner get NT entries.

## Documents per round (`buildout/<ROUND>/`)
`SPEC.md`, `manifest.json`, `RESULTS-A.md`, `BUILD-A.md`, `RESULTS-B.md`, `BUILD-B.md`, optional `RESULTS-N.md`/`BUILD-N.md`, `GRADE.md`, optional `XPATCH.md`, `LEARN.md`, `VERIFY.md`. Nothing in `buildout/` is ever deleted.

## Publication
`buildout/` is public, including every archived round. Three rules keep that safe:
1. **No secrets, transcripts, or environment specifics.** The machine-wide guards refuse secrets, credentials, logs, and `.jsonl` at commit and push. Hostnames, LAN addresses, VPN ranges, device identifiers, and real names are not caught by any scanner; the owner reads every RESULTS, GRADE, and LEARN for them at the viva before the round merges.
2. **Security-relevant deferrals are never described in public.** A grader or owner finding that names an exploitable gap (an authorization check missing, a lock that can be bypassed, a scope key not enforced) goes to a private note outside the repository, not to `NIT-LOG.md` or `DIVERGENCE-LOG.md`. The public log entry for it records only an identifier and the round that resolves it ("NT-7: resolved in P5-2"); the description is added to the public entry after that round merges.
3. **Nothing is deleted, but anything can be redacted.** If a published document is later found to violate rule 1 or 2, the owner rewrites the file and purges history by the recorded runbook; the ledger notes the redaction.

## Timing
`round.py` times the worktree, not the model: start and end stamps plus a file-activity timeline. Active time is the sum of gaps under the idle threshold (default 5 minutes); longer gaps are listed as pauses with their start and length, so an absent owner, exhausted credits, or a question waiting for an answer appear as labeled pauses rather than inflated work. The harness transcript is the ground truth for what happened inside a pause; the grader consults it when active time looks implausibly low.

## Decorrelation, honestly
Two vendors' implementers decorrelate implementation errors. They do nothing for SPEC errors (the planning seat's defects were a real share of the Greek defects) — the behavior contract, "ask before the SPEC lands", and Astra's document reviews are the mitigation. The grader is Claude-family by default and may share style preferences with implementer A; tests as first-order judge, blinding, the tie band, Astra on close rounds, and a third-family calibration grader every fifth round are the mitigations. The owner's observation outranks every seat.

## STOP rules
Stop on a referenced-but-absent artifact. Stop the whole turn on a wrong document. Stop if the live version of a file cannot be read before editing it. Stop when a required fact cannot be located; never invent it.
