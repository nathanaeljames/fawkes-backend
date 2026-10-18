#!/usr/bin/env python3
"""round.py — round bookkeeping for the Fawkes development pipeline.

Runs inside an implementer worktree (or the owner's main checkout for `guard`).
State lives in .round/ (git-ignored). Timing measures the worktree, not the model:
start/end stamps plus a file-activity timeline, from which active time, elapsed
time and labeled pauses are derived. The harness transcript is the ground truth
for what happened inside a pause.

Commands
  open  ROUND --seat A|B|N        stamp base commit and start time
  watch [--interval 30]           poll the tree and record file activity (run in background)
  status                          show timing so far
  close ROUND --seat A|B|N [--transcript PATH] [--inline-diff] [--idle-minutes 5]
                                  write buildout/ROUND/BUILD-<seat>.md
  guard                           verify open rounds' spec tests are unchanged (CI)
  manifest ROUND --tests PATH...  (owner) record spec-test hashes for a round
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())
STATE = ROOT / ".round"
IGNORE_DIRS = {".git", ".round", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv", "node_modules", ".mypy_cache"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def state_path(round_id: str, seat: str) -> pathlib.Path:
    STATE.mkdir(exist_ok=True)
    return STATE / f"{round_id}-{seat}.json"


def load(round_id: str, seat: str) -> dict:
    p = state_path(round_id, seat)
    if not p.exists():
        sys.exit(f"no open round state at {p}; run `round.py open {round_id} --seat {seat}` first")
    return json.loads(p.read_text())


def save(round_id: str, seat: str, data: dict) -> None:
    state_path(round_id, seat).write_text(json.dumps(data, indent=2))


def tree_mtimes() -> dict[str, float]:
    out: dict[str, float] = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for f in filenames:
            p = pathlib.Path(dirpath) / f
            try:
                out[str(p.relative_to(ROOT))] = p.stat().st_mtime
            except OSError:
                pass
    return out


def cmd_open(a: argparse.Namespace) -> None:
    base = git("rev-parse", "HEAD").strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    data = {"round": a.round, "seat": a.seat, "base": base, "branch": branch, "start": now(), "events": [], "snapshot": tree_mtimes()}
    save(a.round, a.seat, data)
    print(f"opened {a.round} seat {a.seat} on {branch} at {base[:10]} ({data['start']})")


def current_open() -> tuple[str, str, dict]:
    files = sorted(STATE.glob("*.json")) if STATE.exists() else []
    if not files:
        sys.exit("no open round; run `round.py open` first")
    data = json.loads(files[-1].read_text())
    return data["round"], data["seat"], data


def cmd_watch(a: argparse.Namespace) -> None:
    round_id, seat, data = current_open()
    print(f"watching {round_id} seat {seat} every {a.interval}s; Ctrl-C to stop")
    try:
        while True:
            time.sleep(a.interval)
            data = load(round_id, seat)
            snap = data.get("snapshot", {})
            cur = tree_mtimes()
            changed = [p for p, m in cur.items() if snap.get(p) != m]
            if changed:
                data["events"].append({"t": now(), "files": len(changed)})
                data["snapshot"] = cur
                save(round_id, seat, data)
    except KeyboardInterrupt:
        print("stopped")


def timeline(data: dict, idle_minutes: float, end: str | None = None) -> dict:
    """Active time = sum of inter-event gaps shorter than the idle threshold; gaps longer are listed as pauses."""
    stamps = [data["start"]] + [e["t"] for e in data.get("events", [])]
    if end:
        stamps.append(end)
    ts = [dt.datetime.fromisoformat(s) for s in stamps]
    active = dt.timedelta()
    pauses = []
    for prev, cur in zip(ts, ts[1:]):
        gap = cur - prev
        if gap <= dt.timedelta(minutes=idle_minutes):
            active += gap
        else:
            pauses.append({"from": prev.isoformat(timespec="seconds"), "minutes": round(gap.total_seconds() / 60, 1)})
    elapsed = ts[-1] - ts[0] if len(ts) > 1 else dt.timedelta()
    return {"active_minutes": round(active.total_seconds() / 60, 1), "elapsed_minutes": round(elapsed.total_seconds() / 60, 1), "pauses": pauses, "events": len(data.get("events", []))}


def cmd_status(a: argparse.Namespace) -> None:
    round_id, seat, data = current_open()
    print(json.dumps({"round": round_id, "seat": seat, "base": data["base"][:10], **timeline(data, a.idle_minutes, now())}, indent=2))


def cmd_close(a: argparse.Namespace) -> None:
    data = load(a.round, a.seat)
    end = now()
    tl = timeline(data, a.idle_minutes, end)
    head = git("rev-parse", "HEAD").strip()
    # Diff against the base: committed work, the working tree, and untracked files (intent-to-add makes them visible).
    untracked = [u for u in git("ls-files", "--others", "--exclude-standard").splitlines() if u.strip()]
    if untracked:
        subprocess.run(["git", "add", "-N", "--", *untracked], cwd=ROOT, check=True)
    excl = ["--", ".", ":(exclude).round", ":(exclude)buildout"]
    diff = git("diff", data["base"], *excl)
    stat = git("diff", "--stat", data["base"], *excl)
    digest = hashlib.sha256(diff.encode()).hexdigest()
    out_dir = ROOT / "buildout" / a.round
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"BUILD-{a.seat}.md"
    lines = [
        f"# BUILD — {a.round} — seat {a.seat}",
        "",
        f"- Branch: `{data['branch']}`",
        f"- Base commit: `{data['base']}`",
        f"- Head at close: `{head}`" + ("" if head != data["base"] else " (uncommitted working tree)"),
        f"- Start (UTC): {data['start']}",
        f"- End (UTC): {end}",
        f"- Active: {tl['active_minutes']} min · Elapsed: {tl['elapsed_minutes']} min · Activity events: {tl['events']}",
        "- Pauses longer than the idle threshold: " + (", ".join(f"{p['from']} ({p['minutes']} min)" for p in tl["pauses"]) or "none"),
        f"- Harness transcript: {a.transcript or 'NOT RECORDED — grader applies the Step 0 penalty'}",
        f"- Diff SHA-256: `{digest}`",
        f"- Reproduce: `git diff {data['base'][:12]}..{data['branch']}`",
        "",
        "## Diff stat",
        "```",
        stat.rstrip(),
        "```",
    ]
    if a.inline_diff:
        lines += ["", "## Complete diff", "```diff", diff.rstrip(), "```"]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.relative_to(ROOT)}  active={tl['active_minutes']}m elapsed={tl['elapsed_minutes']}m pauses={len(tl['pauses'])}")


def manifest_path(round_id: str) -> pathlib.Path:
    return ROOT / "buildout" / round_id / "manifest.json"


def cmd_manifest(a: argparse.Namespace) -> None:
    tests = {}
    for t in a.tests:
        p = ROOT / t
        tests[t] = hashlib.sha256(p.read_bytes()).hexdigest()
    mp = manifest_path(a.round)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps({"round": a.round, "status": "open", "tests": tests, "recorded": now()}, indent=2))
    print(f"recorded {len(tests)} spec test file(s) in {mp.relative_to(ROOT)}")


def cmd_guard(a: argparse.Namespace) -> None:
    bad = []
    for mp in sorted((ROOT / "buildout").glob("*/manifest.json")):
        m = json.loads(mp.read_text())
        if m.get("status") != "open":
            continue
        for t, digest in m["tests"].items():
            p = ROOT / t
            actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
            if actual != digest:
                bad.append(f"{m['round']}: {t} changed or missing (spec tests are pipeline-owned)")
    if bad:
        print("\n".join(bad))
        sys.exit(1)
    print("spec-test guard: all open rounds' tests unchanged")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("open"); p.add_argument("round"); p.add_argument("--seat", required=True, choices=["A", "B", "N"]); p.set_defaults(fn=cmd_open)
    p = sub.add_parser("watch"); p.add_argument("--interval", type=int, default=30); p.set_defaults(fn=cmd_watch)
    p = sub.add_parser("status"); p.add_argument("--idle-minutes", type=float, default=5); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("close"); p.add_argument("round"); p.add_argument("--seat", required=True, choices=["A", "B", "N"]); p.add_argument("--transcript"); p.add_argument("--inline-diff", action="store_true"); p.add_argument("--idle-minutes", type=float, default=5); p.set_defaults(fn=cmd_close)
    p = sub.add_parser("manifest"); p.add_argument("round"); p.add_argument("--tests", nargs="+", required=True); p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("guard"); p.set_defaults(fn=cmd_guard)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
