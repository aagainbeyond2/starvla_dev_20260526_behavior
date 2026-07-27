#!/usr/bin/env python3
"""Merge two node-local DeepSpeed checkpoints before an exact resume.

Run this once on rank-0's node after a failure and before relaunching either
rank. The operation is additive: it copies missing ZeRO/RNG shards in both
directions and never removes or overwrites an existing shard.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

LOCAL_MARKER = "_LOCAL_COMPLETE"
MERGED_MARKER = "_MERGED_COMPLETE"
METADATA = "trainer_state.json"


def run_checked(args: list[str], *, attempts: int = 3) -> subprocess.CompletedProcess[str]:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return subprocess.run(args, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(attempt * 2)
    assert last_error is not None
    raise RuntimeError(
        f"Command failed after {attempts} attempts: {args}\n"
        f"stdout:\n{last_error.stdout}\nstderr:\n{last_error.stderr}"
    ) from last_error


def run_ssh_python(peer: str, code: str, *args: str) -> subprocess.CompletedProcess[str]:
    remote_command = "python3 -c " + shlex.quote(code)
    if args:
        remote_command += " " + " ".join(shlex.quote(str(arg)) for arg in args)
    return run_checked(["ssh", peer, remote_command])


def local_steps(root: Path) -> set[int]:
    steps: set[int] = set()
    if not root.is_dir():
        return steps
    for path in root.glob("steps_*"):
        if not path.is_dir() or not (path / LOCAL_MARKER).is_file() or not (path / METADATA).is_file():
            continue
        try:
            steps.add(int(path.name.removeprefix("steps_")))
        except ValueError:
            continue
    return steps


REMOTE_LIST_CODE = r"""
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
steps = []
if root.is_dir():
    for path in root.glob('steps_*'):
        if path.is_dir() and (path / '_LOCAL_COMPLETE').is_file() and (path / 'trainer_state.json').is_file():
            try:
                steps.append(int(path.name.removeprefix('steps_')))
            except ValueError:
                pass
print(json.dumps(sorted(steps)))
"""


REMOTE_VALIDATE_CODE = r"""
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
metadata = json.loads((path / 'trainer_state.json').read_text())
world = int(metadata['world_size'])
optim = list(path.rglob('*optim_states.pt'))
rng = list(path.glob('random_states_*.pkl'))
models = list(path.rglob('*model_states.pt'))
latest = path / 'latest'
ok = len(optim) == world and len(rng) == world and len(models) >= 1 and latest.is_file()
ok = ok and all(item.stat().st_size > 0 for item in optim + rng + models + [latest])
print(json.dumps({
    'ok': ok,
    'world_size': world,
    'optimizer_shards': len(optim),
    'rng_states': len(rng),
    'model_states': len(models),
}))
raise SystemExit(0 if ok else 2)
"""


REMOTE_MARK_CODE = r"""
import json, os, pathlib, sys
path = pathlib.Path(sys.argv[1]) / '_MERGED_COMPLETE'
payload = json.loads(sys.argv[2])
tmp = path.with_name('.' + path.name + '.tmp-' + str(os.getpid()))
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
os.replace(tmp, path)
"""


def remote_steps(peer: str, root: Path) -> set[int]:
    result = run_ssh_python(peer, REMOTE_LIST_CODE, str(root))
    return {int(step) for step in json.loads(result.stdout)}


def read_metadata(path: Path) -> dict:
    with open(path / METADATA, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_local(path: Path) -> dict:
    metadata = read_metadata(path)
    world_size = int(metadata["world_size"])
    optimizer_shards = list(path.rglob("*optim_states.pt"))
    rng_states = list(path.glob("random_states_*.pkl"))
    model_states = list(path.rglob("*model_states.pt"))
    latest = path / "latest"
    files = optimizer_shards + rng_states + model_states + [latest]
    ok = (
        len(optimizer_shards) == world_size
        and len(rng_states) == world_size
        and len(model_states) >= 1
        and latest.is_file()
        and all(item.stat().st_size > 0 for item in files)
    )
    report = {
        "ok": ok,
        "world_size": world_size,
        "optimizer_shards": len(optimizer_shards),
        "rng_states": len(rng_states),
        "model_states": len(model_states),
    }
    if not ok:
        raise RuntimeError(f"Merged checkpoint validation failed: {report}")
    return report


def atomic_mark_local(path: Path, payload: dict) -> None:
    marker = path / MERGED_MARKER
    tmp = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, marker)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--peer", default="node96")
    parser.add_argument("--step", type=int)
    args = parser.parse_args()

    root = args.run_dir.resolve() / "full_state_checkpoints"
    local = local_steps(root)
    remote = remote_steps(args.peer, root)
    common = sorted(local & remote)
    if args.step is not None:
        if args.step not in common:
            raise RuntimeError(
                f"Requested step {args.step} is not locally complete on both nodes; common={common}"
            )
        step = args.step
    elif common:
        step = common[-1]
    else:
        raise RuntimeError(f"No common locally-complete full-state step; local={sorted(local)}, remote={sorted(remote)}")

    step_dir = root / f"steps_{step:09d}"
    remote_metadata = json.loads(
        run_checked(
            ["ssh", args.peer, "cat " + shlex.quote(str(step_dir / METADATA))]
        ).stdout
    )
    local_metadata = read_metadata(step_dir)
    if local_metadata != remote_metadata:
        raise RuntimeError(
            f"Metadata differs between nodes at step {step}: local={local_metadata}, remote={remote_metadata}"
        )

    run_checked(["rsync", "-a", "--ignore-existing", f"{args.peer}:{step_dir}/", f"{step_dir}/"])
    run_checked(["rsync", "-a", "--ignore-existing", f"{step_dir}/", f"{args.peer}:{step_dir}/"])

    local_report = validate_local(step_dir)
    remote_report = json.loads(
        run_ssh_python(args.peer, REMOTE_VALIDATE_CODE, str(step_dir)).stdout
    )
    if not remote_report.get("ok"):
        raise RuntimeError(f"Remote merged checkpoint validation failed: {remote_report}")

    marker_payload = {
        "completed_steps": step,
        "peer": args.peer,
        "local_validation": local_report,
        "remote_validation": remote_report,
    }
    atomic_mark_local(step_dir, marker_payload)
    run_ssh_python(
        args.peer, REMOTE_MARK_CODE, str(step_dir), json.dumps(marker_payload)
    )
    print(json.dumps({"merged_step": step, "path": str(step_dir), **marker_payload}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
