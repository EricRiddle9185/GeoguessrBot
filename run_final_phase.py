import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_DIR = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def run_phase(iteration_dir, run_dir, action):
    iteration_dir, run_dir = iteration_dir.resolve(), run_dir.resolve()
    if not run_dir.is_relative_to(iteration_dir / "final_execution"):
        raise ValueError("The run log directory must be inside this iteration's final_execution folder.")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = run_dir / f"{action}_{timestamp}.log"
    command = [sys.executable, "-B", "-u", str(REPO_DIR / "growth_training.py"), "--iteration-dir", str(iteration_dir), "--action", action]
    event = {"time_utc": now(), "event": "phase_started", "action": action, "command": command, "log": str(log_path)}
    with (run_dir / "events.jsonl").open("a", encoding="ascii") as stream:
        stream.write(json.dumps(event) + "\n")
    manifest["status"] = action + "_running"
    manifest.setdefault("phases", []).append(event)
    write_json(manifest_path, manifest)
    source = run_dir / "logging_source"
    source.mkdir(exist_ok=True)
    content = Path(__file__).read_bytes()
    (source / "run_final_phase.py").write_bytes(content)
    write_json(source / "source_hash.json", {"run_final_phase.py": hashlib.sha256(content).hexdigest()})
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        log.write(now() + " COMMAND " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=REPO_DIR, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            entry = now() + " " + line.rstrip("\r\n") + "\n"
            log.write(entry)
            log.flush()
            sys.stdout.write(entry)
            sys.stdout.flush()
        exit_code = process.wait()
        log.write(now() + f" EXIT_CODE {exit_code}\n")
    event = {"time_utc": now(), "event": "phase_finished", "action": action, "exit_code": exit_code, "log": str(log_path)}
    with (run_dir / "events.jsonl").open("a", encoding="ascii") as stream:
        stream.write(json.dumps(event) + "\n")
    manifest = json.loads(manifest_path.read_text())
    manifest["phases"].append(event)
    manifest["status"] = action + ("_completed" if exit_code == 0 else "_failed")
    write_json(manifest_path, manifest)
    print("Phase log:", log_path, flush=True)
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log final training and evaluation.")
    parser.add_argument("--iteration-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--action", choices=("final-candidate", "evaluate-final"), required=True)
    args = parser.parse_args()
    raise SystemExit(run_phase(args.iteration_dir, args.run_dir, args.action))
