#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ValidationIssue:
    level: str
    message: str


def _load_json(path: Path, issues: list[ValidationIssue]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issues.append(ValidationIssue("error", f"missing file: {path}"))
    except json.JSONDecodeError as error:
        issues.append(ValidationIssue("error", f"invalid JSON at {path}: {error}"))
    return None


def _check_file(path: Path, issues: list[ValidationIssue], label: str) -> None:
    if not path.is_file():
        issues.append(ValidationIssue("error", f"missing {label}: {path}"))


def _required_run_files(mode: str, status: str) -> list[str]:
    if status != "completed":
        return ["manifest.json"]
    if mode == "train":
        return [
            "manifest.json",
            "training_metrics.csv",
            "training_summary.json",
            "live_rollout.csv",
            "checkpoints/policy_last.pt",
            "checkpoints/policy_last.meta",
        ]
    if mode == "eval":
        return [
            "manifest.json",
            "evaluation_summary.json",
        ]
    return ["manifest.json"]


def _validate_runs(root: Path, strict: bool, issues: list[ValidationIssue]) -> dict[str, dict[str, Any]]:
    runs_dir = root / "runs"
    manifests: dict[str, dict[str, Any]] = {}
    if not runs_dir.exists():
        issues.append(ValidationIssue("error", f"missing runs directory: {runs_dir}"))
        return manifests

    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        manifest_path = run_dir / "manifest.json"
        manifest = _load_json(manifest_path, issues)
        if manifest is None:
            continue

        run_id = str(manifest.get("run_id", ""))
        mode = str(manifest.get("mode", ""))
        status = str(manifest.get("status", ""))
        schema_version = str(manifest.get("schema_version", ""))

        if run_id != run_dir.name:
            issues.append(
                ValidationIssue(
                    "error",
                    f"run_id mismatch: manifest={run_id} dir={run_dir.name} ({manifest_path})",
                )
            )
            continue

        for relative in _required_run_files(mode, status):
            _check_file(run_dir / relative, issues, f"{mode} artifact")

        if status == "completed":
            summary_name = "training_summary.json" if mode == "train" else "evaluation_summary.json"
            summary_path = run_dir / summary_name
            summary = _load_json(summary_path, issues)
            if summary is not None and str(summary.get("run_id", "")) != run_id:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"summary run_id mismatch in {summary_path}: expected {run_id}",
                    )
                )

            if strict and schema_version == "1.0":
                artifacts = manifest.get("artifacts")
                if not isinstance(artifacts, dict) or len(artifacts) == 0:
                    issues.append(ValidationIssue("error", f"manifest artifacts missing in {manifest_path}"))

        manifests[run_id] = manifest

    return manifests


def _validate_latest(root: Path, issues: list[ValidationIssue]) -> None:
    latest_dir = root / "latest"
    _check_file(latest_dir / "manifest.json", issues, "latest manifest")
    _check_file(latest_dir / "checkpoint.pt", issues, "latest checkpoint")
    _check_file(latest_dir / "checkpoint.meta", issues, "latest checkpoint meta")


def _validate_benchmarks(root: Path, issues: list[ValidationIssue]) -> None:
    benchmarks_dir = root / "benchmarks"
    _check_file(benchmarks_dir / "latest.json", issues, "benchmark latest.json")
    _check_file(benchmarks_dir / "latest.csv", issues, "benchmark latest.csv")
    latest_json = _load_json(benchmarks_dir / "latest.json", issues)
    if latest_json is not None:
        for key in ("benchmark_id", "benchmark_name", "train", "eval", "integrity", "functional"):
            if key not in latest_json:
                issues.append(ValidationIssue("error", f"benchmark latest.json missing key: {key}"))

    latest_csv = benchmarks_dir / "latest.csv"
    if latest_csv.exists():
        try:
            with latest_csv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                rows = list(reader)
                if len(rows) < 2:
                    issues.append(ValidationIssue("error", "benchmark latest.csv must contain header + 1 row"))
        except Exception as error:  # noqa: BLE001
            issues.append(ValidationIssue("error", f"unable to read benchmark CSV: {error}"))


def _validate_database(
    root: Path,
    manifests: dict[str, dict[str, Any]],
    strict: bool,
    issues: list[ValidationIssue],
) -> None:
    db_path = root / "experiments.sqlite"
    _check_file(db_path, issues, "SQLite database")
    if not db_path.exists():
        return

    connection = sqlite3.connect(str(db_path))
    try:
        cursor = connection.cursor()
        required_tables = ["runs", "episodes", "events", "benchmarks", "schema_migrations"]
        for table in required_tables:
            cursor.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?;",
                (table,),
            )
            if cursor.fetchone()[0] == 0:
                issues.append(ValidationIssue("error", f"missing SQLite table: {table}"))

        for run_id, manifest in manifests.items():
            cursor.execute(
                "SELECT status, COALESCE(summary_json, ''), COALESCE(config_json, '') FROM runs WHERE run_id = ?;",
                (run_id,),
            )
            row = cursor.fetchone()
            if row is None:
                issues.append(ValidationIssue("error", f"run not found in DB: {run_id}"))
                continue

            db_status, db_summary, _db_config = row
            manifest_status = str(manifest.get("status", ""))
            if db_status != manifest_status:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"status mismatch for {run_id}: manifest={manifest_status} db={db_status}",
                    )
                )

            if db_summary:
                try:
                    parsed = json.loads(db_summary)
                    if parsed.get("run_id") not in (None, run_id):
                        issues.append(ValidationIssue("error", f"db summary run_id mismatch for {run_id}"))
                except json.JSONDecodeError:
                    issues.append(ValidationIssue("error", f"db summary_json invalid JSON for {run_id}"))

            if strict and str(manifest.get("schema_version", "")) == "1.0":
                cursor.execute(
                    "SELECT COUNT(*) FROM run_artifacts WHERE run_id = ?;",
                    (run_id,),
                )
                artifact_count = cursor.fetchone()[0]
                if artifact_count <= 0:
                    issues.append(ValidationIssue("error", f"run_artifacts missing for run: {run_id}"))

                cursor.execute(
                    "SELECT COUNT(*) FROM run_configs WHERE run_id = ?;",
                    (run_id,),
                )
                config_count = cursor.fetchone()[0]
                if config_count <= 0:
                    issues.append(ValidationIssue("error", f"run_configs missing for run: {run_id}"))
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate artifact + SQLite integrity contracts.")
    parser.add_argument("--root", default="artifacts", help="Artifact root directory (default: artifacts)")
    parser.add_argument("--strict", action="store_true", help="Enable strict schema-level checks")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    issues: list[ValidationIssue] = []

    manifests = _validate_runs(root, args.strict, issues)
    _validate_latest(root, issues)
    _validate_benchmarks(root, issues)
    _validate_database(root, manifests, args.strict, issues)

    if issues:
        for issue in issues:
            prefix = "[ERROR]" if issue.level == "error" else "[WARN]"
            print(f"{prefix} {issue.message}")
        return 1

    print(f"[OK] artifact contract validated under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
