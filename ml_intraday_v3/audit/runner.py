"""
Audit runner for V3 pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from ml_intraday_v3.core.instrument import InstrumentSpec, validate_risk_config_no_instrument_economics
from ml_intraday_v3.run_manifest import hash_content

from .schema import write_audit_schema
from .checks_alignment import (
    check_events_on_grid,
    check_features_index,
    check_weights_event_ids,
)
from .checks_leakage import (
    check_train_test_disjoint,
    check_purge_integrity,
    check_embargo_integrity,
)
from .checks_accounting import check_cost_mode, check_pnl_identity
from .checks_risk import (
    check_daily_loss_limit,
    check_trailing_drawdown,
    check_forced_flatten,
)
from .checks_experiments import (
    check_experiment_leaderboards,
    check_experiment_diagnostics,
)


def _load_yaml_with_hash(path: Path) -> tuple[dict, str | None]:
    if not path.exists():
        return {}, None
    with open(path, "r") as f:
        content = f.read()
    parsed = yaml.safe_load(content) or {}
    return parsed, hash_content(content)


def _load_bar_sizes(run_dir: Path) -> list[str]:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        return manifest.get("bar_sizes", ["1m", "5m"])
    return [
        d.name.replace("bar_size=", "")
        for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("bar_size=")
    ]


def _audit_schema_definitions() -> dict:
    return {
        "provenance": [
            "config_sources",
        ],
        "alignment": [
            "events_on_grid",
            "features_index_match",
            "weights_event_id_match",
        ],
        "leakage": [
            "train_test_disjoint",
            "purge_integrity",
            "embargo_integrity",
        ],
        "accounting": [
            "cost_mode_consistency",
            "pnl_identity",
        ],
        "risk": [
            "daily_loss_limit",
            "trailing_drawdown",
            "forced_flatten",
        ],
        "experiments": [
            "leaderboard_links",
            "diagnostics_inputs",
        ],
    }


def _find_config_path_in_run_dir(run_dir: Path, name: str) -> Path | None:
    candidates = [
        run_dir / "configs" / f"{name}.yaml",
        run_dir / "configs" / f"{name}.yml",
        run_dir / f"{name}.yaml",
        run_dir / f"{name}.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _select_config(
    manifest: dict | None, run_dir: Path, name: str, default_path: Path
) -> tuple[dict, dict]:
    provenance = {"name": name}
    config = {}

    if manifest:
        for entry in manifest.get("configs", []):
            if entry.get("name") != name:
                continue
            if "content" in entry and entry["content"] is not None:
                config = entry["content"]
                content_hash = entry.get("content_hash")
                if content_hash is None:
                    content_hash = hash_content(entry["content"])
                provenance.update(
                    {
                        "source": "manifest",
                        "source_detail": "content",
                        "path": entry.get("path"),
                        "content_hash": content_hash,
                    }
                )
                return config, provenance
            if entry.get("path"):
                path = Path(entry["path"])
                if path.exists():
                    config, content_hash = _load_yaml_with_hash(path)
                    provenance.update(
                        {
                            "source": "manifest",
                            "source_detail": "path",
                            "path": str(path),
                            "content_hash": entry.get("content_hash")
                            or content_hash,
                        }
                    )
                    return config, provenance

    run_dir_path = _find_config_path_in_run_dir(run_dir, name)
    if run_dir_path:
        config, content_hash = _load_yaml_with_hash(run_dir_path)
        provenance.update(
            {
                "source": "run_dir",
                "path": str(run_dir_path),
                "content_hash": content_hash,
            }
        )
        return config, provenance

    config, content_hash = _load_yaml_with_hash(default_path)
    provenance.update(
        {
            "source": "default",
            "path": str(default_path),
            "content_hash": content_hash,
            "fallback_used": True,
        }
    )
    return config, provenance


def run_audit(run_dir: Path | str, strict: bool = False) -> dict:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    bar_sizes = _load_bar_sizes(run_dir)
    exp_root = run_dir / "experiments"

    manifest_path = run_dir / "run_manifest.json"
    manifest = None
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

    risk_cfg, risk_prov = _select_config(
        manifest, run_dir, "risk", Path("ml_intraday_v3/configs/risk.yaml")
    )
    validate_risk_config_no_instrument_economics(risk_cfg)
    backtest_cfg, backtest_prov = _select_config(
        manifest,
        run_dir,
        "backtest",
        Path("ml_intraday_v3/configs/backtest.yaml"),
    )
    execution_spec, execution_spec_prov = _select_config(
        manifest,
        run_dir,
        "execution_spec",
        Path("ml_intraday_v3/configs/execution_spec.yaml"),
    )
    instrument_spec = InstrumentSpec.from_execution_spec(execution_spec)
    instrument_prov = {
        "name": "instrument",
        "source": execution_spec_prov.get("source"),
        "path": execution_spec_prov.get("path"),
        "content_hash": hash_content(execution_spec.get("instrument", {})),
        "derived_from": "execution_spec",
    }

    per_bar_reports = {}
    for bar_size in bar_sizes:
        bar_dir = run_dir / f"bar_size={bar_size}"
        bars_path = bar_dir / "bars.parquet"
        events_path = bar_dir / "events.parquet"
        features_path = bar_dir / "features.parquet"
        weights_path = bar_dir / "weights.parquet"
        cv_path = bar_dir / "cv_splits.json"

        bars_df = pd.read_parquet(bars_path) if bars_path.exists() else pd.DataFrame()
        events_df = (
            pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()
        )
        features_df = (
            pd.read_parquet(features_path)
            if features_path.exists()
            else pd.DataFrame()
        )
        weights_df = (
            pd.read_parquet(weights_path)
            if weights_path.exists()
            else pd.DataFrame()
        )

        bars_index = bars_df.index if not bars_df.empty else pd.Index([])

        cv_data = {}
        if cv_path.exists():
            with open(cv_path, "r") as f:
                cv_data = json.load(f)

        alignment_checks = {
            "events_on_grid": check_events_on_grid(bars_index, events_df)
            if not bars_df.empty and not events_df.empty
            else {"status": "SKIP", "reason": "missing_bars_or_events"},
            "features_index_match": check_features_index(bars_index, features_df)
            if not bars_df.empty and not features_df.empty
            else {"status": "SKIP", "reason": "missing_bars_or_features"},
            "weights_event_id_match": check_weights_event_ids(
                events_df, weights_df
            )
            if not events_df.empty and not weights_df.empty
            else {"status": "SKIP", "reason": "missing_events_or_weights"},
        }

        leakage_checks = {
            "train_test_disjoint": check_train_test_disjoint(cv_data)
            if cv_data
            else {"status": "SKIP", "reason": "missing_cv_splits"},
            "purge_integrity": check_purge_integrity(events_df, cv_data)
            if cv_data and not events_df.empty
            else {"status": "SKIP", "reason": "missing_events_or_cv"},
            "embargo_integrity": check_embargo_integrity(
                events_df, bars_index, cv_data
            )
            if cv_data and not events_df.empty and not bars_df.empty
            else {"status": "SKIP", "reason": "missing_inputs"},
        }

        accounting_checks = {
            "cost_mode_consistency": check_cost_mode(bar_dir, events_df)
            if not events_df.empty
            else {"status": "SKIP", "reason": "missing_events"},
            "pnl_identity": check_pnl_identity(
                bar_dir, instrument_spec=instrument_spec
            ),
        }

        risk_checks = {
            "daily_loss_limit": check_daily_loss_limit(bar_dir, risk_cfg),
            "trailing_drawdown": check_trailing_drawdown(bar_dir, risk_cfg),
            "forced_flatten": check_forced_flatten(bar_dir, backtest_cfg),
        }

        experiment_checks = {
            "leaderboard_links": check_experiment_leaderboards(exp_root),
            "diagnostics_inputs": check_experiment_diagnostics(exp_root),
        }

        audit_inputs = {
            "instrument_config_source": instrument_prov.get("source"),
            "instrument_config_path": instrument_prov.get("path"),
            "instrument_config_hash": instrument_prov.get("content_hash"),
            "risk_config_source": risk_prov.get("source"),
            "risk_config_path": risk_prov.get("path"),
            "risk_config_hash": risk_prov.get("content_hash"),
            "backtest_config_source": backtest_prov.get("source"),
            "backtest_config_path": backtest_prov.get("path"),
            "backtest_config_hash": backtest_prov.get("content_hash"),
            "execution_spec_source": execution_spec_prov.get("source"),
            "execution_spec_path": execution_spec_prov.get("path"),
            "execution_spec_hash": execution_spec_prov.get("content_hash"),
        }

        default_sources = [
            name
            for name, prov in [
                ("instrument_config", instrument_prov),
                ("risk_config", risk_prov),
                ("backtest_config", backtest_prov),
                ("execution_spec", execution_spec_prov),
            ]
            if prov.get("source") == "default"
        ]
        provenance_status = "PASS"
        provenance_message = "All configs resolved from manifest or run_dir"
        if default_sources:
            provenance_status = "WARN"
            provenance_message = (
                "Default configs used for: "
                + ", ".join(default_sources)
            )
        provenance_checks = {
            "config_sources": {
                "status": provenance_status,
                "defaults_used": default_sources,
                "message": provenance_message,
                "strict": strict,
            }
        }

        checks = {
            "provenance": provenance_checks,
            "alignment": alignment_checks,
            "leakage": leakage_checks,
            "accounting": accounting_checks,
            "risk": risk_checks,
            "experiments": experiment_checks,
        }

        statuses = [
            c["status"]
            for group in checks.values()
            for c in group.values()
            if "status" in c
        ]
        overall_status = "FAIL" if "FAIL" in statuses else "PASS"

        report = {
            "bar_size": bar_size,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "strict": strict,
            "overall_status": overall_status,
            "audit_inputs": audit_inputs,
            "provenance": {
                "risk_config": risk_prov,
                "backtest_config": backtest_prov,
                "execution_spec": execution_spec_prov,
                "instrument": instrument_prov,
            },
            "checks": checks,
        }

        report_path = bar_dir / "audit_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        schema_path = bar_dir / "audit_schema.json"
        schema_hash = write_audit_schema(
            output_path=schema_path,
            checks=_audit_schema_definitions(),
            code_version="1.0.0",
        )

        per_bar_reports[bar_size] = {
            "audit_report_path": str(report_path.relative_to(run_dir)),
            "audit_schema_path": str(schema_path.relative_to(run_dir)),
            "audit_schema_hash": schema_hash,
            "overall_status": overall_status,
        }

        if strict and overall_status == "FAIL":
            raise ValueError(f"Audit failed for bar_size={bar_size}")

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if "per_bar_artifacts" not in manifest:
            if "per_bar_size_artifacts" in manifest:
                manifest["per_bar_artifacts"] = manifest.get(
                    "per_bar_size_artifacts", {}
                )
            else:
                manifest["per_bar_artifacts"] = {}

        for bar_size, artifacts in per_bar_reports.items():
            if bar_size not in manifest["per_bar_artifacts"]:
                manifest["per_bar_artifacts"][bar_size] = {}
            manifest["per_bar_artifacts"][bar_size].update(artifacts)

        manifest["per_bar_size_artifacts"] = manifest.get(
            "per_bar_artifacts", {}
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    return {
        "bar_sizes": bar_sizes,
        "per_bar_reports": per_bar_reports,
    }
