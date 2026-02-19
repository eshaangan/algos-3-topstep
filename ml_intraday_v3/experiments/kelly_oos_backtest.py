"""
Kelly Criterion OOS Backtest

Loads top N experiment results (by val_auc / median_test_auc), then for each
model simulates Kelly-sized position outcomes on OOS metrics.

Supports two strategies:
  - ml (default): loads ML experiment JSON results from results-dir
  - rule_based: loads trade-level PnL from rule_based OOS results JSON

Usage:
    # ML experiments (original behavior)
    python kelly_oos_backtest.py --results-dir experiments/results/batch1 --top-n 30

    # Rule-based system with Topstep combine Monte Carlo
    python kelly_oos_backtest.py \
        --strategy rule_based \
        --trades-file ml_intraday_v3/diagnostics/rule_based_oos_results.json \
        --output ml_intraday_v3/experiments/results/kelly_rulebased.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.risk.kelly_sizing import kelly_fraction, fractional_kelly_contracts


def load_result_files(results_dir: Path) -> list[dict]:
    """Recursively load all JSON result files from results_dir."""
    json_files = list(results_dir.rglob("*.json"))
    if not json_files:
        print(f"ERROR: No JSON result files found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    records = []
    for path in json_files:
        try:
            with open(path) as f:
                data = json.load(f)
            # Accept both single-result dicts and lists of results
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: Skipping {path}: {exc}", file=sys.stderr)

    if not records:
        print(f"ERROR: Loaded 0 valid records from {results_dir}", file=sys.stderr)
        sys.exit(1)

    return records


def extract_metric(record: dict, *keys, default=0.0):
    """Try multiple key names and return the first found value."""
    for key in keys:
        val = record.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return default


def get_labeling_params(record: dict) -> dict:
    """Extract labeling barrier params from a result record."""
    config = record.get('config', record)
    lp = config.get('labeling_params', {})
    return {
        'pt_mult': float(lp.get('pt_mult', 2.0)),
        'sl_mult': float(lp.get('sl_mult', 1.5)),
        'time_mult': float(lp.get('time_mult', 40)),
    }


def compute_kelly_metrics(
    record: dict,
    kelly_fractions: list[float],
    account_balance: float,
    contract_value: float,
    atr_usd: float = 100.0,
    point_value: float = 5.0,
    atr_points: float = 4.0,
) -> dict:
    """
    Compute Kelly-based position sizing metrics for a single result record.

    Parameters
    ----------
    record : dict
        Experiment result record.
    kelly_fractions : list[float]
        Fractional Kelly multipliers to evaluate.
    account_balance : float
        Account size in USD.
    contract_value : float
        Notional value per contract in USD.
    atr_usd : float
        Proxy ATR in USD per contract (default 100 for MES).
    point_value : float
        USD per point per contract (MES = 5).
    atr_points : float
        Approximate ATR in points.
    """
    lp = get_labeling_params(record)
    pt_mult = lp['pt_mult']
    sl_mult = lp['sl_mult']

    win_prob = extract_metric(
        record,
        'median_test_auc', 'val_auc', 'test_auc', 'mean_test_auc',
        default=0.5,
    )
    # Clamp win_prob to a valid range for Kelly computation
    win_prob = max(0.01, min(0.99, win_prob))

    payoff_ratio = pt_mult / sl_mult if sl_mult > 0 else 1.333

    # USD PnL per unit ATR per contract
    pt_usd = pt_mult * point_value * atr_points
    sl_usd = sl_mult * point_value * atr_points

    trades_per_day = extract_metric(
        record,
        'median_est_trades_per_day', 'est_trades_per_day', 'trades_per_day',
        default=3.0,
    )
    trades_per_day = max(0.1, trades_per_day)

    full_kelly = kelly_fraction(win_prob, payoff_ratio)

    expected_value_per_trade = win_prob * pt_usd - (1.0 - win_prob) * sl_usd
    single_trade_std = math.sqrt(
        win_prob * (pt_usd - expected_value_per_trade) ** 2
        + (1.0 - win_prob) * (-sl_usd - expected_value_per_trade) ** 2
    )

    risk_per_contract = sl_mult * atr_usd

    kelly_rows = []
    for kf in kelly_fractions:
        sizing = fractional_kelly_contracts(
            bankroll_usd=account_balance,
            risk_per_contract_usd=risk_per_contract,
            win_probability=win_prob,
            payoff_ratio=payoff_ratio,
            fraction=kf,
        )
        contracts = sizing['contracts']

        est_daily_pnl = contracts * trades_per_day * expected_value_per_trade
        est_monthly_pnl = est_daily_pnl * 21

        daily_std = contracts * math.sqrt(trades_per_day) * single_trade_std if trades_per_day > 0 else 0.0
        est_sharpe = (est_daily_pnl / daily_std) * math.sqrt(252) if daily_std > 0 else 0.0

        daily_loss_estimate = contracts * sl_usd * trades_per_day * (1.0 - win_prob)
        max_drawdown_estimate = contracts * sl_usd * 5  # approx 5-trade losing streak

        kelly_rows.append({
            'kelly_fraction': kf,
            'contracts': contracts,
            'raw_kelly': sizing['raw_kelly'],
            'fractional_kelly': sizing['fractional_kelly'],
            'est_daily_pnl': round(est_daily_pnl, 2),
            'est_monthly_pnl': round(est_monthly_pnl, 2),
            'est_annual_sharpe': round(est_sharpe, 3),
            'daily_loss_estimate': round(daily_loss_estimate, 2),
            'max_drawdown_estimate': round(max_drawdown_estimate, 2),
            'topstep_daily_limit_ok': daily_loss_estimate < 1000.0,
            'topstep_drawdown_ok': max_drawdown_estimate < 2000.0,
        })

    return {
        'exp_id': record.get('exp_id', 'unknown'),
        'win_prob': round(win_prob, 4),
        'payoff_ratio': round(payoff_ratio, 4),
        'pt_mult': pt_mult,
        'sl_mult': sl_mult,
        'trades_per_day': round(trades_per_day, 2),
        'full_kelly': round(full_kelly, 4),
        'expected_value_per_trade': round(expected_value_per_trade, 2),
        'median_pr_auc': extract_metric(record, 'median_pr_auc', 'pr_auc', default=0.0),
        'kelly_scenarios': kelly_rows,
    }


def build_summary_table(kelly_results: list[dict], kelly_fractions: list[float]) -> pd.DataFrame:
    """Flatten Kelly results into a DataFrame for ranking."""
    rows = []
    for kr in kelly_results:
        for scenario in kr['kelly_scenarios']:
            rows.append({
                'exp_id': kr['exp_id'],
                'win_prob': kr['win_prob'],
                'payoff_ratio': kr['payoff_ratio'],
                'pt_mult': kr['pt_mult'],
                'sl_mult': kr['sl_mult'],
                'trades_per_day': kr['trades_per_day'],
                'full_kelly': kr['full_kelly'],
                'median_pr_auc': kr['median_pr_auc'],
                **scenario,
            })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, top_n: int = 10) -> None:
    """Print ranked summary of top combinations by estimated monthly PnL."""
    print("\n" + "=" * 80)
    print("KELLY OOS BACKTEST SUMMARY — Top combinations by est_monthly_pnl")
    print("=" * 80)

    ranked = df.sort_values('est_monthly_pnl', ascending=False).head(top_n)
    cols = [
        'exp_id', 'kelly_fraction', 'contracts', 'win_prob',
        'payoff_ratio', 'trades_per_day', 'est_monthly_pnl',
        'est_annual_sharpe', 'topstep_daily_limit_ok', 'topstep_drawdown_ok',
    ]
    cols = [c for c in cols if c in df.columns]
    print(ranked[cols].to_string(index=False))

    passing = df[df['topstep_daily_limit_ok'] & df['topstep_drawdown_ok']]
    print(f"\nTopstep-safe combos: {len(passing)} of {len(df)} total")
    if not passing.empty:
        best = passing.sort_values('est_monthly_pnl', ascending=False).iloc[0]
        print(f"\nBest Topstep-safe combo:")
        print(f"  exp_id          : {best['exp_id']}")
        print(f"  kelly_fraction  : {best['kelly_fraction']}")
        print(f"  contracts       : {best['contracts']}")
        print(f"  win_prob        : {best['win_prob']:.4f}")
        print(f"  est_monthly_pnl : ${best['est_monthly_pnl']:,.2f}")
        print(f"  est_annual_sharpe: {best['est_annual_sharpe']:.3f}")
    print("=" * 80)


def load_rule_based_trades(trades_file: str) -> tuple[list, dict]:
    """
    Load trade-level PnL from validate_rule_based_oos.py output JSON.
    Returns (trade_pnl_list, best_params_dict).
    """
    with open(trades_file) as f:
        data = json.load(f)

    # Try to extract trade-level PnL from top result
    all_results = data.get("all_results", data.get("top5", []))
    if not all_results:
        raise ValueError(f"No results found in {trades_file}")

    best = all_results[0]
    params = best.get("params", {})

    # The OOS results JSON has summary stats but not individual trades
    # We reconstruct approximate trade PnL from summary stats
    backtest = best.get("backtest", {})
    n_trades = backtest.get("num_trades", 0)
    win_rate = backtest.get("win_rate", 0.5)
    avg_trade_pnl = backtest.get("avg_trade_pnl", 0.0)
    total_pnl = backtest.get("total_pnl", 0.0)

    if n_trades == 0:
        raise ValueError("No trades found in best result")

    # Reconstruct approximate trade distribution from summary stats
    # avg_win = total_win / n_wins, avg_loss = total_loss / n_losses
    # We use: total_pnl = n_wins*avg_win + n_losses*avg_loss
    # with the constraint payoff_ratio = avg_win / avg_loss
    pt_mult = params.get("pt_atr_mult", 2.0)
    sl_mult = params.get("sl_atr_mult", 1.5)
    payoff_ratio = pt_mult / sl_mult

    n_wins = int(round(n_trades * win_rate))
    n_losses = n_trades - n_wins

    # Derive avg_win and avg_loss from actual payoff ratio and total_pnl
    # total_pnl = n_wins * avg_win - n_losses * avg_loss
    # avg_win = payoff_ratio * avg_loss
    # Solving: avg_loss = total_pnl / (n_wins * payoff_ratio - n_losses)
    denom = n_wins * payoff_ratio - n_losses
    if abs(denom) < 1e-6 or n_losses == 0:
        avg_loss = abs(avg_trade_pnl) * 2
    else:
        avg_loss = abs(total_pnl / denom)

    avg_win = payoff_ratio * avg_loss

    # Build synthetic trade list with small noise for Monte Carlo variety
    rng = np.random.default_rng(42)
    trade_pnls = []
    for _ in range(n_wins):
        trade_pnls.append(float(avg_win * rng.lognormal(0, 0.3)))
    for _ in range(n_losses):
        trade_pnls.append(float(-avg_loss * rng.lognormal(0, 0.2)))

    return trade_pnls, params


def run_rule_based_kelly(
    trades_file: str,
    kelly_fractions: list,
    account_balance: float,
    n_mc_paths: int = 10_000,
    mc_seed: int = 42,
) -> dict:
    """
    Run Kelly sizing analysis for rule-based strategy with Topstep combine simulation.

    Returns dict with per-fraction metrics including P(pass_in_20_days).
    """
    # Import the combine simulator from ml_intraday_v3/diagnostics/
    diag_dir = project_root / "ml_intraday_v3" / "diagnostics"
    if str(diag_dir) not in sys.path:
        sys.path.insert(0, str(diag_dir))
    from topstep_combine_simulator import TopstepCombineSimulator  # noqa: F401

    sim = TopstepCombineSimulator(
        account_size=account_balance,
        profit_target=3_000,
        max_trailing_drawdown=2_000,
        max_daily_loss=1_000,
        consistency_pct=0.30,
        min_trading_days=5,
    )

    trade_pnls, best_params = load_rule_based_trades(trades_file)

    n_trades = len(trade_pnls)
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.5
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(abs(np.mean(losses))) if losses else 1.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

    full_kelly = kelly_fraction(win_rate, payoff_ratio)

    scenarios = []
    for kf in kelly_fractions:
        # Scale trade PnL by kelly fraction relative to full kelly
        # kf=0.25 means use 25% of what full kelly recommends
        scale = kf / max(full_kelly, 0.01)
        scaled_trades = [p * scale for p in trade_pnls]

        mc = sim.monte_carlo(
            trade_pnl_list=scaled_trades,
            n_paths=n_mc_paths,
            trades_per_day_range=(2, 8),
            max_days=40,
            seed=mc_seed,
        )

        # Estimate daily metrics
        est_daily_pnl = np.mean(scaled_trades) * 4  # ~4 trades/day
        est_monthly_pnl = est_daily_pnl * 21

        scenarios.append({
            "kelly_fraction": kf,
            "scale_factor": round(scale, 3),
            "p_pass_in_20_days": mc["p_pass"],
            "median_days_to_pass": mc["median_days"],
            "p_fail_daily_loss": mc["p_fail_daily"],
            "p_fail_trailing": mc["p_fail_trailing"],
            "p95_max_drawdown": mc["p95_max_drawdown"],
            "est_daily_pnl": round(float(est_daily_pnl), 2),
            "est_monthly_pnl": round(float(est_monthly_pnl), 2),
            "topstep_daily_ok": bool(mc["p_fail_daily"] < 0.10),
            "topstep_drawdown_ok": bool(mc.get("p95_max_drawdown", 9999) < 1_800),
        })

    return {
        "strategy": "rule_based",
        "trades_file": str(trades_file),
        "best_primary_params": best_params,
        "n_trades_sample": n_trades,
        "win_rate": round(win_rate, 4),
        "payoff_ratio": round(payoff_ratio, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "full_kelly": round(full_kelly, 4),
        "n_mc_paths": n_mc_paths,
        "kelly_scenarios": scenarios,
    }


def print_rule_based_summary(rb_result: dict) -> None:
    """Print rule-based Kelly / combine summary."""
    print("\n" + "=" * 80)
    print("KELLY OOS BACKTEST — RULE-BASED STRATEGY")
    print("=" * 80)
    print(f"Win rate:      {rb_result['win_rate']:.1%}")
    print(f"Payoff ratio:  {rb_result['payoff_ratio']:.2f}")
    print(f"Full Kelly:    {rb_result['full_kelly']:.3f}")
    print(f"Avg win:       ${rb_result['avg_win']:.2f}")
    print(f"Avg loss:      ${rb_result['avg_loss']:.2f}")
    print()
    print(f"{'Kelly':>8} {'P(pass 20d)':>12} {'Med Days':>9} {'P95 DD':>8} "
          f"{'Est Monthly':>12} {'Daily OK':>9} {'DD OK':>7}")
    print("-" * 80)
    for s in rb_result["kelly_scenarios"]:
        print(
            f"{s['kelly_fraction']:>8.2f} "
            f"{s['p_pass_in_20_days']:>12.1%} "
            f"{str(s['median_days_to_pass']):>9} "
            f"${s['p95_max_drawdown']:>7.0f} "
            f"${s['est_monthly_pnl']:>11.0f} "
            f"{'YES' if s['topstep_daily_ok'] else 'NO':>9} "
            f"{'YES' if s['topstep_drawdown_ok'] else 'NO':>7}"
        )
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kelly Criterion OOS Backtest for ML experiment results"
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='ml',
        choices=['ml', 'rule_based'],
        help="Strategy to analyze: 'ml' for ML experiments (default), 'rule_based' for rule-based system",
    )
    parser.add_argument(
        '--trades-file',
        type=str,
        default=None,
        help="[rule_based only] Path to rule_based_oos_results.json from validate_rule_based_oos.py",
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        help="[ml only] Directory containing JSON result files (searched recursively)",
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=30,
        help="Number of top models to evaluate (ranked by median_test_auc)",
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help="Path to write JSON output (optional)",
    )
    parser.add_argument(
        '--kelly-fractions',
        type=float,
        nargs='+',
        default=[0.10, 0.15, 0.25],
        help="Fractional Kelly multipliers to evaluate",
    )
    parser.add_argument(
        '--account-balance',
        type=float,
        default=50000.0,
        help="Account size in USD (default: 50000 for Topstep 50k)",
    )
    parser.add_argument(
        '--contract-value',
        type=float,
        default=1250.0,
        help="Contract notional value in USD (default: 1250 for MES = 5 x 250)",
    )
    parser.add_argument(
        '--mc-paths',
        type=int,
        default=10_000,
        help="[rule_based only] Monte Carlo paths for combine simulation",
    )
    parser.add_argument(
        '--batch',
        type=str,
        default=None,
        help="Batch filter (e.g. 'batch1', 'all'); used for display only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ----------------------------------------------------------------
    # Rule-based path
    # ----------------------------------------------------------------
    if args.strategy == "rule_based":
        if not args.trades_file:
            print(
                "ERROR: --trades-file is required for --strategy rule_based\n"
                "  Run validate_rule_based_oos.py first to generate it.",
                file=sys.stderr,
            )
            sys.exit(1)

        trades_file = args.trades_file
        if not Path(trades_file).exists():
            print(f"ERROR: trades-file not found: {trades_file}", file=sys.stderr)
            sys.exit(1)

        print(f"Running rule-based Kelly analysis from: {trades_file}")
        rb_result = run_rule_based_kelly(
            trades_file=trades_file,
            kelly_fractions=args.kelly_fractions,
            account_balance=args.account_balance,
            n_mc_paths=args.mc_paths,
        )
        print_rule_based_summary(rb_result)

        output_path = args.output or Path("ml_intraday_v3/experiments/results/kelly_rulebased.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(rb_result, f, indent=2, default=str)
        print(f"\nOutput written to: {output_path}")
        return

    # ----------------------------------------------------------------
    # ML experiments path (original behavior)
    # ----------------------------------------------------------------
    if not args.results_dir:
        print("ERROR: --results-dir is required for --strategy ml (default)", file=sys.stderr)
        sys.exit(1)

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"ERROR: results-dir does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading results from: {results_dir}")
    records = load_result_files(results_dir)
    print(f"Loaded {len(records)} result records")

    # Sort by median_test_auc descending, take top N
    def sort_key(r):
        return extract_metric(r, 'median_test_auc', 'val_auc', 'test_auc', 'mean_test_auc', default=0.0)

    records_sorted = sorted(records, key=sort_key, reverse=True)
    top_records = records_sorted[: args.top_n]
    print(f"Evaluating top {len(top_records)} models (ranked by median_test_auc)")

    # Compute Kelly metrics for each model
    kelly_results = []
    for record in top_records:
        try:
            kr = compute_kelly_metrics(
                record=record,
                kelly_fractions=args.kelly_fractions,
                account_balance=args.account_balance,
                contract_value=args.contract_value,
            )
            kelly_results.append(kr)
        except Exception as exc:
            exp_id = record.get('exp_id', 'unknown')
            print(f"WARNING: Skipping {exp_id}: {exc}", file=sys.stderr)

    if not kelly_results:
        print("ERROR: No Kelly results computed.", file=sys.stderr)
        sys.exit(1)

    df = build_summary_table(kelly_results, args.kelly_fractions)
    print_summary(df, top_n=10)

    # Write output
    if args.output is not None:
        output_path = args.output
    else:
        output_path = results_dir / "kelly_oos_backtest.json"

    output_data = {
        'results_dir': str(results_dir),
        'top_n': args.top_n,
        'account_balance': args.account_balance,
        'contract_value': args.contract_value,
        'kelly_fractions': args.kelly_fractions,
        'n_models_evaluated': len(kelly_results),
        'kelly_results': kelly_results,
        'summary_csv': df.to_dict(orient='records'),
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nOutput written to: {output_path}")


if __name__ == '__main__':
    main()
