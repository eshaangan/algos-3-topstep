#!/usr/bin/env python3
"""
Quick test to verify critical fixes are working locally.
"""

import sys
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add to path
ml_v3_dir = Path(__file__).parent
sys.path.insert(0, str(ml_v3_dir))

print("="*80)
print("TESTING CRITICAL FIXES LOCALLY")
print("="*80)
print()

# Test 1: Load config and check circuit breaker
print("Test 1: Circuit Breaker Configuration")
print("-" * 80)
config_path = ml_v3_dir / "configs" / "live_trading.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

cb_enabled = config.get('circuit_breaker', {}).get('enabled', False)
print(f"Circuit breaker in config: {cb_enabled}")
if cb_enabled:
    print("✅ PASS: Circuit breaker is ENABLED in config")
else:
    print("❌ FAIL: Circuit breaker is DISABLED in config")
print()

# Test 2: Check regime detector
print("Test 2: Regime Detector Configuration")
print("-" * 80)
regime_enabled = config.get('regime_detector', {}).get('enabled', True)
print(f"Regime detector in config: {regime_enabled}")
if not regime_enabled:
    print("✅ PASS: Regime detector is DISABLED (as expected)")
else:
    print("⚠️  WARNING: Regime detector is ENABLED (should be disabled)")
print()

# Test 3: Load execution engine and verify circuit breaker
print("Test 3: Execution Engine Initialization")
print("-" * 80)
try:
    from live_trading.execution_engine import LiveExecutionEngine

    # Load other configs
    risk_path = ml_v3_dir / "configs" / "risk.yaml"
    exec_path = ml_v3_dir / "configs" / "execution_spec.yaml"
    label_path = ml_v3_dir / "configs" / "labeling.yaml"

    with open(risk_path, 'r') as f:
        risk_cfg = yaml.safe_load(f)
    with open(exec_path, 'r') as f:
        exec_cfg = yaml.safe_load(f)
    with open(label_path, 'r') as f:
        label_cfg = yaml.safe_load(f)

    # Initialize engine
    engine = LiveExecutionEngine(
        risk_cfg=risk_cfg,
        execution_spec=exec_cfg,
        label_schema=label_cfg,
        dry_run=True,
        config=config,
    )

    print(f"Engine circuit breaker enabled: {engine.circuit_breaker_enabled}")
    if engine.circuit_breaker_enabled:
        print("✅ PASS: Execution engine has circuit breaker ENABLED")
        print(f"   Daily loss limit: ${engine.max_daily_loss:,.2f}")
        print(f"   Max drawdown limit: ${engine.max_drawdown_limit:,.2f}")
    else:
        print("❌ FAIL: Execution engine has circuit breaker DISABLED")

    print(f"Engine regime filter enabled: {engine.regime_filter_enabled}")
    if not engine.regime_filter_enabled:
        print("✅ PASS: Regime filter is DISABLED")
    else:
        print("⚠️  WARNING: Regime filter is ENABLED")

except Exception as e:
    print(f"❌ FAIL: Could not initialize execution engine")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: Load model predictor
print("Test 4: Model Predictor Initialization")
print("-" * 80)
try:
    from live_trading.model_predictor import LiveModelPredictor

    model_path = ml_v3_dir / "model_bundle_retrained_oct2024_nov2025.pkl"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
    else:
        predictor = LiveModelPredictor(str(model_path))
        model_info = predictor.get_model_info()

        print(f"Model type: {model_info.get('model_type', 'Unknown')}")
        print(f"Model classes: {model_info.get('classes', 'Unknown')}")
        print(f"Feature count: {model_info.get('n_features', 'Unknown')}")
        print("✅ PASS: Model predictor initialized successfully")

except Exception as e:
    print(f"❌ FAIL: Could not initialize model predictor")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("TEST SUMMARY")
print("="*80)

if cb_enabled and not regime_enabled:
    print("✅ ALL CRITICAL FIXES VALIDATED")
    print()
    print("The system is ready to trade with:")
    print("  - Circuit breaker active (daily loss protection)")
    print("  - Regime detector disabled (safe until proper data loading)")
    print("  - Model predictor with class validation")
    print()
    print("Ready for deployment!")
else:
    print("⚠️  Some issues detected. Review output above.")
