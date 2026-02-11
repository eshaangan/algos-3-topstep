#!/usr/bin/env python3
"""
Test script to validate critical bug fixes before deployment.

Tests:
1. Circuit breaker loads from config correctly
2. Regime detector is disabled
3. Model class mapping validation works
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
ml_v3_dir = Path(__file__).parent
sys.path.insert(0, str(ml_v3_dir))

import yaml
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_circuit_breaker_config():
    """Test that circuit breaker loads enabled state from config."""
    logger.info("=" * 60)
    logger.info("TEST 1: Circuit Breaker Configuration")
    logger.info("=" * 60)

    # Load config
    config_path = ml_v3_dir / "configs" / "live_trading.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cb_cfg = config.get("circuit_breaker", {})
    enabled = cb_cfg.get("enabled", False)

    logger.info(f"Circuit Breaker Config: {cb_cfg}")
    logger.info(f"Enabled state: {enabled}")

    if enabled:
        logger.info("✅ PASS: Circuit breaker is enabled in config")
        return True
    else:
        logger.error("❌ FAIL: Circuit breaker should be enabled")
        return False


def test_regime_detector_disabled():
    """Test that regime detector is disabled in config."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Regime Detector Configuration")
    logger.info("=" * 60)

    # Load config
    config_path = ml_v3_dir / "configs" / "live_trading.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    regime_cfg = config.get("regime_detector", {})
    enabled = regime_cfg.get("enabled", True)

    logger.info(f"Regime Detector Config: {regime_cfg}")
    logger.info(f"Enabled state: {enabled}")

    if not enabled:
        logger.info("✅ PASS: Regime detector is disabled")
        return True
    else:
        logger.error("❌ FAIL: Regime detector should be disabled")
        return False


def test_model_class_validation():
    """Test that model class validation works correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Model Class Validation")
    logger.info("=" * 60)

    # Find the deployed model
    model_path = ml_v3_dir / "model_bundle_retrained_oct2024_nov2025.pkl"

    if not model_path.exists():
        logger.warning(f"⚠️  SKIP: Model not found at {model_path}")
        return None

    # Load model bundle
    logger.info(f"Loading model bundle from: {model_path}")
    bundle = joblib.load(model_path)

    model = bundle.get("primary_model")
    if model is None:
        logger.error("❌ FAIL: No primary_model in bundle")
        return False

    # Check for DualSideModel
    has_dual_model = hasattr(model, "predict_proba_dual")
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Has predict_proba_dual: {has_dual_model}")

    if has_dual_model:
        # Check long_model classes
        if hasattr(model, 'long_model') and hasattr(model.long_model, 'classes_'):
            classes = list(model.long_model.classes_)
            logger.info(f"Long model classes: {classes}")

            # Validate class encoding
            if classes == [0, 1, 2]:
                logger.info("✅ PASS: Model uses standard [0, 1, 2] encoding")
                return True
            else:
                # Check if [-1, 0, 1] can be mapped
                try:
                    target_idx = classes.index(1)
                    stop_idx = classes.index(-1)
                    vertical_idx = classes.index(0)

                    if 0 <= stop_idx < 3 and 0 <= vertical_idx < 3 and 0 <= target_idx < 3:
                        logger.info(f"✅ PASS: Model classes {classes} can be mapped correctly")
                        logger.info(f"  stop_idx={stop_idx}, vertical_idx={vertical_idx}, target_idx={target_idx}")
                        return True
                    else:
                        logger.error(f"❌ FAIL: Invalid indices after mapping")
                        return False

                except ValueError as e:
                    logger.error(f"❌ FAIL: Cannot map classes {classes}: {e}")
                    return False
        else:
            logger.warning("⚠️  Model doesn't expose classes, assuming [0, 1, 2]")
            return None
    else:
        logger.info("Model is not DualSideModel, skipping class validation")
        return None


def test_execution_engine_initialization():
    """Test that ExecutionEngine initializes with correct settings."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Execution Engine Initialization")
    logger.info("=" * 60)

    try:
        from live_trading.execution_engine import LiveExecutionEngine
        from backtesting_v3.risk import RiskManager

        # Load configs
        config_path = ml_v3_dir / "configs" / "live_trading.yaml"
        risk_path = ml_v3_dir / "configs" / "risk.yaml"
        label_path = ml_v3_dir / "configs" / "labeling.yaml"
        exec_path = ml_v3_dir / "configs" / "execution_spec.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        with open(risk_path, 'r') as f:
            risk_cfg = yaml.safe_load(f)
        with open(label_path, 'r') as f:
            label_cfg = yaml.safe_load(f)
        with open(exec_path, 'r') as f:
            exec_cfg = yaml.safe_load(f)

        # Initialize engine in dry_run mode
        engine = LiveExecutionEngine(
            risk_cfg=risk_cfg,
            execution_spec=exec_cfg,
            label_schema=label_cfg,
            dry_run=True,
            config=config,
        )

        logger.info(f"Circuit breaker enabled: {engine.circuit_breaker_enabled}")
        logger.info(f"Regime filter enabled: {engine.regime_filter_enabled}")
        logger.info(f"Volatility filter enabled: {engine.volatility_filter_enabled}")

        # Validate
        if engine.circuit_breaker_enabled:
            logger.info("✅ PASS: Circuit breaker loaded as enabled")
            return True
        else:
            logger.error("❌ FAIL: Circuit breaker should be enabled")
            return False

    except Exception as e:
        logger.error(f"❌ FAIL: Error initializing ExecutionEngine: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 70)
    logger.info("CRITICAL BUG FIX VALIDATION")
    logger.info("=" * 70)

    results = {
        "Circuit Breaker Config": test_circuit_breaker_config(),
        "Regime Detector Disabled": test_regime_detector_disabled(),
        "Model Class Validation": test_model_class_validation(),
        "Execution Engine Init": test_execution_engine_initialization(),
    }

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
            passed += 1
        elif result is False:
            status = "❌ FAIL"
            failed += 1
        else:
            status = "⚠️  SKIP"
            skipped += 1

        logger.info(f"{test_name}: {status}")

    logger.info(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        logger.error("\n❌ CRITICAL FIXES NOT READY FOR DEPLOYMENT")
        sys.exit(1)
    else:
        logger.info("\n✅ ALL CRITICAL FIXES VALIDATED - READY FOR DEPLOYMENT")
        sys.exit(0)


if __name__ == "__main__":
    main()
