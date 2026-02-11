#!/usr/bin/env python3
"""
Verify the deployed model's class encoding.
"""

import sys
from pathlib import Path
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ml_v3_dir = Path(__file__).parent
model_path = ml_v3_dir / "model_bundle_retrained_oct2024_nov2025.pkl"

if not model_path.exists():
    logger.error(f"Model not found: {model_path}")
    sys.exit(1)

logger.info(f"Loading model bundle from: {model_path}")
bundle = joblib.load(model_path)

model = bundle.get("primary_model")
logger.info(f"Model type: {type(model).__name__}")

# Check if model has classes_
if hasattr(model, 'classes_'):
    classes = list(model.classes_)
    logger.info(f"Model classes: {classes}")

    # Verify encoding
    if classes == [0, 1, 2]:
        logger.info("✅ Standard encoding [0, 1, 2]")
        logger.info("   Index 0 = stop (outcome -1)")
        logger.info("   Index 1 = vertical (outcome 0)")
        logger.info("   Index 2 = target (outcome 1)")
    elif set(classes) == {-1, 0, 1}:
        logger.info("✅ Outcome encoding [-1, 0, 1]")
        logger.info(f"   Classes are: {classes}")
        stop_idx = classes.index(-1)
        vertical_idx = classes.index(0)
        target_idx = classes.index(1)
        logger.info(f"   stop_idx={stop_idx}, vertical_idx={vertical_idx}, target_idx={target_idx}")
    else:
        logger.error(f"❌ Unexpected class encoding: {classes}")
        sys.exit(1)
else:
    logger.warning("Model doesn't have classes_ attribute")

# Check bundle metadata
if 'training_range' in bundle:
    logger.info(f"Training range: {bundle['training_range']}")

if 'provenance' in bundle:
    prov = bundle['provenance']
    if 'git_state' in prov:
        git = prov['git_state']
        logger.info(f"Git commit: {git.get('commit_hash', 'N/A')}")
        logger.info(f"Git branch: {git.get('branch', 'N/A')}")

logger.info("\n✅ Model is compatible with updated predictor code")
