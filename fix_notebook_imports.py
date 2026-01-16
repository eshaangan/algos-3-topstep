"""
Quick fix for notebook import issues.
Run this in the FIRST cell of the notebook before anything else.
"""

import sys
from pathlib import Path

# Add project root to Python path
repo_root = Path.cwd().parent if Path.cwd().name == "ml_intraday_v3" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

print(f"✓ Added to Python path: {repo_root}")
print(f"✓ Can now import from: ml_intraday_v3/")

# Verify imports work
try:
    from ml_intraday_v3.experiments.diagnostics import compute_dsr
    print("✓ Import test successful!")
except ImportError as e:
    print(f"✗ Import still failing: {e}")
    print(f"\nCurrent working directory: {Path.cwd()}")
    print(f"Python path:\n  " + "\n  ".join(sys.path[:5]))
