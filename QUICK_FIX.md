# Quick Fix for Notebook Import Error

## The Error You're Seeing
```
ModuleNotFoundError: No module named 'ml_intraday_v3'
```

## Solution: Add This to Cell #1

**Before running the notebook, modify the FIRST cell** to include this path fix:

### Option 1: Replace Cell #1 Content

Find the first cell (where it sets RUN_ID), and add this at the **very top**:

```python
# ============================================================================
# FIX IMPORTS - Add parent directory to path
# ============================================================================
import sys
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "ml_intraday_v3" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

print(f"✓ Python path fixed: {repo_root}")

# ============================================================================
# Original cell content starts here
# ============================================================================
from pathlib import Path
from datetime import datetime, timezone
import uuid
import os, json, subprocess, sys
import importlib.util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ... rest of the original cell
```

---

## Option 2: Even Simpler - Change the Import

In **Cell #2** (where the error occurs), change:

### From:
```python
from ml_intraday_v3.experiments.diagnostics import compute_dsr
```

### To:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "ml_intraday_v3" else Path.cwd()))

from ml_intraday_v3.experiments.diagnostics import compute_dsr
```

---

## Option 3: Run Jupyter from Correct Directory

Close the notebook and restart Jupyter from the **repo root**:

```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep
# Make sure you're in the repo root (not in ml_intraday_v3/ subdirectory)
pwd  # Should show: /Users/eshaanganguly/Documents/projects/algos 3 topstep

jupyter notebook ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb
```

The notebook should open, and the imports will work because Python can now find `ml_intraday_v3/` as a subdirectory.

---

## Option 4: Use Relative Imports

If the above don't work, modify **Cell #2** to use relative imports:

### From:
```python
from ml_intraday_v3.experiments.diagnostics import compute_dsr
```

### To:
```python
# Add parent to path
import sys
from pathlib import Path
parent = Path(__file__).parent if '__file__' in globals() else Path.cwd()
if parent.name == "ml_intraday_v3":
    parent = parent.parent
sys.path.insert(0, str(parent))

# Now import
from ml_intraday_v3.experiments.diagnostics import compute_dsr
```

---

## Verification

After applying the fix, run this in a cell to verify:

```python
import sys
from pathlib import Path

print("Current directory:", Path.cwd())
print("\nPython path (first 5):")
for p in sys.path[:5]:
    print(f"  {p}")

# Test import
try:
    from ml_intraday_v3.experiments.diagnostics import compute_dsr
    print("\n✓ Import successful!")
except ImportError as e:
    print(f"\n✗ Still failing: {e}")
```

---

## Recommended Solution

**Use Option 3** (restart Jupyter from repo root) - it's cleanest:

1. Close the current notebook
2. Close Jupyter
3. Open terminal:
   ```bash
   cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep
   jupyter notebook ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb
   ```
4. Run the notebook

This ensures all imports work correctly throughout.

---

## Still Not Working?

If you're still having issues, check:

1. **You're in the right directory**:
   ```bash
   pwd
   # Should show: /Users/eshaanganguly/Documents/projects/algos 3 topstep
   # NOT: /Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3
   ```

2. **The ml_intraday_v3 directory exists**:
   ```bash
   ls ml_intraday_v3/
   # Should show: cli.py, configs/, data/, features/, labels/, etc.
   ```

3. **Try installing as editable package**:
   ```bash
   cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep
   pip install -e .
   ```
   (Only if there's a setup.py file)

Let me know which solution works for you!
