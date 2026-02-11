#!/usr/bin/env python3
"""
Verification script to check all dependencies and files before running grid search.
"""

import subprocess
import sys
from pathlib import Path


def check_file(path: Path, description: str) -> bool:
    """Check if a file exists."""
    if path.exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {path}")
        return False


def check_command(cmd: str, description: str) -> bool:
    """Check if a command is available."""
    try:
        result = subprocess.run([cmd, '--version'], capture_output=True, text=True, timeout=5)
        print(f"✅ {description}: {cmd} available")
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"❌ {description}: {cmd} NOT FOUND")
        return False


def check_python_package(package: str) -> bool:
    """Check if a Python package is installed."""
    try:
        __import__(package)
        print(f"✅ Python package: {package}")
        return True
    except ImportError:
        print(f"❌ Python package: {package} NOT INSTALLED")
        return False


def main():
    print("=" * 80)
    print("Grid Search Setup Verification")
    print("=" * 80)
    print()
    
    all_ok = True
    
    # Check project structure
    print("1. Checking Project Structure...")
    print("-" * 80)
    root = Path(__file__).parent.parent
    
    checks = [
        (root / "data" / "MES_5min_Oct2024_Dec2025.parquet", "Data file"),
        (root / "experiments" / "grid_config.yaml", "Grid config"),
        (root / "experiments" / "comprehensive_grid_search.py", "Single experiment runner"),
        (root / "experiments" / "gcp_orchestrator.py", "GCP orchestrator"),
        (root / "experiments" / "analyze_results.py", "Results analyzer"),
        (root / "experiments" / "gcp_startup.sh", "VM startup script"),
        (root / "experiments" / "requirements_experiments.txt", "Requirements file"),
    ]
    
    for path, desc in checks:
        all_ok &= check_file(path, desc)
    print()
    
    # Check system commands
    print("2. Checking System Commands...")
    print("-" * 80)
    
    commands = [
        ("python3", "Python 3"),
        ("gcloud", "Google Cloud SDK"),
        ("gsutil", "GCS utility"),
    ]
    
    for cmd, desc in commands:
        all_ok &= check_command(cmd, desc)
    print()
    
    # Check Python packages
    print("3. Checking Python Packages...")
    print("-" * 80)
    
    packages = [
        "numpy",
        "pandas",
        "sklearn",
        "lightgbm",
        "joblib",
        "yaml",
    ]
    
    for pkg in packages:
        all_ok &= check_python_package(pkg)
    print()
    
    # Check GCP project
    print("4. Checking GCP Configuration...")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            ['gcloud', 'config', 'get-value', 'project'],
            capture_output=True,
            text=True,
            timeout=5
        )
        project = result.stdout.strip()
        if project:
            print(f"✅ GCP project: {project}")
        else:
            print("❌ GCP project not set (run: gcloud config set project <project-id>)")
            all_ok = False
    except Exception as e:
        print(f"❌ Error checking GCP project: {e}")
        all_ok = False
    print()
    
    # Summary
    print("=" * 80)
    if all_ok:
        print("✅ All checks passed! Ready to run grid search.")
        print()
        print("Next steps:")
        print("  1. Test locally: python experiments/comprehensive_grid_search.py --help")
        print("  2. Upload data: gsutil cp data/*.parquet gs://<bucket>/experiment-data/")
        print("  3. Run Phase 1: python experiments/gcp_orchestrator.py --phase 1 --num-vms 10")
        return 0
    else:
        print("❌ Some checks failed. Fix issues above before running grid search.")
        print()
        print("Common fixes:")
        print("  - Install gcloud: https://cloud.google.com/sdk/docs/install")
        print("  - Install packages: pip install -r experiments/requirements_experiments.txt")
        print("  - Set project: gcloud config set project <project-id>")
        return 1


if __name__ == '__main__':
    sys.exit(main())
