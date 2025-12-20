"""Setup configuration for the Topstep ML trading system."""

from setuptools import find_packages, setup

setup(
    name="topstep-ml",
    version="0.1.0",
    description="ML trading system for TopstepX with risk guardrails",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "tables",
        "python-dotenv",
        "requests",
        "PyYAML",
        "matplotlib",
        "seaborn",
        "jupyter",
        "ipykernel",
    ],
    python_requires=">=3.8",
)

