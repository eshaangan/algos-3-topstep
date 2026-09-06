# MCP Quick Reference Guide

## Context7 MCP - Library Documentation

### When to Use Context7

- Looking up library API documentation
- Finding code examples from official docs
- Checking current features and changes in libraries
- Understanding library best practices

### Common Workflows

#### Learning library features

```text
1. "Use Context7 to resolve the library ID for pandas"
2. "Query pandas docs for handling time series with missing data"
3. "Show examples of pandas.resample with custom aggregations"
```

#### Debugging library issues

```text
1. "Query scikit-learn docs for LogisticRegression class_weight"
2. "Find XGBoost documentation for early stopping"
3. "Look up pytest fixture scope in the official docs"
```

### Libraries Commonly Used in This Project

- **pandas**: Time-series manipulation and resampling
- **NumPy**: Numerical computation and vectorization
- **scikit-learn**: Models, preprocessing, and cross-validation
- **pytest**: Testing
- **XGBoost/LightGBM**: Gradient boosting
- **PyYAML**: Configuration files

### Context7 Commands

```python
resolve_library_id(
    libraryName="pandas",
    query="Pandas documentation for time-series resampling",
)

query_docs(
    libraryId="/pandas/pandas",
    query="How do I resample time series to five-minute OHLCV bars?",
)
```

### Tips

- Use a specific query, such as `pandas.read_parquet with partition columns`.
- Request code examples when useful.
- Include the installed library version when compatibility matters.
- Resolve the library ID before querying documentation.

---

**Project**: algos 3 topstep
