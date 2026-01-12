# MCP Quick Reference Guide

## Serena MCP - Code Intelligence

### When to Use Serena
- Finding specific functions/classes
- Understanding codebase structure
- Editing code at symbol level
- Accessing project memories
- Running shell commands

### Common Serena Workflows

#### 1. Understanding a New Module
```
1. "Use Serena to get symbols overview of ml_intraday_v3/features/returns.py"
2. "Read the project memory about code style conventions"
3. "Find all references to the calculate_returns function"
```

#### 2. Making Changes
```
1. "Find the symbol DataPipeline in the codebase"
2. "Show me the body of the DataPipeline.load_data method"
3. "Replace the body of load_data with [new implementation]"
4. "Find all code that references load_data to check for breakage"
```

#### 3. Navigating Large Codebase
```
1. "List all Python files in ml_intraday_v3/tests/"
2. "Search for pattern 'triple_barrier' in ml_intraday_v3/"
3. "Get symbols overview to see all test classes"
```

#### 4. Using Project Knowledge
```
1. "List available memories"
2. "Read the suggested_commands memory"
3. "Read the task_completion_checklist memory before starting work"
```

### Serena Memory Files (Available Now)
1. **project_overview.md** - Topstep trading system goals, constraints
2. **tech_stack.md** - Python 3.13, pandas, sklearn, pytest, ProjectX API
3. **code_style_conventions.md** - Naming, testing, config management
4. **suggested_commands.md** - CLI commands for pipeline, testing, live trading
5. **task_completion_checklist.md** - Required steps for completing tasks
6. **directory_structure.md** - ml_intraday_v3/ layout and navigation

### Pro Tips for Serena
- Always read `project_overview.md` when starting work on new area
- Check `task_completion_checklist.md` before marking task complete
- Use `find_symbol` with substring matching for exploratory search
- Use symbol-level editing for functions/classes, file-level for small changes
- Execute shell commands through Serena for consistency

---

## Context7 MCP - Library Documentation

### When to Use Context7
- Looking up library API documentation
- Finding code examples from official docs
- Checking latest features/changes in libraries
- Understanding library best practices

### Common Context7 Workflows

#### 1. Learning New Library Features
```
1. "Use Context7 to resolve library ID for pandas"
2. "Query pandas docs for handling time series with missing data"
3. "Show me examples of pandas.resample with custom aggregations"
```

#### 2. Debugging Library Issues
```
1. "Query sklearn docs for LogisticRegression class_weight parameter"
2. "Find Context7 docs on xgboost early stopping"
3. "Look up pytest fixture scope in official docs"
```

#### 3. Best Practices
```
1. "Use Context7 to find pandas best practices for large datasets"
2. "Query numpy docs for vectorization examples"
3. "Find sklearn pipeline examples with custom transformers"
```

### Libraries Commonly Used in This Project
- **pandas**: Time series manipulation, resampling
- **numpy**: Numerical computations, vectorization
- **scikit-learn**: ML models, preprocessing, cross-validation
- **pytest**: Testing framework
- **xgboost/lightgbm**: Gradient boosting models
- **pyyaml**: Config file handling

### Pro Tips for Context7
- Be specific in queries: "pandas.read_parquet with partition columns"
- Request examples: "Show me code examples of sklearn Pipeline"
- Check version compatibility: May need to specify library version
- Use for official docs only (not Stack Overflow-style answers)

---

## Combining Serena + Context7

### Workflow Example: Adding New Feature

**Step 1**: Understand current code (Serena)
```
"Use Serena to find the feature engineering modules in ml_intraday_v3/features/"
"Show me symbols overview of features/returns.py"
```

**Step 2**: Research library usage (Context7)
```
"Use Context7 to find pandas rolling window documentation"
"Query numpy docs for efficient calculation of exponential moving averages"
```

**Step 3**: Read project conventions (Serena)
```
"Read memory: code_style_conventions"
"Read memory: task_completion_checklist"
```

**Step 4**: Implement (Serena)
```
"Use Serena to insert a new function after calculate_returns"
"Execute shell command: pytest tests/test_features.py"
```

**Step 5**: Verify (Serena)
```
"Find all references to the new feature function"
"Search for pattern 'new_feature' in ml_intraday_v3/"
```

---

## Quick Command Reference

### Serena Commands (Most Used)

```python
# Find code
find_symbol(name_path_pattern="ClassName", substring_matching=True)
find_referencing_symbols(name_path="function_name", relative_path="file.py")
search_for_pattern(substring_pattern="regex", relative_path="ml_intraday_v3/")

# Navigate
get_symbols_overview(relative_path="path/to/file.py", depth=1)
list_dir(relative_path="ml_intraday_v3/", recursive=False)
find_file(file_mask="*test*.py", relative_path=".")

# Read/Edit
read_file(relative_path="path/to/file.py")
replace_symbol_body(name_path="Class/method", relative_path="file.py", body="...")
insert_after_symbol(name_path="Class", relative_path="file.py", body="...")

# Memory
list_memories()
read_memory(memory_file_name="project_overview.md")
write_memory(memory_file_name="new_memory.md", content="...")

# Utility
execute_shell_command(command="pytest tests/")
get_current_config()
```

### Context7 Commands

```python
# Library search
resolve_library_id(
    libraryName="pandas",
    query="I need pandas documentation for time series resampling"
)

# Get docs
query_docs(
    libraryId="/pandas/pandas",  # from resolve_library_id
    query="How to resample time series to 5-minute bars with OHLCV aggregation"
)
```

---

## Project-Specific Guidelines

### Always Check These Memories First
1. **Before starting any task**: Read `task_completion_checklist.md`
2. **When working in ml_intraday_v3/**: Read `directory_structure.md`
3. **Before writing code**: Read `code_style_conventions.md`
4. **When running commands**: Check `suggested_commands.md`

### Critical Project Rules (from .claude/CLAUDE.md)
- Work ONLY in `ml_intraday_v3/` directory
- Never modify legacy code without explicit permission
- Follow research-grade standards (no lookahead, proper validation)
- Every task must include tests
- Update notebook: `ml_intraday_v3_pipeline_runner_enhanced.ipynb`

### Testing Requirements
```bash
# Always run before completing task
pytest ml_intraday_v3/tests/test_<module>.py -v
```

### Documentation Requirements
Every completed task must document:
1. Files added/changed (paths)
2. How to run (exact commands)
3. Artifacts written (paths)
4. Tests added + how to run
5. Assumptions made

---

## Troubleshooting

### "Memory not found"
**Solution**: Use `list_memories()` to see available memories

### "Symbol not found"
**Solution**: Try with `substring_matching=True` or use broader search

### "Library ID invalid"
**Solution**: Call `resolve_library_id()` first to get correct ID

### "Command failed"
**Solution**: Check `suggested_commands.md` memory for correct syntax

---

**Last Updated**: 2025-01-08
**Project**: algos 3 topstep
**Serena Version**: 0.1.4
