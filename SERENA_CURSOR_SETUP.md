# Serena & Context7 MCP Setup Guide for Cursor

## Current Status

### ✅ Claude Code (CLI)
Serena MCP is **fully configured and onboarded** for Claude Code:
- Project: "algos 3 topstep" activated
- 6 memory files created:
  - `project_overview.md` - Project goals and context
  - `tech_stack.md` - Technologies and dependencies
  - `code_style_conventions.md` - Coding standards
  - `suggested_commands.md` - Common development commands
  - `task_completion_checklist.md` - Task completion workflow
  - `directory_structure.md` - Codebase navigation
- Language: Python 3.13.5
- Context: desktop-app
- Modes: interactive, editing

### 🔧 Cursor IDE
Needs MCP configuration (see setup instructions below)

---

## Setup Instructions for Cursor

### Step 1: Install Serena MCP

Serena MCP is available from the official repository. Install it using one of these methods:

#### Option A: Using UV (Recommended)
```bash
# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Serena globally
uv tool install serena-mcp
```

#### Option B: Using pipx
```bash
# Install pipx if you don't have it
brew install pipx
pipx ensurepath

# Install Serena
pipx install serena-mcp
```

#### Option C: Using pip in virtual environment
```bash
# Create a dedicated environment for MCPs
python -m venv ~/.mcp-servers
source ~/.mcp-servers/bin/activate
pip install serena-mcp
```

After installation, verify Serena is accessible:
```bash
serena --version
# or
python -m serena --version
```

### Step 2: Configure Cursor MCP Settings

Cursor uses MCP configuration to connect to Model Context Protocol servers. Create or update the configuration file:

**Location**: `~/.cursor/mcp_config.json` or within Cursor settings

Create the file with the following content:

```json
{
  "mcpServers": {
    "serena": {
      "command": "serena",
      "args": ["serve"],
      "env": {
        "SERENA_PROJECT_PATH": "/Users/eshaanganguly/Documents/projects/algos 3 topstep"
      }
    },
    "context7": {
      "command": "npx",
      "args": ["-y", "@context7/mcp-server"]
    }
  }
}
```

**Alternative** if you installed Serena with pip in a virtual environment:
```json
{
  "mcpServers": {
    "serena": {
      "command": "/Users/eshaanganguly/.mcp-servers/bin/python",
      "args": ["-m", "serena", "serve"],
      "env": {
        "SERENA_PROJECT_PATH": "/Users/eshaanganguly/Documents/projects/algos 3 topstep"
      }
    },
    "context7": {
      "command": "npx",
      "args": ["-y", "@context7/mcp-server"]
    }
  }
}
```

### Step 3: Configure Serena Project in Cursor

Once Cursor can connect to Serena MCP, you need to ensure it uses the same project configuration:

1. **Project file location**: `.serena/project.yml` (already exists in your project)

2. **Verify the configuration**:
```bash
cat .serena/project.yml
```

The key settings are:
- `languages: [python]`
- `encoding: "utf-8"`
- `project_name: "algos 3 topstep"`
- `ignore_all_files_in_gitignore: true`

3. **Memory files** are already created in `.serena/memories/`:
   - All 6 memory files from the onboarding are available

### Step 4: Restart Cursor

After configuring MCP:
1. Quit Cursor completely (Cmd+Q)
2. Restart Cursor
3. Open your project: `/Users/eshaanganguly/Documents/projects/algos 3 topstep`
4. Verify MCP connection in Cursor's settings or output panel

---

## Usage in Cursor

### Serena MCP Tools Available in Cursor

Once configured, Cursor will have access to all Serena tools:

#### Code Navigation
- `find_symbol` - Find classes, functions, methods by name
- `find_referencing_symbols` - Find where symbols are used
- `get_symbols_overview` - Get file structure overview
- `search_for_pattern` - Regex search across codebase

#### File Operations
- `read_file` - Read file contents
- `create_text_file` - Create or overwrite files
- `list_dir` - List directory contents
- `find_file` - Find files by pattern

#### Code Editing (Symbol-Level)
- `replace_symbol_body` - Replace entire function/class definition
- `insert_after_symbol` - Add code after a symbol
- `insert_before_symbol` - Add code before a symbol
- `rename_symbol` - Rename with refactoring

#### Memory System
- `read_memory` - Read project memories
- `list_memories` - List available memories
- `write_memory` - Store new information
- `edit_memory` - Update existing memory
- `delete_memory` - Remove memory

Available memories:
1. `project_overview.md` - High-level project context
2. `tech_stack.md` - Technologies used
3. `code_style_conventions.md` - Coding standards
4. `suggested_commands.md` - Common CLI commands
5. `task_completion_checklist.md` - Task workflow
6. `directory_structure.md` - Codebase layout

#### Project Management
- `get_current_config` - View Serena configuration
- `execute_shell_command` - Run terminal commands

### Context7 MCP Tools Available in Cursor

- `resolve-library-id` - Find Context7-compatible library IDs
- `query-docs` - Retrieve up-to-date documentation and code examples

Example workflow:
```
1. Ask Cursor: "How do I use pandas.read_parquet with partitioned datasets?"
2. Cursor uses resolve-library-id to find pandas library ID
3. Cursor uses query-docs to fetch latest pandas documentation
4. Cursor provides answer with citations
```

---

## Testing the Setup

### Test Serena Connection
Ask Cursor:
```
Use Serena to list memories and show me the project overview
```

Expected: Cursor should use `list_memories` and `read_memory` tools to show project information.

### Test Context7 Connection
Ask Cursor:
```
Use Context7 to find the latest pandas documentation for read_parquet
```

Expected: Cursor should use `resolve-library-id` and `query-docs` to fetch documentation.

### Test Integrated Workflow
Ask Cursor:
```
Use Serena to find all test files in ml_intraday_v3/tests/ and explain the testing structure
```

Expected: Cursor should use `find_file` or `list_dir` to locate test files.

---

## Troubleshooting

### Serena Not Found
**Issue**: Cursor can't execute `serena` command

**Solutions**:
1. Verify installation: `serena --version`
2. Use full path to Serena in config:
   ```json
   "command": "/path/to/serena"
   ```
3. Or use Python module form:
   ```json
   "command": "python",
   "args": ["-m", "serena", "serve"]
   ```

### Connection Refused
**Issue**: Cursor can't connect to MCP server

**Solutions**:
1. Check Cursor's MCP logs (usually in: `~/Library/Logs/Cursor/`)
2. Verify MCP config file location and syntax
3. Try manually running: `serena serve` to see error messages
4. Restart Cursor after config changes

### Project Not Found
**Issue**: Serena can't find your project

**Solutions**:
1. Verify `SERENA_PROJECT_PATH` in config points to correct directory
2. Check `.serena/project.yml` exists in project root
3. Ensure project was activated in Claude Code first (already done ✓)

### Context7 Issues
**Issue**: Context7 MCP not responding

**Solutions**:
1. Verify Node.js is installed: `node --version`
2. Check internet connection (Context7 needs network access)
3. Try running manually: `npx -y @context7/mcp-server`

### Memory Files Not Loading
**Issue**: Serena can't read memory files

**Solutions**:
1. Verify memory files exist: `ls .serena/memories/`
2. Check file permissions: `chmod 644 .serena/memories/*.md`
3. Re-run onboarding if needed (already complete ✓)

---

## Alternative: Cursor Settings UI

Some versions of Cursor may have a UI for MCP configuration:

1. Open Cursor Settings (Cmd+,)
2. Search for "MCP" or "Model Context Protocol"
3. Add servers manually through the UI:
   - **Name**: serena
   - **Command**: serena serve
   - **Working Directory**: /Users/eshaanganguly/Documents/projects/algos 3 topstep

---

## Verification Checklist

- [ ] Serena MCP installed (`serena --version` works)
- [ ] MCP config file created with both Serena and Context7
- [ ] Project path correctly set in config
- [ ] `.serena/project.yml` exists and is valid
- [ ] Memory files present in `.serena/memories/`
- [ ] Cursor restarted after configuration
- [ ] Test query in Cursor successfully uses Serena tools
- [ ] Test query in Cursor successfully uses Context7 tools

---

## Next Steps After Setup

Once both MCPs are working in Cursor:

1. **Leverage Memories**: Cursor can now read project context from memory files
2. **Symbol Navigation**: Use Serena's symbol search for faster code navigation
3. **Documentation**: Context7 provides up-to-date library docs
4. **Consistency**: Both Claude Code and Cursor use same project knowledge base

---

## Notes

- **Memory Sync**: Both Claude Code and Cursor read from the same `.serena/memories/` directory
- **Project State**: Changes made in either tool are visible to the other
- **Config Independence**: Claude Code and Cursor use separate MCP configurations but share project state
- **Performance**: Serena's symbol-based tools are faster than full file reads for large codebases

---

**Last Updated**: 2025-01-08
**Serena Version**: 0.1.4
**Project**: algos 3 topstep
