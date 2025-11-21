---

allowed-tools: \[Bash]
description: Documentation maintenance and validation
approach: script-delegation
token-cost: ~100 (vs ~2000 for direct implementation)
best-for: Repetitive documentation updates
------------------------------------------

# Documentation Command

Efficiently manage documentation using dedicated scripts for Node.js and Python code.

## Usage

<bash>
#!/bin/bash

# Parse command
cmd="${1:-all}"

case "$cmd" in
  python|py)
    echo "📚 Updating Python API documentation..."
    python scripts/python-docs.py
    ;;
  python:validate|py:validate)
    echo "🔍 Validating Python documentation..."
    python scripts/python-docs.py validate
    ;;
  python:stats|py:stats)
    echo "📊 Python documentation statistics..."
    python scripts/python-docs.py stats
    ;;
  *)
    echo "🔄 Updating all documentation..."
    node scripts/docs.js "$@"
    if [ -f scripts/python-docs.py ]; then
      python scripts/python-docs.py
    fi
    ;;
esac
</bash>

## Commands

- `/docs` - Update all documentation (Node.js + Python API)
- `/docs python` - Generate Python API reference from docstrings
- `/docs python:validate` - Check for undocumented Python code
- `/docs python:stats` - Show Python documentation statistics

## What It Does

**Node.js Documentation** (from `scripts/docs.js`):
- Update README badges and statistics
- Validate internal markdown links
- List available commands
- Update command catalog

**Python Documentation** (from `scripts/python-docs.py`):
- Extract docstrings from Python source files
- Generate or update `docs/API_REFERENCE.md`
- Track classes and functions
- Identify undocumented code

For advanced operations, see `.claude/commands/detailed/docs-detailed.md`.
