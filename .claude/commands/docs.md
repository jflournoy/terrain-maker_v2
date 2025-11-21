---
allowed-tools: \[Bash]
description: Documentation maintenance and validation
approach: script-delegation
best-for: Repetitive documentation updates
---

# Documentation Command

Manage docs using dedicated scripts for Node.js and Python.

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

- `/docs` - Update all docs (Node.js + Python API)
- `/docs python` - Generate Python API reference
- `/docs python:validate` - Check for undocumented code
- `/docs python:stats` - Python documentation stats

Updates README, validates links, generates API docs, identifies undocumented code. See `docs-detailed.md` for advanced operations.