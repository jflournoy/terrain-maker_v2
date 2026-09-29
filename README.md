# Claude Code Command Templates 🚀

**A living reference implementation of professional Claude Code commands and workflows**

[![Version](https://img.shields.io/badge/version-2.0.0-blue)](https://github.com/rmurphey/claude-setup)
[![Commands](https://img.shields.io/badge/commands-14-green)](.claude/commands/)
[![License](https://img.shields.io/badge/license-MIT-purple)](LICENSE)
[![Token Efficiency](https://img.shields.io/badge/token%20savings-87%25-orange)](docs/TOKEN_EFFICIENCY.md)
[![Agent Audit](https://github.com/rmurphey/claude-setup/workflows/Agent%20Audit%20with%20Claude%20Code/badge.svg)](https://github.com/rmurphey/claude-setup/actions/workflows/agent-audit.yml)

## What This Is

This repository serves **dual purposes**:

1. **Terrain-Maker Python Package**: A reusable library for creating custom terrain visualizations with arbitrary spatial data overlays
2. **Claude Code Development Framework**: A working reference repository that demonstrates opinionated practices for Claude Code development using TDD workflows

### 🗺️ Terrain-Maker: Flexible Geospatial Visualization

**Package Goal**: Create a Python library that makes it easy to:

- Load and visualize Digital Elevation Models (DEMs) from any source
- Apply custom transformations and analysis
- Render 3D terrain visualizations with Blender
- Cache results for efficient reprocessing

**Example Projects**: Detroit Terrain Analysis

- **Real Elevation**: Load actual SRTM tiles covering Detroit metro (1.3M vertices mesh)
- **Multi-View Rendering**: Generate renders from any cardinal direction with intelligent camera positioning
- **Visualization**: 3D terrain with color-mapped overlays in Blender
- **Easy CLI Control**: Use npm commands to quickly generate comparison views: `npm run py:example:detroit-north`, etc.

**Key Features**:

- **DEM Processing**: Load and merge Digital Elevation Models (SRTM .hgt, GeoTIFF)
- **Flexible Data Layers**: Add any spatial dataset (raster or vector formats)
- **Geospatial Operations**: Reproject, resample, smooth, and transform terrain data
- **Intuitive Camera Control**: Position cameras with cardinal directions (north, south, east, west, above) with intelligent framing
- **Blender Integration**: Generate high-quality 3D renderings with custom materials
- **Smart Caching**: Avoid recomputation with hash-based caching system
- **Extensible Design**: Easy to add new data sources and visualization methods

👉 **[See Python Setup Guide](docs/PYTHON_SETUP.md)** for installation and usage instructions

📚 **[API Reference](docs/API_REFERENCE.md)** - Complete documentation of all methods and classes

## Quick Start: Using This Repository

### Python Package Installation

For the **terrain-maker Python package**:

```bash
# See Python Setup Guide for installation
# (currently requires manual setup from source)
cd terrain-maker_v2
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

See [Python Setup Guide](docs/PYTHON_SETUP.md) for detailed instructions.

### Verifying an Update on Your Machine

Cloud sessions can't run everything: they have no real DEM or score data, no Blender GPU
render, and limited network. After pulling, run the rest locally:

```bash
scripts/verify-local.sh            # deps, full tests, mock pins, real-data pins
scripts/verify-local.sh --quick    # deps + full tests only
scripts/verify-local.sh --render   # also render the saved preset for a visual check
```

Results land in `verify-output/<date>-<commit>/report.md`, with one log per step, the reasons
any tests were skipped, and anything that changed. Real-data outputs are compared against a
baseline in `tools/pins/local/` (recorded on the first run, not committed). Accept intended
changes with `uv run python tools/pin_combined_render.py record --real`.

`scripts/install-verify-hook.sh` adds a git hook that reminds you after each `git pull`, or
runs the check automatically if you set `TERRAIN_VERIFY_ON_PULL=quick` or `full`.

### Claude Code Commands

The `.claude/commands/` directory contains 14+ ready-to-use workflow commands:

```bash
/hygiene                          # Check project health
/tdd start "new feature"          # Start with tests
/commit feat "add user auth"      # Quality-checked commit
/learn "TDD clarified the API"    # Capture insights
/todo                             # Manage tasks
/next                             # Get next steps
/docs                             # Update documentation
/push                             # Push with quality checks
```

These commands are **integrated into Claude Code** - just type the slash command in the Claude Code UI.

## Why This Repository is Useful

✅ **Dual Purpose**: Works as both a Python geospatial library AND a Claude Code reference implementation
✅ **TDD Foundation**: All new code developed test-first for clarity and correctness
✅ **Best Practices**: Based on Anthropic's guidelines and 2025 industry standards
✅ **Production-Ready**: Real working code for terrain visualization with extensible design
✅ **Self-Improving**: Repository uses its own commands - see our git history to learn from practice

## 🤖 Automated Intelligence Features

This repository includes agents that can use the Claude API for advanced automation:

- **Automated agent quality audits** via GitHub Actions
- **Intelligent code analysis** beyond simple pattern matching
- **Deep project insights** using Claude's reasoning capabilities

👉 **[See API Setup Guide](docs/API_SETUP.md)** for configuration instructions

## 🎯 Repository Philosophy

This repository demonstrates balanced approaches to Claude Code commands, optimizing for both learning and efficiency. We achieve **87% token reduction** through strategic script delegation while maintaining flexibility where needed.

👉 **[See Token Efficiency Guide](docs/TOKEN_EFFICIENCY.md)** for detailed guidance on when to use different command approaches.

## Command Categories

### 🎯 Core Workflow Commands

- **`/hygiene`** - Comprehensive project health check ([view command](.claude/commands/hygiene.md))
- **`/todo`** - Task management with GitHub Issues ([view command](.claude/commands/todo.md))
- **`/commit`** - Quality-checked commits ([view command](.claude/commands/commit.md))
- **`/next`** - AI-recommended next steps ([view command](.claude/commands/next.md))

### 📚 Documentation & Learning

- **`/docs`** - Documentation generation ([view command](.claude/commands/docs.md))
- **`/learn`** - Capture insights ([view command](.claude/commands/learn.md))

### 🚀 Release & Quality

- **`/push`** - Push with quality checks ([view command](.claude/commands/push.md))
- **`/tdd`** - Test-driven development workflow ([view command](.claude/commands/tdd.md))
- **`feature:check`** - Verify new features have tests & docs ([documentation](docs/FEATURE_CHECK.md))

## 🤖 Claude Code Agents

**When commands aren't enough, agents provide intelligent analysis and complex automation.**

### Commands vs Agents

- **Commands**: Routine tasks you do the same way every time (`/hygiene`, `/commit`)
- **Agents**: Complex analysis requiring "intelligence" and decision-making

### Available Agents

#### 📊 Analysis & Optimization

- **`command-analyzer`** - Analyzes command usage patterns and suggests optimizations
- **`session-insights`** - Extracts patterns from development session history
- **`documentation-auditor`** - Audits documentation completeness and consistency
- **`repo-quality-auditor`** - Comprehensive repository audit for completeness and conflicts

#### 🎯 Planning & Guidance

- **`next-priorities`** - Analyzes project state to recommend next development priorities
- **`usage-estimator`** - Provides intelligent Claude usage estimates for development tasks

#### 🧪 Testing & Quality

- **`test-coverage-advisor`** - Identifies untested code and recommends testing opportunities
- **`agent-auditor`** - Audits other agents for quality, correctness, and relevance

### When to Use Agents

Use agents when you need:

- 🧠 **Analysis**: Analysis across multiple files with pattern recognition
- 📈 **Insights**: Deep understanding of your development practices
- 🔧 **Optimization**: Strategic improvements to your workflows
- 📋 **Planning**: Custom workflows for complex, multi-step processes
- 🤖 **Self-Maintenance**: Automated quality audits via GitHub Actions

### Quick Examples

```bash
# Use a command for routine tasks
/hygiene

# Use an agent for analysis
"Use the session-insights agent to analyze my development patterns from the last 3 months"

# Use an agent for optimization  
"Use the command-analyzer agent to find opportunities to streamline my command library"
```

**📖 See [AGENTS.md](AGENTS.md) for the complete guide on when and why to use agents vs commands.**

## Development Method

This repository uses Test-Driven Development (TDD) which helps Claude write focused, correct code.
See [TDD with Claude Guide](docs/TDD_WITH_CLAUDE.md) for why it works so well with AI assistance.

## Real-World Usage Examples

### This Repository Uses Its Own Commands

Explore our git history to see these commands in action:

```bash
git log --grep="Generated with Claude Code" --oneline
```

Check our project health:

```bash
# In Claude Code
/hygiene
```

See our active work:

```bash
npm run todo:list  # Shows open GitHub issues
```

## Example Workflow

```bash
/hygiene                          # Check project health
/tdd start "new feature"          # Start with tests
/commit feat "add user auth"      # Quality-checked commit
/learn "TDD clarified the API"    # Capture insights
```

👉 **[See Complete Workflows Guide](docs/WORKFLOWS.md)** for detailed development patterns

## Customization Guide

### Adapting Commands to Your Needs

All commands are customizable markdown files in `.claude/commands/`. Common customizations:

#### Adjust Quality Thresholds

Edit `/hygiene` command to match your standards:

```bash
# In .claude/commands/hygiene.md
# Change from "max-warnings 10" to your preference
npx eslint . --max-warnings 0  # Strict: no warnings
```

#### Add Project-Specific Checks

Extend commands with your tools:

```bash
# In .claude/commands/commit.md
# Add your specific checks
npm run typecheck
npm run your-custom-check
```

#### Create Custom Commands

Copy any template as a starting point:

```bash
cp .claude/commands/hygiene.md .claude/commands/deploy.md
# Edit to create deployment workflow
```

## Command Categories

### 🏃 Quick Actions (< 1 min)

`/todo`, `/next`, `/hygiene`

### 📝 Documentation & Learning

`/docs`, `/learn`, `/retrospective`

### 🔧 Development Workflow

`/commit`, `/push`, `/tdd`

## Understanding Command Templates

Each command is a structured markdown file that guides Claude through specific workflows:

```markdown
---
allowed-tools: [Bash, Read, Write]  # Tools Claude can use
description: Brief command description
---

# Command Name

Detailed instructions for Claude to execute...

When you type `/<command>` in Claude Code:
1. Claude reads the template from `.claude/commands/<command>.md`
2. Executes the workflow described in the template
3. Uses only the allowed tools specified
4. Provides consistent, professional assistance
```

## Repository Structure

```
.claude/
├── commands/           # Command templates (23+ files)
│   ├── hygiene.md     # Project health checks
│   ├── commit.md      # Quality-checked commits
│   ├── todo.md        # Task management
│   └── ...            # More production-ready commands
├── learnings/         # Monthly learning archives
├── agents/            # Claude Code agents
└── session-history/   # Development session archives

CLAUDE.md              # Project AI guidelines
GitHub Issues          # Task tracking (via /todo command)
package.json           # NPM scripts for token efficiency

docs/
├── BEST_PRACTICES.md  # Comprehensive best practices guide
├── COMMAND_CATALOG.md # Detailed command reference
├── TOKEN_EFFICIENCY.md # Token optimization strategies
└── QUICK_REFERENCE.md # Quick command lookup
```

## Tips for Success

### 🎯 Start Small

Begin with core commands: `/hygiene`, `/todo`, `/commit`

### 📊 Track Progress

Use `/todo` and `/next` to maintain focus

### 🔄 Regular Health Checks

Run `/hygiene` before and after major changes

### 💡 Capture Learnings

Use `/learn` to build project knowledge base

### 🎨 Customize Gradually

Adapt templates as you learn what works for your team

## Advanced Usage

### Continuous Testing

Run tests automatically as you code:

```bash
npm run test:watch    # Node.js watch mode - reruns on file changes
```

### Chaining Commands

```bash
/hygiene && /todo list && /next
# Full status check → task list → recommendations
```

### Command Aliases

Create shortcuts for common workflows:

```bash
# In .claude/commands/status.md
# Combine hygiene + todo + next into one command
```

### CI/CD Integration

Use command patterns in your automation:

```bash
# In .github/workflows/ci.yml
# Implement same checks as /hygiene command
```

## Documentation

### Essential Guides

- **[Best Practices Guide](docs/BEST_PRACTICES.md)** - Claude Code best practices with citations
- **[Command Catalog](docs/COMMAND_CATALOG.md)** - Detailed reference for all commands
- **[Token Efficiency](docs/TOKEN_EFFICIENCY.md)** - How we achieve 87% token reduction
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Quick command lookup

### Getting Help

- **💬 Issues**: [GitHub Issues](https://github.com/rmurphey/claude-setup/issues)
- **🤝 Discussions**: [GitHub Discussions](https://github.com/rmurphey/claude-setup/discussions)
- **📧 Contact**: Via GitHub profile

## Why Use This Reference Implementation?

### Proven Benefits

✅ **87% Token Reduction** - Measured and documented savings\
✅ **Battle-Tested** - Commands refined through real usage\
✅ **Living Documentation** - See actual usage in git history\
✅ **Best Practices** - Based on Anthropic guidelines and industry standards\
✅ **Self-Improving** - Repository uses its own commands for development

### Informed By ...

- [Anthropic's Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [How Anthropic Teams Use Claude Code](https://www.anthropic.com/news/how-anthropic-teams-use-claude-code)
- [Conventional Commits v1.0.0](https://www.conventionalcommits.org/)
- Community patterns from [Awesome Claude Code](https://github.com/hesreallyhim/awesome-claude-code)
- Real-world testing and metrics

## Metrics and Validation

This repository tracks its own usage metrics:

- Token usage per command
- Execution time statistics
- Error rates and recovery patterns
- Real workflow timings

See [.claude/metrics.json](.claude/metrics.json) for current data.

## 🤖 Automated Agent Audits

This repository uses Claude Code CLI to automatically audit agents via GitHub Actions.

**Quick Setup**: Add `ANTHROPIC_API_KEY` to GitHub Secrets → Actions

👉 **[Complete API Setup Guide](docs/API_SETUP.md)** - Detailed configuration, security, and troubleshooting

The audit runs weekly and validates agent quality using Claude's intelligence, not just pattern matching. See `.claude/agents/agent-auditor.md` for the audit logic.

## Contributing

This repository serves as a reference, so all contributions must:

- Include working examples
- Document token efficiency
- Provide citations for claims
- Pass our test suite
- Include tests as appropriate

## Version History

- **v2.0.0** - Complete restructure as living reference implementation
- **v1.0.0** - Initial template collection

See git history for detailed change log.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

*A living reference implementation for the Claude Code community*\
*This repository uses its own commands - explore our git history to see them in action*

### 📚 Living Examples: Learning from Our History

Explore how we use our own tools in practice. Each example demonstrates a key principle:

#### Simplification & Token Efficiency

- [feat: remove all detailed command variants for token efficiency](../../commit/938fd09) — Less is more
- [feat: remove 3 more complex maintenance commands](../../commit/f69d8e2) — Ruthless simplification
- [refactor: remove script tests in favor of utility testing](../../commit/03001e4) — Focus testing effort

#### Architecture & Design

- [feat: rewrite setup.js with intelligent conflict handling](../../commit/94aa977) — Smart automation
- [feat: add repo-quality-auditor agent for comprehensive quality analysis](../../commit/1a2b2fb) — Agent vs command pattern
- [refactor: migrate to GitHub Issues for task management](../../commit/de7ca65) — Modern workflow

#### Continuous Improvement

- [fix: update docs script to count all commands recursively](../../commit/50afdcc) — Iterative refinement
- [docs: add critical instruction to always use date command](../../commit/a207133) — Learning from mistakes
