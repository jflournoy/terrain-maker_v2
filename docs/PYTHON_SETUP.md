# Python Terrain-Maker Setup

## Overview

The terrain-maker component is a Python 3.11+ geospatial data processing and visualization toolkit integrated with Blender 3D.

## Prerequisites

- Python 3.11 or higher
- Conda (recommended) or pip
- Blender 4.3+ (for 3D rendering features)
- GDAL/GEOS libraries (for geospatial processing)

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate rayshade
```

### Option 2: Using pip

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

The project uses symbolic links to access large data files without duplication:

### Data Directory Structure

```
data/
├── snodas_data/          # → Symlink to /data/jflournoy/snodas_data (340GB)
├── dem/
│   └── detroit/          # → Symlink to SRTM .hgt files
├── samples/              # Small test files (copied)
└── cache/                # Generated caches (gitignored)
    ├── dem/
    ├── terrain/
    ├── features/
    └── snow/
```

### Verify Data Access

```bash
# Check symlinks
ls -lh data/snodas_data
ls -lh data/dem/detroit

# Check samples
ls -lh data/samples/
```

## Development Workflow

### Running Tests

```bash
# Run all tests
npm run py:test

# Run with coverage
npm run py:test:cov

# Run specific test file
pytest tests/test_config.py -v
```

### Code Quality

```bash
# Format code
npm run py:format

# Check formatting
npm run py:format:check

# Lint code
npm run py:lint

# Type check
npm run py:type

# Run all checks
npm run py:check
```

### TDD Workflow with Claude

```bash
# Start TDD workflow for a new feature
/tdd start "terrain elevation processing"

# Let Claude guide you through:
# 1. Write failing test (RED)
# 2. Write minimal code (GREEN)
# 3. Refactor (REFACTOR)
# 4. Commit (COMMIT)
```

## Project Structure

```
terrain-maker_v2/
├── src/                    # Python source code
│   ├── config.py           # Configuration (data paths)
│   ├── terrain/            # Terrain processing
│   │   ├── core.py         # Terrain class
│   │   ├── transforms.py   # Transform pipeline
│   │   ├── rendering.py    # Blender integration
│   │   └── cache.py        # Caching system
│   ├── snow/               # Snow analysis
│   │   └── analysis.py     # SnowAnalysis class
│   └── utils/              # Utilities
│       ├── helpers.py      # Helper functions
│       └── __init__.py
├── tests/                  # Pytest tests
│   ├── conftest.py         # Test fixtures
│   ├── test_config.py      # Config tests
│   └── test_basic.py       # Basic tests
├── examples/               # Example scripts
├── data/                   # Data directory (see above)
└── docs/                   # Documentation
```

## Usage Examples

### Basic Terrain Processing

```python
from src.terrain.core import Terrain
from src.config import DEM_DIR

# Initialize terrain
terrain = Terrain()

# Load DEM data
dem_files = list((DEM_DIR / "detroit").glob("*.hgt"))
terrain.load_dem_files(dem_files)

# Process terrain
terrain.process()

# Render with Blender
terrain.render(output_path="output/terrain.png")
```

### Snow Analysis

```python
from src.snow.analysis import SnowAnalysis
from src.config import SNODAS_DIR

# Initialize snow analysis
snow = SnowAnalysis(
    snodas_root_dir=SNODAS_DIR,
    cache_dir="data/cache/snow"
)

# Process snow data
stats, metadata, failed = snow.process_snow_data()

# Calculate sledding scores
sledding_score = snow.calculate_sledding_score(
    min_depth_mm=100,
    min_coverage=0.3
)

# Visualize
snow.visualize_snow_data(data_type='sledding_score')
```

## Cache Management

The project automatically caches processed data to avoid recomputation:

```python
from src.config import CACHE_DIR

# Caches are stored in:
# - data/cache/dem/        - DEM preprocessing
# - data/cache/terrain/    - Terrain meshes
# - data/cache/features/   - Feature calculations
# - data/cache/snow/       - Snow statistics

# To clear caches:
# rm -rf data/cache/*
```

## Integration with Claude Code

This project uses Claude Code's TDD workflow for development:

1. **`/tdd`** - Test-driven development cycle
2. **`/hygiene`** - Check Python code quality
3. **`/commit`** - Quality-checked commits
4. **`/learn`** - Capture development insights

See CLAUDE.md in the project root for full Claude Code guidelines.

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the project root and have activated the environment:

```bash
# Check current directory
pwd  # Should be terrain-maker_v2/

# Activate environment
conda activate rayshade  # Or: source .venv/bin/activate
```

### GDAL/Rasterio Issues

If rasterio installation fails:

```bash
# Ubuntu/Debian
sudo apt-get install libgdal-dev

# macOS
brew install gdal

# Then reinstall
pip install rasterio
```

### Blender Not Found

Blender is optional for visualization only. Install from:

- https://www.blender.org/download/

## Contributing

1. Follow TDD workflow (`/tdd`)
2. Run quality checks before commit (`npm run py:check`)
3. Update tests for new features
4. Document in docstrings

## Resources

- [Design Specification](design-spec.md) - Original terrain-maker design
- [TDD with Claude](TDD_WITH_CLAUDE.md) - TDD workflow guide
- [API Setup](API_SETUP.md) - Claude API configuration
