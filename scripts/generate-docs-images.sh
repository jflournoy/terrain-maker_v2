#!/usr/bin/env bash
# Generate documentation images from example scripts
#
# This script runs example scripts with REAL data to generate
# high-quality visualization images for documentation.
#
# WARNING: This uses real DEM and SNODAS data, which takes several minutes.
# Requires data files to be present in data/ directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==================================================================="
echo "Generating Documentation Images (with real data)"
echo "==================================================================="
echo ""
echo "⚠️  This will take several minutes to load and process real DEM/SNODAS data"
echo ""

cd "$PROJECT_ROOT"

# Sledding example - generates multiple pipeline visualization stages
echo "→ Running sledding example (with REAL data)..."
echo "  Output: docs/images/01_raw/, 02_slope_stats/, 03_slope_penalties/,"
echo "          04_score_components/, 05_final/"
echo "  Expected time: 2-5 minutes depending on data size"
echo ""

uv run python examples/detroit_snow_sledding.py --all-steps --snodas-dir data/snodas_data

echo ""
echo "==================================================================="
echo "✓ Documentation images generated successfully"
echo "==================================================================="
echo ""
echo "Generated image directories:"
echo "  - docs/images/01_raw/          (DEM, snow depth)"
echo "  - docs/images/02_slope_stats/  (slope analysis)"
echo "  - docs/images/03_slope_penalties/ (terrain penalties)"
echo "  - docs/images/04_score_components/ (score components)"
echo "  - docs/images/05_final/        (final sledding scores)"
echo ""
