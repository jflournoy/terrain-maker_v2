"""Configuration module for terrain-maker project.

Centralizes data paths and configuration settings.
"""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
SNODAS_DIR = DATA_DIR / "snodas_data"  # Symlink to /data/jflournoy/snodas_data
DEM_DIR = DATA_DIR / "dem"
SAMPLES_DIR = DATA_DIR / "samples"

# Cache directories (created as needed)
CACHE_DIR = DATA_DIR / "cache"
DEM_CACHE = CACHE_DIR / "dem"
TERRAIN_CACHE = CACHE_DIR / "terrain"
FEATURE_CACHE = CACHE_DIR / "features"
SNOW_CACHE = CACHE_DIR / "snow"

# Ensure cache directories exist
for cache_dir in [DEM_CACHE, TERRAIN_CACHE, FEATURE_CACHE, SNOW_CACHE]:
    cache_dir.mkdir(parents=True, exist_ok=True)

# Default settings
DEFAULT_DEM_PATTERN = "*.hgt"
DEFAULT_CACHE_SIZE = "1GB"
DEFAULT_LOG_LEVEL = "INFO"
