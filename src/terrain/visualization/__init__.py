"""Visualization utilities for terrain-maker."""

from .flow_diagnostics import (
    # Main orchestrator
    create_flow_diagnostics,
    # Individual plot functions
    save_flow_plot,
    plot_dem,
    plot_ocean_mask,
    plot_water_bodies,
    plot_endorheic_basins,
    plot_conditioned_dem,
    plot_fill_depth,
    plot_flow_direction,
    plot_drainage_area,
    plot_stream_network,
    plot_precipitation,
    plot_upstream_rainfall,
    plot_discharge_potential,
    plot_validation_summary,
    # Constants
    FLOW_COLORMAPS,
    D8_VECTORS,
)

__all__ = [
    "create_flow_diagnostics",
    "save_flow_plot",
    "plot_dem",
    "plot_ocean_mask",
    "plot_water_bodies",
    "plot_endorheic_basins",
    "plot_conditioned_dem",
    "plot_fill_depth",
    "plot_flow_direction",
    "plot_drainage_area",
    "plot_stream_network",
    "plot_precipitation",
    "plot_upstream_rainfall",
    "plot_discharge_potential",
    "plot_validation_summary",
    "FLOW_COLORMAPS",
    "D8_VECTORS",
]
