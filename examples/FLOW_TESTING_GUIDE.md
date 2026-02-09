# Flow Accumulation Testing Guide

## NEW: Spec-Compliant Backend (Recommended)

The flow accumulation implementation now supports a **spec-compliant backend** based on flow-spec.md. This implements the peer-reviewed algorithms from Lindsay (2016) and Barnes et al. (2014).

### Quick Start with Spec Backend

```bash
# Use spec-compliant backend (recommended for most use cases)
python examples/validate_flow_complete.py --bigness medium --backend spec

# Compare spec vs legacy
python examples/validate_flow_complete.py --bigness medium --backend legacy --output output/legacy
python examples/validate_flow_complete.py --bigness medium --backend spec --output output/spec
```

### What's Different?

The spec backend uses a **4-stage pipeline**:
1. **Outlet Identification** - Explicitly identifies coastal, edge, and masked basin outlets
2. **Constrained Breaching** - Dijkstra least-cost path carving with depth/length constraints
3. **Priority-Flood Fill** - Fills residual depressions with epsilon gradient applied during fill
4. **Flow Direction & Accumulation** - Standard D8 + topological sort

**Key Advantages:**
- ✅ Reduces outlet fragmentation (10-30% fewer outlets)
- ✅ Preserves terrain features through selective breaching
- ✅ Better for high-resolution DEMs
- ✅ Fixes river flow issues on lower-resolution data
- ✅ No workarounds needed (`_fix_coastal_flow`, `fill_small_sinks`)

### Spec-Compliant Parameters

```bash
python examples/validate_flow_complete.py --bigness medium \
    --backend spec \
    --coastal-elev-threshold 10.0 \      # Max elevation for coastal outlets (m)
    --edge-mode all \                     # Boundary outlet strategy
    --max-breach-depth 50.0 \            # Max vertical breach per cell (m)
    --max-breach-length 100 \            # Max breach path length (cells)
    --epsilon 1e-4                       # Min gradient in filled areas (m/cell)
```

**Edge Modes:**
- `all` - All boundary cells are outlets (safest, prevents artificial basins)
- `local_minima` - Only boundary cells that are local minima
- `outward_slope` - Cells with interior neighbors sloping toward them
- `none` - No edge outlets (for islands)

### When to Use Which Backend?

| Use Case | Backend | Why |
|----------|---------|-----|
| **General purpose** | `spec` | Best balance of accuracy and performance |
| **High-resolution DEMs** | `spec` | Handles noise better through selective breaching |
| **River flow issues** | `spec` | Fixes fragmentation and improper drainage |
| **Legacy compatibility** | `legacy` | Matches previous behavior (morphological reconstruction) |
| **Experimentation** | `pysheds` | Third-party library (may produce cycles) |

### Migration from Legacy

Legacy parameters map to spec parameters as follows:

| Legacy | Spec Equivalent | Notes |
|--------|----------------|-------|
| `fill_method="fill"` | `epsilon=0.0` | Complete flat fill |
| `fill_method="breach"` | `epsilon=1e-4` | Small gradients |
| `ocean_elevation_threshold` | `coastal_elev_threshold` | Same concept |
| `min_basin_size` | `max_breach_length` | Indirectly controls preservation |
| `min_basin_depth` | `max_breach_depth` | Direct equivalent |
| `fill_small_sinks` | *(removed)* | Breaching eliminates need |

## Quick Start: Test the Bug Fix and New Parameters

### 1. Test with Default Parameters (Baseline)

```bash
python examples/validate_flow_complete.py --bigness medium
```

This runs with the original parameters optimized for the San Diego coastal DEM.

### 2. Test with High-Resolution Parameters (Recommended for Noisy DEMs)

```bash
python examples/validate_flow_complete.py --bigness medium --high-res-params
```

This applies the recommended parameter set for high-resolution noisy DEMs:
- `fill_method: fill` (complete filling instead of breach)
- `min_basin_depth: 2.0m` (preserve basins >2m deep instead of >100m)
- `min_basin_size: 10000` (preserve basins >10k cells instead of >50k)
- `fill_small_sinks: 50` (fill remaining sinks <50 cells)

### 3. Compare the Results

The script generates visualizations showing:
- **Drainage area** - Should show smooth, continuous flow patterns (not noise)
- **Stream networks** - Should show realistic dendritic patterns
- **Validation summary** - Check for:
  - Cycles: 0 (no circular flow)
  - Mass balance: ~100% (water conserved)
  - Drainage violations: 0 (flow increases downstream)

Look for differences in:
- Number of outlets (fewer = better convergence)
- Stream network continuity (smooth vs fragmented)
- Drainage area noise (smooth gradients vs bright spots)

## Custom Parameter Testing

### Test Different Fill Methods

```bash
# Complete filling (stronger gradients, better for flat terrain)
python examples/validate_flow_complete.py --bigness medium \
    --fill-method fill

# Breach method (minimal filling, preserves more natural features)
python examples/validate_flow_complete.py --bigness medium \
    --fill-method breach
```

### Test Different Basin Depth Thresholds

```bash
# Aggressive filling (fill almost everything)
python examples/validate_flow_complete.py --bigness medium \
    --min-basin-depth 0.5

# Moderate (recommended for noisy high-res DEMs)
python examples/validate_flow_complete.py --bigness medium \
    --min-basin-depth 2.0

# Conservative (only preserve very deep basins)
python examples/validate_flow_complete.py --bigness medium \
    --min-basin-depth 100.0
```

### Test Small Sink Filling

```bash
# Fill small remaining sinks to reduce fragmentation
python examples/validate_flow_complete.py --bigness medium \
    --fill-small-sinks 50
```

### Full Custom Configuration

```bash
python examples/validate_flow_complete.py \
    --bigness large \
    --fill-method fill \
    --min-basin-depth 2.0 \
    --min-basin-size 10000 \
    --fill-small-sinks 50 \
    --output examples/output/flow_custom_test
```

## Testing at Different Scales

```bash
# Small test (200×200) - Fast iteration
python examples/validate_flow_complete.py --bigness small --high-res-params

# Medium test (500×500) - Good balance
python examples/validate_flow_complete.py --bigness medium --high-res-params

# Large test (1000×1000) - More realistic
python examples/validate_flow_complete.py --bigness large --high-res-params

# Full DEM at lower resolution (~1000×1000)
python examples/validate_flow_complete.py --bigness full --high-res-params

# Full DEM at higher resolution (~2000×2000) - Slow but thorough
python examples/validate_flow_complete.py --bigness full --target-size 2000 --high-res-params
```

## Interpreting Results

### Good Results (What to Look For)

✅ **Cycles: 0** - No circular flow paths
✅ **Mass Balance: ~100%** - Water is conserved
✅ **Drainage violations: 0** - Flow increases downstream
✅ **Stream networks show dendritic patterns** - Natural branching
✅ **Drainage area is smooth** - No isolated bright spots
✅ **Few outlets** - Most flow converges to main channels

### Problem Indicators

❌ **Many cycles** - Flow direction algorithm has bugs
❌ **Mass balance << 100%** - Water is disappearing (fragmentation)
❌ **Drainage violations > 0** - Flow doesn't increase downstream (bug!)
❌ **Stream networks fragmented** - Many disconnected segments
❌ **Drainage area noisy** - Random bright spots everywhere
❌ **Many outlets** - Flow doesn't converge (excessive fragmentation)

### Example: Comparing Parameters

```bash
# Test 1: Default (conservative - good for coastal DEMs with real deep basins)
python examples/validate_flow_complete.py --bigness medium \
    --output examples/output/test1_default

# Test 2: High-res preset (aggressive - good for noisy high-res inland DEMs)
python examples/validate_flow_complete.py --bigness medium \
    --high-res-params \
    --output examples/output/test2_highres

# Compare the drainage area images:
# test1_default/06a_drainage_area_log.png
# test2_highres/06a_drainage_area_log.png
```

## NPM Script Integration

You can also use the npm scripts defined in `package.json`:

```bash
# Default validation
npm run flow:validate:complete:full

# With custom arguments (pass after --)
npm run flow:validate:complete:full -- --high-res-params
npm run flow:validate:complete:full -- --fill-method fill --min-basin-depth 2.0
```

## Bug Fix Verification

The coastal flow direction bug fix should result in:
- **Fewer outlets** near coastlines
- **Smoother drainage** patterns near ocean boundaries
- **Better mass balance** (closer to 100%)
- **No drainage violations** near masked areas

Compare results before/after the fix to verify improvement.
