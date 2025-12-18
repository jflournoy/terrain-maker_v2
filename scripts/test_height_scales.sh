#!/bin/bash
#
# Test different height scales for terrain rendering
# Quick script to generate comparison renders with varying vertical exaggeration
#
# Usage:
#   ./scripts/test_height_scales.sh [camera_direction] [output_dir]
#
# Examples:
#   ./scripts/test_height_scales.sh south ./test_renders
#   ./scripts/test_height_scales.sh above ./overhead_tests

CAMERA_DIR="${1:-south}"
OUTPUT_DIR="${2:-docs/images/height_scale_tests}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Height scales to test (you can adjust these)
HEIGHTS=(10 15 20 25 30 35 40 45 50)

echo "════════════════════════════════════════════════════════════════"
echo "Testing Height Scales - Camera Direction: $CAMERA_DIR"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Height scales to test: ${HEIGHTS[*]}"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

for HEIGHT in "${HEIGHTS[@]}"; do
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "Rendering: height_scale=$HEIGHT"
    echo "────────────────────────────────────────────────────────────────"

    # Create subfolder for this test
    TEST_DIR="$OUTPUT_DIR/height_${HEIGHT}"
    mkdir -p "$TEST_DIR"

    # Run render (standard quality for speed)
    python examples/detroit_combined_render.py \
        --camera-direction "$CAMERA_DIR" \
        --height-scale "$HEIGHT" \
        --output-dir "$TEST_DIR" \
        2>&1 | tee "$TEST_DIR/render.log"

    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ Height scale $HEIGHT complete"
        # Rename output for clarity
        if [ -f "$TEST_DIR/sledding_with_xc_parks_3d.png" ]; then
            mv "$TEST_DIR/sledding_with_xc_parks_3d.png" \
               "$OUTPUT_DIR/height_${HEIGHT}_${CAMERA_DIR}.png"
            echo "  Saved: $OUTPUT_DIR/height_${HEIGHT}_${CAMERA_DIR}.png"
        fi
    else
        echo "✗ Height scale $HEIGHT failed"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ All renders complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Results in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null || echo "No PNG files found"
echo ""
echo "To create a comparison grid, you can use ImageMagick:"
echo "  montage $OUTPUT_DIR/height_*_${CAMERA_DIR}.png -geometry 800x600+10+10 -tile 3x3 $OUTPUT_DIR/comparison_grid.png"
